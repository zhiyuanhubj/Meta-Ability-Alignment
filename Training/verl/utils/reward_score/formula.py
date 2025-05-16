# file: compute_reward.py

import re
from typing import Dict, Tuple, Optional

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer (<answer>...</answer>) from the model's response string."""
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into { var_name: 'true'/'false' } dictionary.
    
    假设 solution_text_format:
      (1) A is false
      (2) B is false
      (3) C is false
      (4) D is true
    或者有些人写 "A is false", "B is true" 等。
    我们就用一个正则:  \b(\w+)\b.*?\bis\s+(true|false)\b
    
    解析得到: { "A":"false", "B":"false", "C":"false", "D":"true" }
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")
    
    # 按行分割
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # 改成匹配 something like "A is true" or "A is false"
        match = re.search(r'\b([A-Za-z]+)\b\s+is\s+(True|False)\b', line, re.IGNORECASE)
        if match:
            name, val = match.groups()
            val_lower = val.lower()
            status_dict[name] = val_lower
            print(f"  Found: {name} → {val_lower}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's <answer> into { var_name: 'true'/'false' } dict.
    
    1) 先统计出现了多少个 'true' 或 'false'，若没达到期望数量，就报错 None
    2) 再逐一匹配 "A is true/false" 格式。
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    # 计数
    true_count = len(re.findall(r'\bis\s+True\b', answer_text, re.IGNORECASE))
    false_count = len(re.findall(r'\bis\s+False\b', answer_text, re.IGNORECASE))
    total_role_count = true_count + false_count

    print(f"  Number of predicted assignments: {total_role_count}")
    if total_role_count != len(expected_names):
        print(f"  [Error] Number of variables mismatch: {total_role_count} != {len(expected_names)}")
        return None

    # 依次检查每个变量
    for name in expected_names:
        pattern = re.compile(
            rf'\b{name}\b\s+is\s+(True|False)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            val = match.group(1).lower()
            status_dict[name] = val
            print(f"  Found: {name} → {val}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure: <think>, </think>, <answer>, </answer>."""
    print("\n[Structure Validation]")
    validation_passed = True

    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={positions[tag_name]}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if not (positions['think_start'] < positions['think_end'] < positions['answer_start'] < positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def compute_score(solution_str: str, 
                  ground_truth: Dict[str, str],
                  format_reward: int = 1,
                  answer_reward: float = 1.0) -> float:
    """
    计算综合分数:
      1) 检查标签格式 <think>...</think><answer>...</answer>
         - 通过 => +format_reward，不通过 => -format_reward
      2) 内容比对
         - 若解析成功且与 gt 完全匹配 => +2
         - 否则 => -1.5 或者 -2（可自己改）
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # ground_truth 里形如 { 'solution_text_format': '(1) A is false\n(2) B is true\n...' }
    solution_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final assignment: {gt_status}")

    # 1) Extract <answer>...</answer>
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    # 2) Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # 3) Validate answer content
    answer_score = 0
    if format_correct and answer_text:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            
            if pred_status == gt_status:
                answer_score = 2
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -1.5
                print("  Content validation: MISMATCH")
        else:
            answer_score = -2
            print("Fail to parse answer")
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score

