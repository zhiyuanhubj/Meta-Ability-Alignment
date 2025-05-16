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


def parse_ground_truth(ground_truth: Dict[str, any]) -> int:
    """
    从 ground_truth 数据里解析“正确答案”的数值。
    ground_truth 可能长这样:
    {
        "problem_id": "1-seq-15",
        "difficulty": 1,
        "puzzle_text": "...",
        "solution_text": 16,
        "ability": "sequence",
        "complete_sequence": [1,2,4,8,16]
    }
    
    我们只关心 solution_text（即正确答案）。
    """
    # 这里假设 solution_text 就是一个整数
    print(ground_truth)
    correct_answer = ground_truth.get('solution_text_format')
    print(correct_answer)
    print(type(correct_answer))
    print('checkpoint here')
    if not isinstance(correct_answer, int):
        print("[Warning] ground_truth['solution_text'] 不存在或不是整数，默认设为 None")
        return None
    return correct_answer


def parse_model_answer(answer_text: str) -> Optional[int]:
    """
    从模型的 <answer> 内容里解析一个数值。
    例子:
       <answer>16</answer>
    我们就把它转成 int(16)
    如果解析失败，就返回 None。
    """
    print("\n[Model Answer Parsing]")
    
    try:
        predicted_value = int(answer_text.strip())
        print(f"  Model answer parsed as integer: {predicted_value}")
        return predicted_value
    except ValueError:
        print("  [Error] Failed to parse model's answer as integer.")
        return None


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
    # 如果你对标签顺序没那么严苛，可以放松此判断
    if not (positions['think_start'] < positions['think_end'] < positions['answer_start'] < positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def compute_score(solution_str: str, 
                  ground_truth: Dict[str, any],
                  format_reward: int = 1,
                  answer_reward: float = 1.0) -> float:
    """
    计算综合分数:
      1) 检查标签格式 <think>...</think><answer>...</answer>
         - 通过 => +format_reward，不通过 => -format_reward
      2) 内容比对 (这里是数字答案比对)
         - 若解析成功且与 gt 完全匹配 => +2
         - 否则 => -2
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # 1) Ground truth 中的正确数值
    correct_answer = parse_ground_truth(ground_truth)
    print(f"[Ground Truth] solution_text = {correct_answer}")

    # 2) 提取 <answer>...</answer>
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    # 3) 验证格式 <think>...</think><answer>...</answer>
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # 4) 验证模型给的答案内容
    answer_score = 0
    if format_correct and answer_text is not None:
        pred_value = parse_model_answer(answer_text)
        if pred_value is not None and correct_answer is not None:
            if pred_value == correct_answer:
                answer_score = 2
                print("  Content validation: FULL MATCH (pred == gt)")
            else:
                answer_score = -1.5
                print(f"  Content validation: MISMATCH (pred={pred_value}, gt={correct_answer})")
        else:
            answer_score = -2
            print("  [Error] Failed to parse model answer or missing ground truth")
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing <answer>")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score
