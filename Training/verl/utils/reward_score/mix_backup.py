# file: compute_reward.py

import re

from verl.utils.reward_score import formula, backward_reasoning
from typing import Dict, Tuple, Optional

def compute_score(solution_str: str, 
                  ground_truth: Dict[str, str],
                  format_reward: int = 1,
                  answer_reward: float = 1.0) -> float:
    # 将 ground_truth 的所有值拼接成一个字符串
    ground_truth_str = ground_truth.get('solution_text_format', '').lower()  # 转换为小写，避免大小写影响匹配

    if "true" in ground_truth_str or "false" in ground_truth_str:
        return formula.compute_score(solution_str, ground_truth, format_reward, answer_reward)
    
    elif "reachable" in ground_truth_str or "unreachable" in ground_truth_str:
        return backward_reasoning.compute_score(solution_str, ground_truth, format_reward, answer_reward)
    
    else:
        raise ValueError(f"Unknown ground truth format: {ground_truth}. Expected 'true/false' or 'reachable/unreachable' in values.")
