"""Module định nghĩa hệ hàm phần thưởng (Reward Functions) cho thuật toán GRPO trong RLVR.
Triển khai đúng chuẩn nghiên cứu DeepSeek-R1:
1. Accuracy Reward (Verifiable Reward): Kiểm chứng đáp số chính xác dựa trên quy tắc tất định.
2. Format Reward: Ép mô hình tuân thủ cấu trúc tư duy <think>...</think> và trình bày câu trả lời.
"""

import re
from typing import Any, Dict, List, Optional, Union

# Thêm common vào sys.path nếu cần
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.metrics import (
    extract_answer_from_response,
    is_answer_correct,
    normalize_numeric_str,
    split_think_and_answer,
)


def _extract_completion_text(completion: Any) -> str:
    """Trích xuất chuỗi văn bản từ completion, hỗ trợ cả định dạng chuỗi và danh sách dict (chat)."""
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list):
        if completion and isinstance(completion[0], dict) and "content" in completion[0]:
            return completion[0]["content"]
        return " ".join(str(c) for c in completion)
    elif isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


def _align_target_list(targets: Any, num_completions: int) -> List[Any]:
    """Căn chỉnh danh sách ground_truth cho khớp với số lượng completions (khi num_generations G > 1)."""
    if not isinstance(targets, list):
        return [targets] * num_completions
    if len(targets) == num_completions:
        return targets
    if len(targets) > 0 and num_completions % len(targets) == 0:
        group_size = num_completions // len(targets)
        aligned = []
        for t in targets:
            aligned.extend([t] * group_size)
        return aligned
    # Fallback padding
    return targets + [targets[-1]] * max(0, num_completions - len(targets))


def accuracy_reward_func(
    prompts: List[Any],
    completions: List[Any],
    ground_truth: Optional[List[Any]] = None,
    **kwargs: Any,
) -> List[float]:
    """Hàm phần thưởng độ chính xác (Accuracy Reward / Verifiable Reward).
    
    Quy tắc:
    - Bắt buộc có thẻ đóng </think>: lời giải bị cắt giữa pha suy nghĩ nhận 0 điểm (như ThinkPrune),
      nếu không extract_answer_from_response sẽ lấy số cuối cùng trong phần nháp và vẫn có thể được thưởng,
      khiến mô hình không bị phạt khi suy nghĩ vượt ngân sách.
    - Trích xuất đáp số cuối cùng từ phần trả lời sau </think>.
    - So khớp với đáp án chuẩn.
    - Thưởng +2.0 điểm nếu ĐÚNG hoàn toàn; 0.0 điểm nếu SAI hoặc không trích xuất được.
    """
    gts = ground_truth or kwargs.get("answer") or kwargs.get("target") or []
    aligned_gts = _align_target_list(gts, len(completions))

    rewards = []
    for comp, gt in zip(completions, aligned_gts):
        text = _extract_completion_text(comp)
        gt_str = str(gt).strip() if gt is not None else ""

        if "</think>" not in text:
            rewards.append(0.0)
            continue

        # Trích xuất đáp án từ câu trả lời
        pred_ans = extract_answer_from_response(text)

        if pred_ans is not None and gt_str:
            if is_answer_correct(pred_ans, gt_str):
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward_func(
    prompts: List[Any],
    completions: List[Any],
    **kwargs: Any,
) -> List[float]:
    """Hàm phần thưởng định dạng (Format Reward).
    
    Quy tắc tuân thủ cấu trúc CoT của DeepSeek-R1:
    - Prompt đã kết thúc sẵn bằng '<think>'.
    - Completion phải chứa thẻ đóng '</think>'.
    - Không được lặp lại thẻ mở '<think>' thừa bên trong completion.
    - Phải có phần văn bản câu trả lời sau thẻ '</think>' (không được rỗng).
    
    Điểm thưởng:
    - Thỏa mãn cấu trúc chuẩn có </think> và có câu trả lời: +1.0 điểm.
    - Thưởng thêm +0.25 nếu có định dạng đáp án rõ ràng (\\boxed{} hoặc ####).
    - Vi phạm cấu trúc (không đóng thẻ </think>, lặp thẻ): 0.0 điểm.
    """
    rewards = []
    for comp in completions:
        text = _extract_completion_text(comp)
        score = 0.0

        has_close_think = "</think>" in text
        # Kiểm tra không lặp thẻ mở <think>
        no_duplicate_open = "<think>" not in text

        if has_close_think and no_duplicate_open:
            think_part, answer_part = split_think_and_answer(text)
            # Yêu cầu phần câu trả lời sau </think> không được rỗng
            if len(answer_part.strip()) > 0:
                score += 1.0

                # Thưởng khuyến khích định dạng xuất đáp án chuyên nghiệp
                if r"\boxed{" in answer_part or "####" in answer_part:
                    score += 0.25
        elif has_close_think:
            # Có thẻ đóng nhưng dính thẻ mở lặp lại
            score += 0.2
        else:
            # Không có thẻ đóng (bị cắt cụt hoặc quên đóng thẻ)
            score = 0.0

        rewards.append(score)

    return rewards


def brevity_reward_func(
    prompts: List[Any],
    completions: List[Any],
    ground_truth: Optional[List[Any]] = None,
    max_ideal_length: int = 250,
    **kwargs: Any,
) -> List[float]:
    """Hàm phần thưởng súc tích (Tùy chọn phụ trợ phục vụ nghiên cứu so sánh).
    Khuyến khích mô hình suy nghĩ cô đọng nếu đã giải đúng (giảm thiểu overthinking).
    Chỉ kích hoạt thưởng khi đáp án ĐÃ ĐÚNG.
    """
    gts = ground_truth or kwargs.get("answer") or []
    aligned_gts = _align_target_list(gts, len(completions))

    rewards = []
    for comp, gt in zip(completions, aligned_gts):
        text = _extract_completion_text(comp)
        gt_str = str(gt).strip() if gt is not None else ""
        pred_ans = extract_answer_from_response(text)

        if pred_ans is not None and is_answer_correct(pred_ans, gt_str):
            # Tách phần suy nghĩ để đếm ước lượng từ/token
            think_part, _ = split_think_and_answer(text)
            approx_tokens = len(think_part.split())
            if approx_tokens <= max_ideal_length:
                rewards.append(0.5)
            else:
                # Phạt nhẹ tuyến tính nếu suy nghĩ quá dài dòng
                excess = approx_tokens - max_ideal_length
                penalty = min(0.4, excess * 0.002)
                rewards.append(0.5 - penalty)
        else:
            rewards.append(0.0)

    return rewards


def get_reward_functions(include_brevity: bool = False) -> List[Any]:
    """Trả về danh sách các hàm phần thưởng cấu hình cho GRPOTrainer.
    
    Mặc định chuẩn DeepSeek-R1:
    - accuracy_reward_func (+2.0 điểm khi giải đúng)
    - format_reward_func (+1.0 đến +1.25 điểm khi chuẩn cấu trúc </think>)
    """
    funcs = [accuracy_reward_func, format_reward_func]
    if include_brevity:
        funcs.append(brevity_reward_func)
    return funcs
