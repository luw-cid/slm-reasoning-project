"""Suy ra kết quả ở mọi ngân sách B từ một lượt sinh duy nhất ở ngân sách lớn nhất.

Với giải mã greedy, quyết định phanh của RPDI-EE (và của baseline) chỉ phụ thuộc phần tiền tố đã
sinh, không phụ thuộc ngân sách; nên lượt chạy ở ngân sách B chính là B token đầu của lượt chạy 16k.
Đáp án được chấm trên ``decode(prompt_ids + output_ids[:B])``, giống ``run_controls.py`` (HF).
"""

import hashlib
from typing import Any, Callable, Dict, List, Sequence

from common.metrics import extract_answer_from_response, is_answer_correct


def expand_to_budgets(
    raw: Dict[str, Any],
    decode: Callable[[List[int]], str],
    think_close_id: int,
    budgets: Sequence[int],
) -> List[Dict[str, Any]]:
    """Một bản ghi (định dạng run_controls.py) cho mỗi ngân sách B <= raw["max_tokens"]."""
    output_ids = list(raw["output_ids"])
    close_index = raw.get("close_index")
    records = []
    for B in sorted(set(budgets)):
        if B > raw["max_tokens"]:
            continue
        ids = output_ids[:B]
        closed_within = close_index is not None and close_index < B
        think_tokens = close_index if closed_within else len(ids)
        exited = bool(raw.get("forced_close")) and closed_within
        pred = extract_answer_from_response(decode(list(raw["prompt_ids"]) + ids))
        records.append({
            "sample_id": raw["sample_id"],
            "method": raw["method"],
            "max_new_tokens": B,
            "ground_truth": raw["ground_truth"],
            "predicted_answer": pred,
            "is_correct": bool(is_answer_correct(pred, raw["ground_truth"])),
            "total_tokens": len(ids),
            "think_tokens": think_tokens,
            "answer_tokens": len(ids) - think_tokens,
            "early_exited": exited,
            "exit_step": raw["exit_step"] if exited else -1,
            "rpdi_at_exit": raw.get("rpdi_at_exit", 0.0) if exited else 0.0,
            "hit_budget": len(ids) >= B,
            "response_sha1": hashlib.sha1(decode(ids).encode("utf-8")).hexdigest(),
            "engine": "vllm",
            "source_max_tokens": raw["max_tokens"],
        })
    return records
