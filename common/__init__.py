"""Các tiện ích dùng chung cho đồ án nâng cao năng lực suy luận của SLM.
Bao gồm quản lý seed ngẫu nhiên, nạp dữ liệu kiểm thử GSM8K, khởi tạo mô hình và các chỉ số đo lường suy luận.
"""

from .seed import set_seed
from .eval_dataset import load_gsm8k_test, get_sample_by_id
from .metrics import (
    extract_answer_from_response,
    is_answer_correct,
    split_think_and_answer,
    compute_aggregate_metrics,
)
from .model_loader import load_model_and_tokenizer, format_reasoning_prompt

__all__ = [
    "set_seed",
    "load_gsm8k_test",
    "get_sample_by_id",
    "extract_answer_from_response",
    "is_answer_correct",
    "split_think_and_answer",
    "compute_aggregate_metrics",
    "load_model_and_tokenizer",
    "format_reasoning_prompt",
]
