"""Bộ điều khiển thực thi sinh văn bản cho chế độ Chuẩn (Standard) và RPDI-EE.
Hỗ trợ giải mã Greedy (do_sample=False) và Sampling ngẫu nhiên, theo dõi số lượng token và thời gian thực thi.
"""

import time
from typing import Any, Dict, Optional
import torch
from transformers import LogitsProcessorList

from common.metrics import (
    extract_answer_from_response,
    split_think_and_answer,
)
from common.model_loader import format_reasoning_prompt
from common.seed import set_seed
from .logits_processor import RPDILogitsProcessor


def generate_standard(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Chạy sinh văn bản ở chế độ cơ sở (Standard baseline), không can thiệp phanh suy luận.

    Tham số:
        model: Mô hình ngôn ngữ CausalLM.
        tokenizer: PreTrainedTokenizer tương ứng.
        question: Câu hỏi toán học / suy luận đầu vào.
        max_new_tokens: Giới hạn tổng số token được sinh mới.
        do_sample: False cho giải mã Greedy (tái lập kết quả / benchmark chuẩn), True cho sampling ngẫu nhiên.
        temperature: Nhiệt độ sampling (bỏ qua nếu do_sample=False).
        top_p: Xác suất tích lũy Top-p (bỏ qua nếu do_sample=False).
        seed: Seed ngẫu nhiên.

    Trả về:
        Dictionary chứa toàn bộ văn bản sinh ra, thời gian xử lý, số token và đáp án trích xuất được.
    """
    if seed is not None:
        set_seed(seed)

    prompt = format_reasoning_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    latency = time.time() - start_time

    generated_ids = outputs[0][prompt_len:]
    total_tokens = len(generated_ids)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    think_text, answer_text = split_think_and_answer(response_text)
    think_tokens = len(tokenizer.encode(think_text, add_special_tokens=False)) if think_text else 0
    answer_tokens = max(0, total_tokens - think_tokens)

    pred_answer = extract_answer_from_response(response_text)

    return {
        "response_text": response_text,
        "think_text": think_text,
        "answer_text": answer_text,
        "predicted_answer": pred_answer,
        "latency": latency,
        "total_tokens": total_tokens,
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "early_exited": False,
        "exit_step": -1,
        "rpdi_at_exit": 0.0,
    }


def generate_rpdi(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 512,
    W: int = 8,
    lambda_th: float = 1.2,
    min_steps: Optional[int] = None,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Chạy sinh văn bản kết hợp RPDILogitsProcessor trong một lệnh generate() duy nhất.

    Tham số:
        model: Mô hình ngôn ngữ CausalLM.
        tokenizer: PreTrainedTokenizer tương ứng.
        question: Câu hỏi toán học / suy luận đầu vào.
        max_new_tokens: Giới hạn tổng số token được sinh mới.
        W: Kích thước cửa sổ trượt tính entropy cục bộ.
        lambda_th: Ngưỡng phát hiện lệch hướng suy luận.
        min_steps: Số bước suy nghĩ tối thiểu trước khi phanh có hiệu lực.
        do_sample: False cho giải mã Greedy, True cho sampling ngẫu nhiên.
        temperature: Nhiệt độ sampling.
        top_p: Xác suất tích lũy Top-p.
        seed: Seed ngẫu nhiên.

    Trả về:
        Dictionary chứa văn bản sinh ra, thời gian xử lý, số token, chỉ số dừng sớm và đáp án trích xuất.
    """
    if seed is not None:
        set_seed(seed)

    prompt = format_reasoning_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    processor = RPDILogitsProcessor(
        tokenizer=tokenizer,
        W=W,
        lambda_th=lambda_th,
        min_steps=min_steps,
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "logits_processor": LogitsProcessorList([processor]),
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    latency = time.time() - start_time

    generated_ids = outputs[0][prompt_len:]
    total_tokens = len(generated_ids)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    think_text, answer_text = split_think_and_answer(response_text)
    think_tokens = len(tokenizer.encode(think_text, add_special_tokens=False)) if think_text else 0
    answer_tokens = max(0, total_tokens - think_tokens)

    pred_answer = extract_answer_from_response(response_text)

    return {
        "response_text": response_text,
        "think_text": think_text,
        "answer_text": answer_text,
        "predicted_answer": pred_answer,
        "latency": latency,
        "total_tokens": total_tokens,
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "early_exited": processor.early_exited,
        "exit_step": processor.force_exit_step,
        "rpdi_at_exit": processor.rpdi_at_exit,
    }
