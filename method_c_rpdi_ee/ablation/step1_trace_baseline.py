"""Bước 1: Thu thập toàn bộ vết suy luận (Traces) của mô hình ở chế độ Standard Baseline (Greedy).
Ghi lại chuỗi token, phân bố entropy H(t_i) tại mỗi bước và vị trí các token biên câu.
File kết quả được lưu vào results/baseline_traces.json để phục vụ mô phỏng offline ở Bước 2.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

# Đảm bảo in ký tự tiếng Việt an toàn trên console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.eval_dataset import load_gsm8k_test
from common.metrics import (
    extract_answer_from_response,
    is_answer_correct,
    split_think_and_answer,
)
from common.model_loader import DEFAULT_MODEL_ID, format_reasoning_prompt, load_model_and_tokenizer
from common.seed import set_seed


class TraceRecorderLogitsProcessor(LogitsProcessor):
    """LogitsProcessor ghi lại chuỗi entropy của từng token trong quá trình sinh giải mã."""

    def __init__(self):
        super().__init__()
        self.entropies: List[float] = []

    def reset(self) -> None:
        self.entropies.clear()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores[0] là logits dự đoán token tiếp theo
        logits = scores[0]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy_val = -torch.sum(probs * log_probs, dim=-1).item()
        self.entropies.append(entropy_val)
        return scores


def get_boundary_token_ids(tokenizer, boundary_symbols: Optional[Set[str]] = None) -> Set[int]:
    """Lấy danh sách Token ID tương ứng với các ký tự phân cách ranh giới câu."""
    if boundary_symbols is None:
        boundary_symbols = {".", "\n", ";", "?", "!", ","}

    boundary_ids: Set[int] = set()
    for sym in boundary_symbols:
        ids = tokenizer.encode(sym, add_special_tokens=False)
        boundary_ids.update(ids)
        ids_space = tokenizer.encode(" " + sym, add_special_tokens=False)
        if ids_space:
            boundary_ids.add(ids_space[-1])
    return boundary_ids


def find_natural_think_end(generated_ids: List[int], tokenizer) -> Optional[int]:
    """Tìm chỉ số token kết thúc pha suy nghĩ tự nhiên (thẻ </think>) nếu có."""
    close_think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    target_len = len(close_think_tokens)
    if target_len == 0:
        return None

    for i in range(len(generated_ids) - target_len + 1):
        if generated_ids[i : i + target_len] == close_think_tokens:
            return i  # Chỉ số token bắt đầu xuất hiện </think>
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bước 1: Chạy Standard Baseline và thu thập vết entropy phục vụ mô phỏng offline."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"ID mô hình Hugging Face (mặc định: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Đường dẫn file dữ liệu GSM8K kiểm thử (mặc định: common/data/gsm8k_test_100.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Số lượng mẫu tối đa cần chạy (ví dụ 10 hoặc 20 để test nhanh; mặc định chạy toàn bộ).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=768,
        help="Giới hạn số token sinh tối đa (mặc định: 768)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên (mặc định: 42)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "baseline_traces.json"),
        help="Đường dẫn lưu file JSON chứa các vết suy luận (mặc định: results/baseline_traces.json)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="Kiểu dữ liệu trọng số (auto, bfloat16, float16)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 85)
    print("🚀 BƯỚC 1: CHẠY STANDARD BASELINE VÀ THU THẬP VẾT SUY LUẬN (GREEDY DECODING)")
    print("=" * 85)

    # 1. Tải tập dữ liệu
    dataset = load_gsm8k_test(filepath=args.data_file, limit=args.limit)
    num_samples = len(dataset)
    print(f"[*] Đã tải {num_samples} mẫu câu hỏi từ tập dữ liệu.")

    # 2. Tải mô hình và tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        torch_dtype=args.torch_dtype,
    )
    boundary_token_ids = get_boundary_token_ids(tokenizer)
    recorder = TraceRecorderLogitsProcessor()

    traces_list: List[Dict[str, Any]] = []
    correct_count = 0
    total_tokens_accum = 0
    total_think_tokens_accum = 0
    start_total_time = time.time()

    print(f"[*] Bắt đầu sinh văn bản và ghi vết entropy...")
    for idx, sample in enumerate(dataset):
        sample_id = sample.get("id", idx)
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        prompt = format_reasoning_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        prompt_token_ids = inputs["input_ids"][0].tolist()

        recorder.reset()
        gen_start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                logits_processor=LogitsProcessorList([recorder]),
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=False,
            )
        gen_latency = time.time() - gen_start_time

        output_tokens = outputs[0].tolist()
        generated_token_ids = output_tokens[prompt_len:]
        total_gen_len = len(generated_token_ids)
        response_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

        # Phân tách phần think và answer tự nhiên
        think_text, answer_text = split_think_and_answer(response_text)
        think_tokens = len(tokenizer.encode(think_text, add_special_tokens=False)) if think_text else 0
        answer_tokens = max(0, total_gen_len - think_tokens)
        natural_think_end_idx = find_natural_think_end(generated_token_ids, tokenizer)

        pred_answer = extract_answer_from_response(response_text)
        is_correct = is_answer_correct(pred_answer, ground_truth)
        if is_correct:
            correct_count += 1

        total_tokens_accum += total_gen_len
        total_think_tokens_accum += think_tokens

        # Lưu vết cho câu hỏi hiện tại
        record = {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_token_ids,
            "entropies": recorder.entropies,
            "natural_think_end_idx": natural_think_end_idx,
            "standard_metrics": {
                "predicted_answer": pred_answer,
                "is_correct": is_correct,
                "latency_s": round(gen_latency, 3),
                "total_tokens": total_gen_len,
                "think_tokens": think_tokens,
                "answer_tokens": answer_tokens,
            },
            "response_text": response_text,
        }
        traces_list.append(record)

        status_symbol = "✅ ĐÚNG" if is_correct else "❌ SAI"
        print(
            f"  [{idx + 1:3d}/{num_samples}] ID: {sample_id:<3} | {status_symbol} | "
            f"Tổng: {total_gen_len:3d} tok (Think: {think_tokens:3d}) | "
            f"Thời gian: {gen_latency:5.2f}s | Đáp án: {pred_answer}"
        )

        # Xóa cache GPU để giữ bộ nhớ ổn định
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_duration = time.time() - start_total_time
    avg_accuracy = (correct_count / num_samples) * 100.0
    avg_total_tokens = total_tokens_accum / num_samples
    avg_think_tokens = total_think_tokens_accum / num_samples

    print("\n" + "=" * 85)
    print("📊 TỔNG KẾT STANDARD BASELINE:")
    print(f" - Tổng số mẫu: {num_samples}")
    print(f" - Độ chính xác (Accuracy): {correct_count}/{num_samples} ({avg_accuracy:.2f}%)")
    print(f" - Số token trung bình: {avg_total_tokens:.1f} (Suy nghĩ: {avg_think_tokens:.1f})")
    print(f" - Tổng thời gian chạy: {total_duration:.1f}s ({total_duration / 60:.2f} phút)")
    print("=" * 85)

    # Đóng gói và lưu kết quả
    output_data = {
        "metadata": {
            "model_id": args.model_id,
            "decoding_mode": "Greedy (do_sample=False)",
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "num_samples": num_samples,
            "boundary_token_ids": sorted(list(boundary_token_ids)),
            "summary": {
                "accuracy_pct": avg_accuracy,
                "avg_total_tokens": avg_total_tokens,
                "avg_think_tokens": avg_think_tokens,
            },
        },
        "traces": traces_list,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Đã lưu toàn bộ vết suy luận thành công vào: {args.output_file}")


if __name__ == "__main__":
    main()
