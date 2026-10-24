"""Script đánh giá hiệu năng mô hình sau huấn luyện RLVR (GRPO + LoRA) trên 100 câu GSM8K chuẩn.
So sánh trực tiếp kết quả với Baseline gốc (Chưa can thiệp) và Phương pháp RPDI-EE.
"""

import argparse
import io
import json
import os
import sys
import time
from typing import Any, Dict, List

# Đảm bảo in ký tự tiếng Việt an toàn trên console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc của project vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.eval_dataset import load_gsm8k_test
from common.metrics import (
    compute_aggregate_metrics,
    is_answer_correct,
)
from common.model_loader import DEFAULT_MODEL_ID
from common.seed import set_seed
from method_c_rpdi_ee.runner import generate_standard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình sau khi huấn luyện RLVR (GRPO) trên tập kiểm thử GSM8K."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"ID mô hình base từ Hugging Face (mặc định: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="results/rlvr_adapter",
        help="Đường dẫn tới thư mục chứa LoRA adapter đã huấn luyện (mặc định: results/rlvr_adapter).",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Đường dẫn tùy chỉnh đến file JSON chứa 100 câu kiểm thử GSM8K.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số câu hỏi chạy thử nghiệm (ví dụ: 5 để kiểm tra nhanh).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=768,
        help="Số lượng token sinh mới tối đa (mặc định: 768).",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Bật Sampling (mặc định: False - Greedy Decoding để đảm bảo tính tất định).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/method_b_results.json",
        help="Đường dẫn lưu kết quả chi tiết JSON (mặc định: results/method_b_results.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên toàn cục.",
    )
    return parser.parse_args()


def load_rlvr_model(base_model_id: str, adapter_dir: str):
    """Nạp mô hình cơ sở và tích hợp LoRA adapter của RLVR."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[ModelLoader] Đang nạp tokenizer từ '{base_model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    print(f"[ModelLoader] Đang nạp mô hình nền '{base_model_id}' với dtype={dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if os.path.exists(adapter_dir):
        print(f"[ModelLoader] Đang tích hợp LoRA adapter RLVR từ '{adapter_dir}'...")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        print("[ModelLoader] Đã nạp thành công mô hình RLVR (Base + LoRA).")
    else:
        print(f"[Cảnh báo] Không tìm thấy adapter tại '{adapter_dir}'. Sử dụng mô hình Base gốc để đánh giá!")
        model = base_model

    model.eval()
    return model, tokenizer


def print_comparison_table(metrics: Dict[str, Any], baseline_file: str = "results/method_c_results.json"):
    """In bảng so sánh kết quả của RLVR với Baseline và RPDI-EE."""
    std_acc = 59.0
    std_time = 25.16
    std_tokens = 615.2
    std_think = 461.3

    rpdi_acc = 74.0
    rpdi_time = 22.15
    rpdi_tokens = 540.5
    rpdi_think = 95.6

    # Nếu có file kết quả của method C thì nạp số liệu thực tế
    if os.path.exists(baseline_file):
        try:
            with open(baseline_file, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                agg = c_data.get("aggregate_metrics", {})
                std_acc = agg.get("standard", {}).get("accuracy_pct", std_acc)
                std_time = agg.get("standard", {}).get("avg_latency_s", std_time)
                std_tokens = agg.get("standard", {}).get("avg_total_tokens", std_tokens)
                std_think = agg.get("standard", {}).get("avg_think_tokens", std_think)

                rpdi_acc = agg.get("rpdi_ee", {}).get("accuracy_pct", rpdi_acc)
                rpdi_time = agg.get("rpdi_ee", {}).get("avg_latency_s", rpdi_time)
                rpdi_tokens = agg.get("rpdi_ee", {}).get("avg_total_tokens", rpdi_tokens)
                rpdi_think = agg.get("rpdi_ee", {}).get("avg_think_tokens", rpdi_think)
        except Exception:
            pass

    cur_acc = metrics.get("accuracy_pct", 0.0)
    cur_time = metrics.get("avg_latency_s", 0.0)
    cur_tokens = metrics.get("avg_total_tokens", 0.0)
    cur_think = metrics.get("avg_think_tokens", 0.0)
    cur_ans = metrics.get("avg_answer_tokens", 0.0)

    delta_acc_std = cur_acc - std_acc

    print("\n" + "=" * 105)
    print("                 BẢNG SO SÁNH HIỆU NĂNG: BASELINE GỐC vs RPDI-EE vs RLVR (GRPO)                 ")
    print("=" * 105)
    print(f"{'Chỉ số đánh giá':<28} | {'Baseline Gốc':<16} | {'RPDI-EE (Phanh)':<16} | {'RLVR (GRPO)':<16} | {'Δ vs Baseline':<16}")
    print("-" * 105)
    print(f"{'Độ chính xác (Accuracy)':<28} | {std_acc:>13.2f}% | {rpdi_acc:>13.2f}% | {cur_acc:>13.2f}% | {delta_acc_std:>+13.2f}%")
    print(f"{'Thời gian TB mỗi câu (s)':<28} | {std_time:>14.2f}s | {rpdi_time:>14.2f}s | {cur_time:>14.2f}s | {cur_time - std_time:>+14.2f}s")
    print(f"{'Tổng token trung bình':<28} | {std_tokens:>14.1f}  | {rpdi_tokens:>14.1f}  | {cur_tokens:>14.1f}  | {cur_tokens - std_tokens:>+14.1f} ")
    print(f"{'Token suy nghĩ (<think>)':<28} | {std_think:>14.1f}  | {rpdi_think:>14.1f}  | {cur_think:>14.1f}  | {cur_think - std_think:>+14.1f} ")
    print(f"{'Token câu trả lời':<28} | {153.9:>14.1f}  | {444.9:>14.1f}  | {cur_ans:>14.1f}  | {cur_ans - 153.9:>+14.1f} ")
    print("=" * 105)


def run_evaluation():
    args = parse_args()
    set_seed(args.seed)

    print(f"[Dữ liệu] Đang tải tập dữ liệu GSM8K kiểm thử...")
    dataset = load_gsm8k_test(filepath=args.data_file, limit=args.limit)
    total_samples = len(dataset)
    print(f"[Dữ liệu] Đã tải thành công {total_samples} mẫu kiểm thử.")

    model, tokenizer = load_rlvr_model(args.model_id, args.adapter_dir)

    print(f"\n[Thực nghiệm] Bắt đầu đánh giá mô hình RLVR trên {total_samples} câu:")
    print(f" - Chiến lược giải mã: {'Sampling' if args.do_sample else 'Greedy Decoding (do_sample=False)'}")
    print(f" - Giới hạn max_new_tokens: {args.max_new_tokens}")
    print("-" * 90)

    sample_records = []
    start_total_time = time.time()

    for idx, sample in enumerate(dataset):
        s_id = sample.get("id", idx)
        question = sample["question"]
        gt = sample["ground_truth"]

        print(f"\n👉 [Câu {idx + 1}/{total_samples}] (ID: {s_id})")
        print(f"   Đề bài: \"{question[:80]}...\"")
        print(f"   Đáp án chuẩn: {gt}")

        res = generate_standard(
            model=model,
            tokenizer=tokenizer,
            question=question,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            seed=args.seed,
        )

        is_corr = is_answer_correct(res["predicted_answer"], gt)
        res["is_correct"] = is_corr
        status_str = "ĐÚNG" if is_corr else "SAI"

        print(
            f"   [RLVR-GRPO]: Dự đoán='{res['predicted_answer']}' [{status_str}] | "
            f"{res['total_tokens']} tokens ({res['think_tokens']} suy nghĩ) | {res['latency']:.2f}s"
        )

        sample_records.append({
            "sample_id": s_id,
            "question": question,
            "ground_truth": gt,
            "rlvr": res,
        })

    total_experiment_time = time.time() - start_total_time

    # Tính toán chỉ số tổng hợp
    rlvr_metrics = compute_aggregate_metrics([r["rlvr"] for r in sample_records])

    # In bảng so sánh
    print_comparison_table(rlvr_metrics)
    print(f"\nTổng thời gian thực nghiệm: {total_experiment_time:.2f} giây.")

    # Lưu kết quả chi tiết
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    full_output = {
        "metadata": {
            "model_id": args.model_id,
            "adapter_dir": args.adapter_dir,
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_new_tokens,
            "num_samples": total_samples,
            "total_experiment_time_s": round(total_experiment_time, 2),
        },
        "aggregate_metrics": rlvr_metrics,
        "sample_records": sample_records,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)

    print(f"[Kết quả] Đã lưu kết quả chi tiết thành công vào: {args.output_file}")


if __name__ == "__main__":
    run_evaluation()
