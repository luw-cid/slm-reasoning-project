"""Script thực nghiệm tổng hợp đối chứng 4 kịch bản nghiên cứu của Đề tài Tốt nghiệp:
1. Kịch bản 1: Baseline gốc (Chưa can thiệp)
2. Kịch bản 2: Baseline + RPDI-EE (Can thiệp lúc suy luận - Nhánh C)
3. Kịch bản 3: RLVR (Học tăng cường nội vi qua GRPO - Nhánh B)
4. Kịch bản 4: RLVR + RPDI-EE (Kết hợp đồng thời Nhánh B và Nhánh C)

Đánh giá trên cùng 100 câu kiểm thử GSM8K cố định nhằm đảm bảo tính công bằng và nhất quán khoa học tuyệt đối.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# Đảm bảo in tiếng Việt an toàn
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.eval_dataset import load_gsm8k_test
from common.metrics import compute_aggregate_metrics, is_answer_correct
from common.model_loader import DEFAULT_MODEL_ID
from common.seed import set_seed
from method_c_rpdi_ee.runner import generate_rpdi, generate_standard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thực nghiệm tổng hợp 4 kịch bản đối chứng trên tập kiểm thử GSM8K."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"ID mô hình cơ sở Hugging Face (mặc định: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="results/rlvr_adapter",
        help="Đường dẫn chứa trọng số LoRA adapter của mô hình RLVR.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số mẫu thử nghiệm (ví dụ: 5 để test nhanh, để trống để chạy 100 câu).",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Đường dẫn file dữ liệu test (mặc định: common/data/gsm8k_test_100.json).",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=8,
        help="Cửa sổ trượt W của RPDI-EE (mặc định: 8).",
    )
    parser.add_argument(
        "--lambda_th",
        type=float,
        default=1.2,
        help="Ngưỡng kích hoạt phanh lambda của RPDI-EE (mặc định: 1.2).",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=64,
        help="Số bước suy nghĩ tối thiểu trước khi phanh (mặc định: 64).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=768,
        help="Giới hạn số token sinh tối đa (mặc định: 768).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/combined_4_scenarios.json",
        help="Đường dẫn lưu kết quả tổng hợp JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên toàn cục.",
    )
    return parser.parse_args()


def load_models(base_model_id: str, adapter_dir: str):
    """Nạp cả mô hình gốc và mô hình tích hợp LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[ModelLoader] Đang nạp tokenizer từ '{base_model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    print(f"[ModelLoader] Đang nạp mô hình Base '{base_model_id}'...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    base_model.eval()

    rlvr_model = None
    if os.path.exists(adapter_dir):
        print(f"[ModelLoader] Đang nạp mô hình RLVR với adapter từ '{adapter_dir}'...")
        rlvr_model = PeftModel.from_pretrained(base_model, adapter_dir)
        rlvr_model.eval()
    else:
        print(f"[Cảnh báo] Chưa tìm thấy LoRA adapter tại '{adapter_dir}'. Kịch bản 3 và 4 sẽ dùng mô hình base tạm thời!")
        rlvr_model = base_model

    return base_model, rlvr_model, tokenizer


def print_4_scenarios_table(
    s1_metrics: Dict[str, Any],
    s2_metrics: Dict[str, Any],
    s3_metrics: Dict[str, Any],
    s4_metrics: Dict[str, Any],
):
    """In bảng tổng kết so sánh 4 kịch bản chuẩn định dạng báo cáo khoa học."""
    print("\n" + "=" * 120)
    print("                      BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG 4 KỊCH BẢN ĐỐI CHỨNG (GSM8K)                      ")
    print("=" * 120)
    print(f"{'Chỉ số đánh giá':<28} | {'KB1: Baseline Gốc':<18} | {'KB2: Base + RPDI':<18} | {'KB3: RLVR (GRPO)':<18} | {'KB4: RLVR + RPDI':<18}")
    print("-" * 120)
    print(
        f"{'Độ chính xác (Accuracy)':<28} | "
        f"{s1_metrics.get('accuracy_pct', 0.0):>15.2f}% | "
        f"{s2_metrics.get('accuracy_pct', 0.0):>15.2f}% | "
        f"{s3_metrics.get('accuracy_pct', 0.0):>15.2f}% | "
        f"{s4_metrics.get('accuracy_pct', 0.0):>15.2f}%"
    )
    print(
        f"{'Thời gian TB mỗi câu (s)':<28} | "
        f"{s1_metrics.get('avg_latency_s', 0.0):>16.2f}s | "
        f"{s2_metrics.get('avg_latency_s', 0.0):>16.2f}s | "
        f"{s3_metrics.get('avg_latency_s', 0.0):>16.2f}s | "
        f"{s4_metrics.get('avg_latency_s', 0.0):>16.2f}s"
    )
    print(
        f"{'Tổng token trung bình':<28} | "
        f"{s1_metrics.get('avg_total_tokens', 0.0):>16.1f}  | "
        f"{s2_metrics.get('avg_total_tokens', 0.0):>16.1f}  | "
        f"{s3_metrics.get('avg_total_tokens', 0.0):>16.1f}  | "
        f"{s4_metrics.get('avg_total_tokens', 0.0):>16.1f} "
    )
    print(
        f"{'Token suy nghĩ (<think>)':<28} | "
        f"{s1_metrics.get('avg_think_tokens', 0.0):>16.1f}  | "
        f"{s2_metrics.get('avg_think_tokens', 0.0):>16.1f}  | "
        f"{s3_metrics.get('avg_think_tokens', 0.0):>16.1f}  | "
        f"{s4_metrics.get('avg_think_tokens', 0.0):>16.1f} "
    )
    print(
        f"{'Token câu trả lời':<28} | "
        f"{s1_metrics.get('avg_answer_tokens', 0.0):>16.1f}  | "
        f"{s2_metrics.get('avg_answer_tokens', 0.0):>16.1f}  | "
        f"{s3_metrics.get('avg_answer_tokens', 0.0):>16.1f}  | "
        f"{s4_metrics.get('avg_answer_tokens', 0.0):>16.1f} "
    )
    print(
        f"{'Tỷ lệ phanh Early-Exit':<28} | "
        f"{'-':>17} | "
        f"{s2_metrics.get('early_exit_rate_pct', 0.0):>15.2f}% | "
        f"{'-':>17} | "
        f"{s4_metrics.get('early_exit_rate_pct', 0.0):>15.2f}%"
    )
    print("=" * 120)


def run_4_scenarios():
    args = parse_args()
    set_seed(args.seed)

    print(f"[Dữ liệu] Đang tải tập dữ liệu kiểm thử GSM8K...")
    dataset = load_gsm8k_test(filepath=args.data_file, limit=args.limit)
    total_samples = len(dataset)
    print(f"[Dữ liệu] Đã tải {total_samples} mẫu.")

    base_model, rlvr_model, tokenizer = load_models(args.model_id, args.adapter_dir)

    s1_results, s2_results, s3_results, s4_results = [], [], [], []
    sample_records = []

    print("\n🚀 [Thực nghiệm] Bắt đầu đánh giá đồng thời 4 kịch bản:")
    start_time = time.time()

    for idx, sample in enumerate(dataset):
        s_id = sample.get("id", idx)
        question = sample["question"]
        gt = sample["ground_truth"]

        print(f"\n👉 [Câu {idx + 1}/{total_samples}] (ID: {s_id}): \"{question[:60]}...\" (GT={gt})")

        # 1. Kịch bản 1: Baseline Gốc
        r1 = generate_standard(base_model, tokenizer, question, max_new_tokens=args.max_new_tokens, seed=args.seed)
        r1["is_correct"] = is_answer_correct(r1["predicted_answer"], gt)
        s1_results.append(r1)
        print(f"   [KB1 - Baseline Gốc ]: Dự đoán='{r1['predicted_answer']}' [{'ĐÚNG' if r1['is_correct'] else 'SAI'}] | {r1['total_tokens']} tok | {r1['latency']:.2f}s")

        # 2. Kịch bản 2: Baseline + RPDI-EE
        r2 = generate_rpdi(base_model, tokenizer, question, max_new_tokens=args.max_new_tokens, W=args.W, lambda_th=args.lambda_th, min_steps=args.min_steps, seed=args.seed)
        r2["is_correct"] = is_answer_correct(r2["predicted_answer"], gt)
        s2_results.append(r2)
        print(f"   [KB2 - Base + RPDI  ]: Dự đoán='{r2['predicted_answer']}' [{'ĐÚNG' if r2['is_correct'] else 'SAI'}] | {r2['total_tokens']} tok (bước {r2['exit_step']}) | {r2['latency']:.2f}s")

        # 3. Kịch bản 3: RLVR (GRPO)
        r3 = generate_standard(rlvr_model, tokenizer, question, max_new_tokens=args.max_new_tokens, seed=args.seed)
        r3["is_correct"] = is_answer_correct(r3["predicted_answer"], gt)
        s3_results.append(r3)
        print(f"   [KB3 - RLVR (GRPO)  ]: Dự đoán='{r3['predicted_answer']}' [{'ĐÚNG' if r3['is_correct'] else 'SAI'}] | {r3['total_tokens']} tok | {r3['latency']:.2f}s")

        # 4. Kịch bản 4: RLVR + RPDI-EE
        r4 = generate_rpdi(rlvr_model, tokenizer, question, max_new_tokens=args.max_new_tokens, W=args.W, lambda_th=args.lambda_th, min_steps=args.min_steps, seed=args.seed)
        r4["is_correct"] = is_answer_correct(r4["predicted_answer"], gt)
        s4_results.append(r4)
        print(f"   [KB4 - RLVR + RPDI  ]: Dự đoán='{r4['predicted_answer']}' [{'ĐÚNG' if r4['is_correct'] else 'SAI'}] | {r4['total_tokens']} tok (bước {r4['exit_step']}) | {r4['latency']:.2f}s")

        sample_records.append({
            "sample_id": s_id,
            "question": question,
            "ground_truth": gt,
            "scenario_1_baseline": r1,
            "scenario_2_base_rpdi": r2,
            "scenario_3_rlvr": r3,
            "scenario_4_rlvr_rpdi": r4,
        })

    total_time = time.time() - start_time

    # Tính toán số liệu tổng hợp
    s1_metrics = compute_aggregate_metrics(s1_results)
    s2_metrics = compute_aggregate_metrics(s2_results)
    s3_metrics = compute_aggregate_metrics(s3_results)
    s4_metrics = compute_aggregate_metrics(s4_results)

    # In bảng Markdown
    print_4_scenarios_table(s1_metrics, s2_metrics, s3_metrics, s4_metrics)
    print(f"\nTổng thời gian hoàn tất: {total_time:.2f} giây.")

    # Lưu kết quả JSON
    output_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    full_summary = {
        "metadata": {
            "model_id": args.model_id,
            "adapter_dir": args.adapter_dir,
            "num_samples": total_samples,
            "total_time_s": round(total_time, 2),
            "W": args.W,
            "lambda_th": args.lambda_th,
            "min_steps": args.min_steps,
        },
        "aggregate_metrics": {
            "scenario_1_baseline": s1_metrics,
            "scenario_2_base_rpdi": s2_metrics,
            "scenario_3_rlvr": s3_metrics,
            "scenario_4_rlvr_rpdi": s4_metrics,
        },
        "sample_records": sample_records,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)

    print(f"[Kết quả] Đã lưu kết quả 4 kịch bản vào: {args.output_file}")


if __name__ == "__main__":
    run_4_scenarios()
