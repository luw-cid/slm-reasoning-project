"""Script thực nghiệm đánh giá và so sánh toàn diện Standard Baseline vs RPDI-EE.
Chạy trên tập 100 câu GSM8K với chế độ giải mã Greedy (mặc định) hoặc Sampling,
đo lường đồng thời 3 trục: Độ chính xác (Accuracy), Số lượng Token sinh ra và Thời gian thực thi.
"""

import argparse
import io
import json
import os
import sys
import time
from typing import Any, Dict, List

# Đảm bảo in ký tự tiếng Việt an toàn trên console Windows và Linux
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc của project vào sys.path để import common và method_c_rpdi_ee thuận tiện
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.eval_dataset import load_gsm8k_test
from common.metrics import (
    compute_aggregate_metrics,
    is_answer_correct,
)
from common.model_loader import DEFAULT_MODEL_ID
from common.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Xử lý các tham số dòng lệnh phục vụ thực nghiệm."""
    parser = argparse.ArgumentParser(
        description="Đánh giá hiệu năng Standard vs RPDI-EE trên tập kiểm thử GSM8K."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"ID hoặc đường dẫn mô hình Hugging Face (mặc định: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số câu hỏi chạy thử nghiệm (ví dụ: 5 để test nhanh, để trống để chạy full 100 câu).",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Đường dẫn tùy chỉnh đến file JSON chứa tập dữ liệu kiểm thử (mặc định dùng common/data/gsm8k_test_100.json).",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=8,
        help="Kích thước cửa sổ trượt W tính entropy cục bộ (mặc định: 8; tham số bài báo arXiv:2603.14251 là 512).",
    )
    parser.add_argument(
        "--lambda_th",
        type=float,
        default=1.2,
        help="Ngưỡng kích hoạt dừng sớm lambda (mặc định: 1.2; tham số bài báo arXiv:2603.14251 là 2.0).",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=64,
        help="Số token suy nghĩ tối thiểu trước khi cho phép kích hoạt phanh dừng sớm (mặc định: 64).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Số lượng token sinh mới tối đa cho mỗi câu hỏi (mặc định: 512).",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Bật chế độ Sampling ngẫu nhiên (mặc định: False - sử dụng Greedy Decoding để đảm bảo tính tất định).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Nhiệt độ sampling khi bật --do_sample (mặc định: 0.6).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p khi bật --do_sample (mặc định: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên cho thực nghiệm (mặc định: 42).",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Thiết bị nạp mô hình (mặc định: 'auto').",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="Độ chính xác trọng số mô hình: 'auto', 'bfloat16', 'float16', 'float32' (mặc định: 'auto').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "results", "method_c_results.json"
        ),
        help="Đường dẫn file JSON để ghi kết quả thực nghiệm chi tiết.",
    )
    return parser.parse_args()


def print_comparison_table(
    std_metrics: Dict[str, Any],
    rpdi_metrics: Dict[str, Any],
    avg_exit_step: float,
    decoding_mode: str,
    W: int,
    lambda_th: float,
) -> None:
    """In bảng so sánh tổng kết hiệu năng theo định dạng trực quan."""
    diff_acc = rpdi_metrics["accuracy_pct"] - std_metrics["accuracy_pct"]
    diff_latency = (
        (std_metrics["avg_latency_s"] - rpdi_metrics["avg_latency_s"])
        / std_metrics["avg_latency_s"]
        * 100.0
        if std_metrics["avg_latency_s"] > 0
        else 0.0
    )
    diff_total_tok = (
        (std_metrics["avg_total_tokens"] - rpdi_metrics["avg_total_tokens"])
        / std_metrics["avg_total_tokens"]
        * 100.0
        if std_metrics["avg_total_tokens"] > 0
        else 0.0
    )
    diff_think_tok = (
        (std_metrics["avg_think_tokens"] - rpdi_metrics["avg_think_tokens"])
        / std_metrics["avg_think_tokens"]
        * 100.0
        if std_metrics["avg_think_tokens"] > 0
        else 0.0
    )
    diff_ans_tok = (
        (rpdi_metrics["avg_answer_tokens"] - std_metrics["avg_answer_tokens"])
        / std_metrics["avg_answer_tokens"]
        * 100.0
        if std_metrics["avg_answer_tokens"] > 0
        else 0.0
    )
    diff_tps = (
        (rpdi_metrics["tokens_per_second"] - std_metrics["tokens_per_second"])
        / std_metrics["tokens_per_second"]
        * 100.0
        if std_metrics["tokens_per_second"] > 0
        else 0.0
    )

    width = 112
    print("\n" + "=" * width)
    print(
        f"{'BẢNG SO SÁNH HIỆU NĂNG THỰC NGHIỆM: STANDARD BASELINE vs RPDI-EE':^{width}}"
    )
    print(
        f"{f'Cấu hình giải mã: {decoding_mode} | Tham số phanh: W={W}, lambda={lambda_th}':^{width}}"
    )
    print("=" * width)
    print(
        f"{'Chỉ số đánh giá':<30} | {'Standard Baseline':^24} | {'RPDI-EE (Can thiệp)':^24} | {'Chênh lệch (Δ / %)':^24}"
    )
    print("-" * width)

    # 1. Accuracy
    acc_std_str = f"{std_metrics['accuracy_pct']:.2f}% ({std_metrics['correct']}/{std_metrics['num_samples']})"
    acc_rpdi_str = f"{rpdi_metrics['accuracy_pct']:.2f}% ({rpdi_metrics['correct']}/{rpdi_metrics['num_samples']})"
    acc_diff_str = f"{diff_acc:+.2f}%"
    print(f"{'Độ chính xác (Accuracy)':<30} | {acc_std_str:^24} | {acc_rpdi_str:^24} | {acc_diff_str:^24}")

    # 2. Latency
    lat_std_str = f"{std_metrics['avg_latency_s']:.2f}s"
    lat_rpdi_str = f"{rpdi_metrics['avg_latency_s']:.2f}s"
    lat_diff_str = f"{diff_latency:+.2f}% {'(Nhanh hơn)' if diff_latency > 0 else '(Chậm hơn)'}"
    print(f"{'Thời gian TB mỗi câu':<30} | {lat_std_str:^24} | {lat_rpdi_str:^24} | {lat_diff_str:^24}")

    # 3. Total Tokens
    tok_std_str = f"{std_metrics['avg_total_tokens']:.1f} tokens"
    tok_rpdi_str = f"{rpdi_metrics['avg_total_tokens']:.1f} tokens"
    tok_diff_str = f"{-diff_total_tok:+.2f}% {'(Tiết kiệm)' if diff_total_tok > 0 else '(Tăng)'}"
    print(f"{'Tổng số token trung bình':<30} | {tok_std_str:^24} | {tok_rpdi_str:^24} | {tok_diff_str:^24}")

    # 4. Think Tokens
    think_std_str = f"{std_metrics['avg_think_tokens']:.1f} tokens"
    think_rpdi_str = f"{rpdi_metrics['avg_think_tokens']:.1f} tokens"
    think_diff_str = f"{-diff_think_tok:+.2f}% {'(Cắt ngắn)' if diff_think_tok > 0 else '(Dài hơn)'}"
    print(f"{'Số token suy nghĩ (<think>)':<30} | {think_std_str:^24} | {think_rpdi_str:^24} | {think_diff_str:^24}")

    # 5. Answer Tokens
    ans_std_str = f"{std_metrics['avg_answer_tokens']:.1f} tokens"
    ans_rpdi_str = f"{rpdi_metrics['avg_answer_tokens']:.1f} tokens"
    ans_diff_str = f"{diff_ans_tok:+.2f}%"
    print(f"{'Số token câu trả lời':<30} | {ans_std_str:^24} | {ans_rpdi_str:^24} | {ans_diff_str:^24}")

    # 6. Throughput TPS
    tps_std_str = f"{std_metrics['tokens_per_second']:.2f} tok/s"
    tps_rpdi_str = f"{rpdi_metrics['tokens_per_second']:.2f} tok/s"
    tps_diff_str = f"{diff_tps:+.2f}%"
    print(f"{'Tốc độ sinh (TPS)':<30} | {tps_std_str:^24} | {tps_rpdi_str:^24} | {tps_diff_str:^24}")

    # 7. Early Exit Statistics
    ee_rate_str = f"{rpdi_metrics['early_exit_rate_pct']:.2f}% ({rpdi_metrics['early_exit_count']}/{rpdi_metrics['num_samples']})"
    print(f"{'Tỷ lệ kích hoạt Early-Exit':<30} | {'-':^24} | {ee_rate_str:^24} | {'-':^24}")

    exit_step_str = f"{avg_exit_step:.1f} tokens" if avg_exit_step > 0 else "N/A"
    print(f"{'Bước dừng sớm trung bình':<30} | {'-':^24} | {exit_step_str:^24} | {'-':^24}")

    print("=" * width + "\n")


def main() -> None:
    args = parse_args()

    import torch
    from common.model_loader import load_model_and_tokenizer
    from method_c_rpdi_ee.runner import generate_rpdi, generate_standard

    # 1. Khởi tạo seed ngẫu nhiên
    set_seed(args.seed)

    # 2. Tải tập dữ liệu kiểm thử
    print(f"\n[Dữ liệu] Đang tải tập dữ liệu GSM8K kiểm thử...")
    dataset = load_gsm8k_test(filepath=args.data_file, limit=args.limit)
    num_samples = len(dataset)
    print(f"[Dữ liệu] Đã tải thành công {num_samples} mẫu kiểm thử.")

    # 3. Tải mô hình và Tokenizer
    print(f"\n[Mô hình] Đang tải mô hình '{args.model_id}'...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    decoding_mode = "Sampling (temp=0.6, top_p=0.95)" if args.do_sample else "Greedy Decoding (do_sample=False)"
    print(f"\n[Thực nghiệm] Bắt đầu chạy đánh giá đối chứng:")
    print(f" - Chiến lược giải mã: {decoding_mode}")
    print(f" - Cấu hình RPDI-EE: W={args.W}, lambda={args.lambda_th}, min_steps={args.min_steps or args.W}")
    print(f" - Giới hạn max_new_tokens: {args.max_new_tokens}")
    print(f" - Tổng số câu hỏi: {num_samples}")
    print("-" * 90)

    std_run_results: List[Dict[str, Any]] = []
    rpdi_run_results: List[Dict[str, Any]] = []
    sample_records: List[Dict[str, Any]] = []
    exit_steps_collected: List[int] = []

    start_eval_total_time = time.time()

    for idx, sample in enumerate(dataset):
        sample_id = sample.get("id", idx)
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        print(f"\n👉 [Câu {idx + 1}/{num_samples}] (ID: {sample_id})")
        print(f"   Đề bài: \"{question[:80]}...\"")
        print(f"   Đáp án chuẩn: {ground_truth}")

        # --- A. Chạy chế độ Standard Baseline ---
        std_out = generate_standard(
            model=model,
            tokenizer=tokenizer,
            question=question,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + idx if args.do_sample else None,
        )
        std_is_correct = is_answer_correct(std_out["predicted_answer"], ground_truth)
        std_record = {
            "is_correct": std_is_correct,
            "latency": std_out["latency"],
            "total_tokens": std_out["total_tokens"],
            "think_tokens": std_out["think_tokens"],
            "answer_tokens": std_out["answer_tokens"],
            "early_exited": False,
        }
        std_run_results.append(std_record)

        # Giải phóng bộ nhớ đệm CUDA giữa 2 lần chạy để đảm bảo tính công bằng
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- B. Chạy chế độ RPDI-EE (Có phanh dừng sớm) ---
        rpdi_out = generate_rpdi(
            model=model,
            tokenizer=tokenizer,
            question=question,
            max_new_tokens=args.max_new_tokens,
            W=args.W,
            lambda_th=args.lambda_th,
            min_steps=args.min_steps,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + idx if args.do_sample else None,
        )
        rpdi_is_correct = is_answer_correct(rpdi_out["predicted_answer"], ground_truth)
        rpdi_record = {
            "is_correct": rpdi_is_correct,
            "latency": rpdi_out["latency"],
            "total_tokens": rpdi_out["total_tokens"],
            "think_tokens": rpdi_out["think_tokens"],
            "answer_tokens": rpdi_out["answer_tokens"],
            "early_exited": rpdi_out["early_exited"],
        }
        rpdi_run_results.append(rpdi_record)

        if rpdi_out["early_exited"] and rpdi_out["exit_step"] >= 0:
            exit_steps_collected.append(rpdi_out["exit_step"])

        # Hiển thị tóm tắt lượt chạy của câu hiện tại
        std_status = "ĐÚNG" if std_is_correct else "SAI"
        rpdi_status = "ĐÚNG" if rpdi_is_correct else "SAI"
        ee_note = f" (PHANH tại bước {rpdi_out['exit_step']})" if rpdi_out["early_exited"] else " (Không phanh)"

        print(
            f"   [Standard]: Dự đoán='{std_out['predicted_answer']}' [{std_status}] | "
            f"{std_out['total_tokens']} tokens ({std_out['think_tokens']} suy nghĩ) | {std_out['latency']:.2f}s"
        )
        print(
            f"   [RPDI-EE ]: Dự đoán='{rpdi_out['predicted_answer']}' [{rpdi_status}]{ee_note} | "
            f"{rpdi_out['total_tokens']} tokens ({rpdi_out['think_tokens']} suy nghĩ) | {rpdi_out['latency']:.2f}s"
        )

        sample_records.append(
            {
                "sample_id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "standard": {
                    "predicted_answer": std_out["predicted_answer"],
                    "is_correct": std_is_correct,
                    "latency": round(std_out["latency"], 3),
                    "total_tokens": std_out["total_tokens"],
                    "think_tokens": std_out["think_tokens"],
                    "answer_tokens": std_out["answer_tokens"],
                    "response_text": std_out["response_text"],
                },
                "rpdi_ee": {
                    "predicted_answer": rpdi_out["predicted_answer"],
                    "is_correct": rpdi_is_correct,
                    "latency": round(rpdi_out["latency"], 3),
                    "total_tokens": rpdi_out["total_tokens"],
                    "think_tokens": rpdi_out["think_tokens"],
                    "answer_tokens": rpdi_out["answer_tokens"],
                    "early_exited": rpdi_out["early_exited"],
                    "exit_step": rpdi_out["exit_step"],
                    "rpdi_at_exit": round(rpdi_out["rpdi_at_exit"], 3),
                    "response_text": rpdi_out["response_text"],
                },
            }
        )

    total_eval_elapsed = time.time() - start_eval_total_time

    # 4. Tính toán các chỉ số tổng hợp
    std_metrics = compute_aggregate_metrics(std_run_results)
    rpdi_metrics = compute_aggregate_metrics(rpdi_run_results)
    avg_exit_step = (
        sum(exit_steps_collected) / len(exit_steps_collected)
        if exit_steps_collected
        else -1.0
    )

    # 5. In bảng tổng kết hiệu năng
    print_comparison_table(
        std_metrics=std_metrics,
        rpdi_metrics=rpdi_metrics,
        avg_exit_step=avg_exit_step,
        decoding_mode=decoding_mode,
        W=args.W,
        lambda_th=args.lambda_th,
    )
    print(f"Tổng thời gian toàn bộ thực nghiệm: {total_eval_elapsed:.2f} giây.")

    # 6. Lưu kết quả ra file JSON
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)

    final_payload = {
        "metadata": {
            "model_id": args.model_id,
            "decoding_mode": decoding_mode,
            "do_sample": args.do_sample,
            "W": args.W,
            "lambda_th": args.lambda_th,
            "min_steps": args.min_steps or args.W,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "num_samples": num_samples,
            "total_eval_time_seconds": round(total_eval_elapsed, 2),
        },
        "aggregate_metrics": {
            "standard": std_metrics,
            "rpdi_ee": rpdi_metrics,
            "comparison": {
                "delta_accuracy_pct": round(
                    rpdi_metrics["accuracy_pct"] - std_metrics["accuracy_pct"], 2
                ),
                "speedup_latency_pct": round(
                    (
                        (std_metrics["avg_latency_s"] - rpdi_metrics["avg_latency_s"])
                        / std_metrics["avg_latency_s"]
                        * 100.0
                    )
                    if std_metrics["avg_latency_s"] > 0
                    else 0.0,
                    2,
                ),
                "token_reduction_pct": round(
                    (
                        (
                            std_metrics["avg_total_tokens"]
                            - rpdi_metrics["avg_total_tokens"]
                        )
                        / std_metrics["avg_total_tokens"]
                        * 100.0
                    )
                    if std_metrics["avg_total_tokens"] > 0
                    else 0.0,
                    2,
                ),
                "think_token_reduction_pct": round(
                    (
                        (
                            std_metrics["avg_think_tokens"]
                            - rpdi_metrics["avg_think_tokens"]
                        )
                        / std_metrics["avg_think_tokens"]
                        * 100.0
                    )
                    if std_metrics["avg_think_tokens"] > 0
                    else 0.0,
                    2,
                ),
                "average_exit_step": round(avg_exit_step, 1),
            },
        },
        "sample_records": sample_records,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[Kết quả] Đã lưu kết quả chi tiết thành công vào: {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main()
