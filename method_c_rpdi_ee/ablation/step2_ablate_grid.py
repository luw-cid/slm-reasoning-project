"""Bước 2: Quét lưới Ablation Study (W x lambda) kết hợp Offline Simulation + Sinh tiếp Answer trên GPU.
Mô phỏng offline cực nhanh trên CPU điểm dừng sớm của tất cả 16 cấu hình (W x lambda) và các baseline cắt cứng.
Chỉ nạp GPU để sinh phần câu trả lời (Answer Phase) cho các điểm dừng phân biệt (Unique Exit Cache),
tiết kiệm 85-90% thời gian tính toán!
Kết quả được lưu vào results/ablation_grid_results.json.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

# Đảm bảo in ký tự tiếng Việt an toàn trên console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.metrics import (
    extract_answer_from_response,
    is_answer_correct,
)
from common.model_loader import DEFAULT_MODEL_ID, load_model_and_tokenizer


def simulate_rpdi_exit(
    entropies: List[float],
    generated_token_ids: List[int],
    boundary_ids: Set[int],
    natural_think_end_idx: Optional[int],
    W: int,
    lambda_th: float,
    min_steps: Optional[int] = None,
) -> Optional[int]:
    """Mô phỏng thuật toán RPDI-EE offline trên CPU dựa vào mảng entropy và chuỗi token.

    Trả về:
        Số lượng token suy nghĩ đã sinh trước khi phanh (think_tokens), hoặc None nếu không phanh.
    """
    effective_min_steps = min_steps if min_steps is not None else W
    max_step = natural_think_end_idx if natural_think_end_idx is not None else len(entropies)

    S_global = 0.0
    S_local = 0.0

    for i in range(1, max_step + 1):
        H_ti = entropies[i - 1]
        S_global += H_ti
        S_local += H_ti
        if i > W:
            S_local -= entropies[i - W - 1]

        # Kiểm tra token biên được sinh ở bước trước (i > 1)
        if i > 1:
            prev_tok = generated_token_ids[i - 2]
            if prev_tok in boundary_ids:
                if i >= effective_min_steps and i >= W:
                    gtf = S_global / i
                    ltf = S_local / W
                    rpdi = (ltf / gtf) if gtf > 0 else 0.0

                    if rpdi > lambda_th:
                        # Kích hoạt dừng sớm tại bước i:
                        # Số token suy nghĩ đã sinh trước khi chèn </think> là i - 1
                        return i - 1

    return None


def simulate_hard_truncation(
    natural_think_end_idx: Optional[int],
    total_generated: int,
    N: int,
) -> Optional[int]:
    """Mô phỏng baseline cắt cứng (Fixed Hard Truncation) tại N token suy nghĩ."""
    current_think_len = natural_think_end_idx if natural_think_end_idx is not None else total_generated
    if current_think_len > N:
        return N
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bước 2: Quét lưới Ablation Study (W x lambda) với Offline Simulation + Answer Generation."
    )
    parser.add_argument(
        "--traces_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "baseline_traces.json"),
        help="Đường dẫn file vết suy luận từ Bước 1 (mặc định: results/baseline_traces.json)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "ablation_grid_results.json"),
        help="Đường dẫn lưu kết quả quét lưới (mặc định: results/ablation_grid_results.json)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"ID mô hình Hugging Face (mặc định: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="Kiểu dữ liệu trọng số (auto, bfloat16, float16)",
    )
    parser.add_argument(
        "--W_list",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Danh sách giá trị kích thước cửa sổ W cần ablate (mặc định: 16 32 64 128)",
    )
    parser.add_argument(
        "--lambda_list",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 2.5, 3.0],
        help="Danh sách giá trị ngưỡng lambda cần ablate (mặc định: 1.5 2.0 2.5 3.0)",
    )
    parser.add_argument(
        "--hard_trunc_list",
        type=int,
        nargs="+",
        default=[32, 64, 96, 128, 256],
        help="Danh sách các mốc cắt cứng token suy nghĩ để làm baseline đối chứng (mặc định: 32 64 96 128 256)",
    )
    parser.add_argument(
        "--min_steps_mode",
        type=str,
        choices=["w", "none", "fixed64"],
        default="w",
        help="Quy ước min_steps: 'w' (dùng min_steps=W theo bài báo), 'fixed64' (ép min_steps=64 như cũ).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 85)
    print("🚀 BƯỚC 2: QUÉT LƯỚI ABLATION STUDY (OFFLINE SIMULATION + UNIQUE ANSWER RESUMPTION)")
    print("=" * 85)

    if not os.path.exists(args.traces_file):
        raise FileNotFoundError(
            f"Không tìm thấy file vết suy luận tại '{args.traces_file}'. "
            f"Vui lòng chạy 'step1_trace_baseline.py' trước!"
        )

    with open(args.traces_file, "r", encoding="utf-8") as f:
        traces_data = json.load(f)

    traces_list = traces_data["traces"]
    boundary_token_ids = set(traces_data["metadata"]["boundary_token_ids"])
    max_new_tokens = traces_data["metadata"]["max_new_tokens"]
    num_samples = len(traces_list)

    # Kiểm tra tính toàn vẹn của dữ liệu vết (Sanity check)
    for sample in traces_list:
        ent_len = len(sample.get("entropies", []))
        tok_len = len(sample.get("generated_token_ids", []))
        assert ent_len == tok_len, (
            f"Lỗi toàn vẹn dữ liệu ở mẫu ID {sample.get('sample_id')}: "
            f"len(entropies)={ent_len} != len(generated_token_ids)={tok_len}! "
            f"Vui lòng chạy lại 'step1_trace_baseline.py' để ghi lại vết hợp lệ."
        )

    print(f"[*] Đã tải {num_samples} vết suy luận từ '{args.traces_file}'.")
    print(f"[*] Cấu hình lưới tham số:")
    print(f"    - W ∈ {args.W_list}")
    print(f"    - λ ∈ {args.lambda_list}")
    print(f"    - Baselines Cắt cứng N ∈ {args.hard_trunc_list}")
    print(f"    - Chế độ min_steps: '{args.min_steps_mode}' (theo bài báo)")

    # ---------------------------------------------------------
    # PHA 1: MÔ PHỎNG OFFLINE TRÊN CPU (VÀI GIÂY)
    # ---------------------------------------------------------
    start_sim_time = time.time()
    print("\n[*] PHA 1: Đang mô phỏng dừng sớm offline trên CPU cho toàn bộ cấu hình...")

    # Cấu trúc lưu: exit_plan[config_name][sample_idx] = think_tokens (hoặc None nếu không dừng)
    exit_plans: Dict[str, List[Optional[int]]] = {}

    # 1. Các cấu hình RPDI (W x lambda)
    for W in args.W_list:
        for lam in args.lambda_list:
            cfg_name = f"RPDI(W={W},λ={lam})"
            cfg_plan: List[Optional[int]] = []
            for sample in traces_list:
                min_steps_val = W if args.min_steps_mode == "w" else (64 if args.min_steps_mode == "fixed64" else None)
                exit_think = simulate_rpdi_exit(
                    entropies=sample["entropies"],
                    generated_token_ids=sample["generated_token_ids"],
                    boundary_ids=boundary_token_ids,
                    natural_think_end_idx=sample["natural_think_end_idx"],
                    W=W,
                    lambda_th=lam,
                    min_steps=min_steps_val,
                )
                cfg_plan.append(exit_think)
            exit_plans[cfg_name] = cfg_plan

    # 2. Các baseline cắt cứng (Hard Truncation)
    for N in args.hard_trunc_list:
        cfg_name = f"HardTrunc(N={N})"
        cfg_plan: List[Optional[int]] = []
        for sample in traces_list:
            exit_think = simulate_hard_truncation(
                natural_think_end_idx=sample["natural_think_end_idx"],
                total_generated=len(sample["generated_token_ids"]),
                N=N,
            )
            cfg_plan.append(exit_think)
        exit_plans[cfg_name] = cfg_plan

    sim_duration = time.time() - start_sim_time
    print(f"[+] Mô phỏng hoàn tất trong {sim_duration:.3f}s!")

    # ---------------------------------------------------------
    # PHA 2: TẬP HỢP CÁC ĐIỂM DỪNG PHÂN BIỆT (UNIQUE EXIT CACHE)
    # ---------------------------------------------------------
    # Tập hợp các cặp cần sinh tiếp: (sample_idx, think_tokens)
    needed_continuations: Set[Tuple[int, int]] = set()
    for cfg_name, plan in exit_plans.items():
        for sample_idx, think_tokens in enumerate(plan):
            if think_tokens is not None:
                needed_continuations.add((sample_idx, think_tokens))

    total_possible_runs = len(exit_plans) * num_samples
    print(f"\n[*] PHA 2: Thống kê tối ưu hóa tính toán:")
    print(f"    - Tổng số lần chạy nếu sinh thông thường: {total_possible_runs}")
    print(f"    - Số lượt Answer Phase thực sự cần sinh trên GPU: {len(needed_continuations)}")
    savings_pct = (1.0 - (len(needed_continuations) / max(1, total_possible_runs))) * 100.0
    print(f"    - Tỷ lệ tiết kiệm tài nguyên GPU: {savings_pct:.1f}%")

    # ---------------------------------------------------------
    # PHA 3: NẠP MODEL VÀ SINH TIẾP ANSWER PHASE TRÊN GPU
    # ---------------------------------------------------------
    continuation_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    if needed_continuations:
        print(f"\n[*] PHA 3: Đang nạp mô hình vào GPU để sinh các phần Answer Phase còn thiếu...")
        model, tokenizer = load_model_and_tokenizer(
            model_id=args.model_id,
            torch_dtype=args.torch_dtype,
        )
        close_tokens: List[int] = tokenizer.encode("</think>\n", add_special_tokens=False)
        close_len = len(close_tokens)

        sorted_tasks = sorted(list(needed_continuations), key=lambda x: (x[0], x[1]))
        start_gpu_time = time.time()

        for task_idx, (sample_idx, think_tokens) in enumerate(sorted_tasks):
            sample = traces_list[sample_idx]
            prompt_token_ids = sample["prompt_token_ids"]
            generated_token_ids = sample["generated_token_ids"]
            ground_truth = sample["ground_truth"]

            # Ghép chuỗi prefix: Prompt + Think tokens đến điểm dừng + '</think>\n'
            prefix_tokens = prompt_token_ids + generated_token_ids[:think_tokens] + close_tokens
            tokens_used = think_tokens + close_len
            remaining_tokens = max(0, max_new_tokens - tokens_used)

            if remaining_tokens <= 0:
                answer_text = ""
                pred_answer = ""
                is_correct = False
                ans_token_count = 0
            else:
                input_tensor = torch.tensor([prefix_tokens], dtype=torch.long, device=model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_tensor,
                        max_new_tokens=remaining_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        do_sample=False,
                    )
                full_ids = outputs[0].tolist()
                answer_token_ids = full_ids[len(prefix_tokens):]
                ans_token_count = len(answer_token_ids)
                full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
                pred_answer = extract_answer_from_response(full_text)
                is_correct = is_answer_correct(pred_answer, ground_truth)

            total_tokens = tokens_used + ans_token_count
            continuation_cache[(sample_idx, think_tokens)] = {
                "predicted_answer": pred_answer,
                "is_correct": is_correct,
                "total_tokens": total_tokens,
                "think_tokens": think_tokens,
                "answer_tokens": ans_token_count,
                "exit_step": tokens_used,
            }

            if (task_idx + 1) % 10 == 0 or (task_idx + 1) == len(sorted_tasks):
                elapsed = time.time() - start_gpu_time
                print(f"  -> Đã sinh [{task_idx + 1:3d}/{len(sorted_tasks)}] Answer continuations ({elapsed:.1f}s)")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PHA 4: TỔNG HỢP VÀ ĐÁNH GIÁ TỪNG CẤU HÌNH
    # ---------------------------------------------------------
    print("\n[*] PHA 4: Tổng hợp chỉ số hiệu năng (Accuracy vs Tokens) cho từng cấu hình...")

    aggregated_configs: Dict[str, Dict[str, Any]] = {}

    # 1. Điểm tham chiếu Standard Baseline
    std_correct = sum(1 for s in traces_list if s["standard_metrics"]["is_correct"])
    std_total_tokens = sum(s["standard_metrics"]["total_tokens"] for s in traces_list)
    std_think_tokens = sum(s["standard_metrics"]["think_tokens"] for s in traces_list)
    std_answer_tokens = sum(s["standard_metrics"]["answer_tokens"] for s in traces_list)

    aggregated_configs["Standard (No Exit)"] = {
        "type": "standard_baseline",
        "W": None,
        "lambda_th": None,
        "N": None,
        "correct": std_correct,
        "total_samples": num_samples,
        "accuracy_pct": round((std_correct / num_samples) * 100.0, 2),
        "avg_total_tokens": round(std_total_tokens / num_samples, 2),
        "avg_think_tokens": round(std_think_tokens / num_samples, 2),
        "avg_answer_tokens": round(std_answer_tokens / num_samples, 2),
        "early_exit_count": 0,
        "early_exit_rate_pct": 0.0,
        "avg_exit_step": 0.0,
    }

    # 2. Các cấu hình RPDI và Cắt cứng
    for cfg_name, plan in exit_plans.items():
        correct_cnt = 0
        tot_tok_sum = 0
        thk_tok_sum = 0
        ans_tok_sum = 0
        exit_cnt = 0
        exit_step_sum = 0

        for sample_idx, think_tokens in enumerate(plan):
            sample = traces_list[sample_idx]
            if think_tokens is None:
                # Không phanh: Dùng lại nguyên vẹn kết quả Standard Baseline
                metrics = sample["standard_metrics"]
                is_corr = metrics["is_correct"]
                tot_tok = metrics["total_tokens"]
                thk_tok = metrics["think_tokens"]
                ans_tok = metrics["answer_tokens"]
            else:
                # Có phanh: Lấy kết quả từ cache sinh tiếp
                cached = continuation_cache[(sample_idx, think_tokens)]
                is_corr = cached["is_correct"]
                tot_tok = cached["total_tokens"]
                thk_tok = cached["think_tokens"]
                ans_tok = cached["answer_tokens"]
                exit_cnt += 1
                exit_step_sum += cached["exit_step"]

            if is_corr:
                correct_cnt += 1
            tot_tok_sum += tot_tok
            thk_tok_sum += thk_tok
            ans_tok_sum += ans_tok

        # Phân loại cấu hình
        if cfg_name.startswith("RPDI"):
            cfg_type = "rpdi"
            # Trích xuất W và lambda từ tên cấu hình
            # RPDI(W=16,λ=1.5)
            w_val = int(cfg_name.split("W=")[1].split(",")[0])
            lam_val = float(cfg_name.split("λ=")[1].split(")")[0])
            n_val = None
        else:
            cfg_type = "hard_truncation"
            w_val = None
            lam_val = None
            n_val = int(cfg_name.split("N=")[1].split(")")[0])

        aggregated_configs[cfg_name] = {
            "type": cfg_type,
            "W": w_val,
            "lambda_th": lam_val,
            "N": n_val,
            "correct": correct_cnt,
            "total_samples": num_samples,
            "accuracy_pct": round((correct_cnt / num_samples) * 100.0, 2),
            "avg_total_tokens": round(tot_tok_sum / num_samples, 2),
            "avg_think_tokens": round(thk_tok_sum / num_samples, 2),
            "avg_answer_tokens": round(ans_tok_sum / num_samples, 2),
            "early_exit_count": exit_cnt,
            "early_exit_rate_pct": round((exit_cnt / num_samples) * 100.0, 2),
            "avg_exit_step": round(exit_step_sum / max(1, exit_cnt), 2),
        }

    # Bảng kết quả in ra màn hình
    print("\n" + "=" * 95)
    print(f"{'Cấu hình':<25} | {'Acc (%)':<8} | {'Avg Total':<10} | {'Avg Think':<10} | {'Exit Rate (%)':<14} | {'Avg Exit':<8}")
    print("-" * 95)
    for cfg_name, metrics in aggregated_configs.items():
        print(
            f"{cfg_name:<25} | {metrics['accuracy_pct']:>6.2f}%  | {metrics['avg_total_tokens']:>8.1f}   | "
            f"{metrics['avg_think_tokens']:>8.1f}   | {metrics['early_exit_rate_pct']:>11.1f}%   | {metrics['avg_exit_step']:>7.1f}"
        )
    print("=" * 95)

    # Đóng gói và lưu file
    final_output = {
        "metadata": {
            "model_id": args.model_id,
            "num_samples": num_samples,
            "W_list": args.W_list,
            "lambda_list": args.lambda_list,
            "hard_trunc_list": args.hard_trunc_list,
            "min_steps_mode": args.min_steps_mode,
        },
        "configurations": aggregated_configs,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Đã lưu toàn bộ dữ liệu quét lưới thành công vào: {args.output_file}")


if __name__ == "__main__":
    main()
