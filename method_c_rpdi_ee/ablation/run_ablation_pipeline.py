"""Script điều phối toàn bộ Pipeline 3 bước của Ablation Study:
1. Ghi vết Baseline (step1_trace_baseline.py)
2. Mô phỏng & Quét lưới (step2_ablate_grid.py)
3. Vẽ đồ thị Pareto Frontier (step3_plot_pareto.py)
"""

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy tự động toàn bộ Pipeline Ablation Study & Pareto Frontier (RPDI-EE)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số mẫu câu hỏi để chạy thử nghiệm nhanh (ví dụ: 10, 20 hoặc để trống cho full 100 câu).",
    )
    parser.add_argument(
        "--skip_step1",
        action="store_true",
        default=False,
        help="Bỏ qua Bước 1 nếu đã có file results/baseline_traces.json.",
    )
    parser.add_argument(
        "--skip_step2",
        action="store_true",
        default=False,
        help="Bỏ qua Bước 2 nếu đã có file results/ablation_grid_results.json.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="ID mô hình Hugging Face.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="Kiểu dữ liệu trọng số (auto, bfloat16, float16).",
    )
    return parser.parse_args()


def run_command(cmd_list: list):
    print(f"\n[RUN] {' '.join(cmd_list)}\n")
    res = subprocess.run(cmd_list, check=True)
    if res.returncode != 0:
        print(f"[!] Lỗi khi thực thi lệnh: {' '.join(cmd_list)}")
        sys.exit(res.returncode)


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    py_exec = sys.executable

    step1_script = os.path.join(base_dir, "step1_trace_baseline.py")
    step2_script = os.path.join(base_dir, "step2_ablate_grid.py")
    step3_script = os.path.join(base_dir, "step3_plot_pareto.py")

    # BƯỚC 1: Chạy Baseline ghi vết
    if not args.skip_step1:
        cmd1 = [py_exec, step1_script, "--model_id", args.model_id, "--torch_dtype", args.torch_dtype]
        if args.limit is not None:
            cmd1.extend(["--limit", str(args.limit)])
        run_command(cmd1)
    else:
        print("[*] Đã bỏ qua Bước 1 (--skip_step1).")

    # BƯỚC 2: Mô phỏng Offline & Sinh tiếp Answer
    if not args.skip_step2:
        cmd2 = [py_exec, step2_script, "--model_id", args.model_id, "--torch_dtype", args.torch_dtype]
        run_command(cmd2)
    else:
        print("[*] Đã bỏ qua Bước 2 (--skip_step2).")

    # BƯỚC 3: Vẽ biểu đồ Pareto & Xuất báo cáo
    cmd3 = [py_exec, step3_script]
    run_command(cmd3)

    print("\n" + "=" * 85)
    print("🎉 HOÀN TẤT TOÀN BỘ PIPELINE ABLATION STUDY THÀNH CÔNG!")
    print(" - Dữ liệu vết: results/baseline_traces.json")
    print(" - Dữ liệu quét lưới: results/ablation_grid_results.json")
    print(" - Biểu đồ Pareto: report/pareto_accuracy_vs_tokens.png")
    print(" - Báo cáo tổng hợp: report/ablation_summary.md")
    print("=" * 85)


if __name__ == "__main__":
    main()
