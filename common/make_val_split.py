"""Tạo tập validation cố định từ GSM8K train để chọn siêu tham số RPDI-EE (W, lambda)
mà không chạm vào tập test.

Tập này cũng được prepare_train_data.py (RLVR) loại trừ, để hai phương pháp không dùng chung câu.
Yêu cầu chạy common/download_gsm8k.py --split train trước.
"""

import argparse
import json
import os
import random
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo tập validation từ GSM8K train.")
    parser.add_argument("--train_file", type=str, default=os.path.join(DATA_DIR, "gsm8k_train_full.json"))
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_file", type=str, default=os.path.join(DATA_DIR, "gsm8k_val_200.json"))
    args = parser.parse_args()

    with open(args.train_file, "r", encoding="utf-8") as f:
        train = json.load(f)
    picked = random.Random(args.seed).sample(train, args.num_samples)
    val = [{**s, "id": i, "train_idx": s["id"]} for i, s in enumerate(picked)]

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    print(f"[OK] Đã lưu {len(val)} câu validation vào {args.output_file}")


if __name__ == "__main__":
    main()
