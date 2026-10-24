"""Script tải toàn bộ tập dữ liệu GSM8K (Grade School Math 8K) gốc.

Hỗ trợ tải đầy đủ:
- Tập Test (1,319 câu): lưu vào common/data/gsm8k_test_full.json
- Tập Train (7,473 câu): lưu vào common/data/gsm8k_train_full.json

Tự động trích xuất đáp án chuẩn số học (ground_truth sau '####') chuẩn hóa
theo đúng cấu trúc của dự án.
Hỗ trợ cả tải qua Hugging Face Datasets hoặc tải trực tiếp từ GitHub gốc của OpenAI.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from typing import Any, Dict, List

# Đảm bảo in tiếng Việt trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.metrics import normalize_numeric_str

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

OPENAI_RAW_URLS = {
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
}


def extract_ground_truth(solution: str) -> str:
    """Trích xuất và chuẩn hóa đáp án số từ lời giải GSM8K."""
    if "####" in solution:
        ans_part = solution.split("####")[-1].strip()
        norm = normalize_numeric_str(ans_part)
        if norm:
            return norm
    numbers = re.findall(r"-?\d+(?:[\.,]\d+)*", solution)
    if numbers:
        return normalize_numeric_str(numbers[-1])
    return ""


def download_via_hf(split: str) -> List[Dict[str, Any]]:
    """Tải qua thư viện datasets của Hugging Face."""
    from datasets import load_dataset

    print(f"[*] Đang tải GSM8K (split='{split}') từ Hugging Face Hub...")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for idx, row in enumerate(ds):
        q = row["question"].strip()
        ans = row["answer"].strip()
        gt = extract_ground_truth(ans)
        items.append({
            "id": idx,
            "question": q,
            "full_solution": ans,
            "ground_truth": gt,
        })
    return items


def download_via_github(split: str) -> List[Dict[str, Any]]:
    """Tải trực tiếp từ repository gốc của OpenAI trên GitHub (không cần thư viện datasets)."""
    url = OPENAI_RAW_URLS[split]
    print(f"[*] Đang tải trực tiếp từ GitHub OpenAI: {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8")

    items = []
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        obj = json.loads(line)
        q = obj["question"].strip()
        ans = obj["answer"].strip()
        gt = extract_ground_truth(ans)
        items.append({
            "id": idx,
            "question": q,
            "full_solution": ans,
            "ground_truth": gt,
        })
    return items


def fetch_gsm8k_split(split: str) -> List[Dict[str, Any]]:
    """Thử tải bằng Hugging Face trước, nếu không được fallback tải trực tiếp từ GitHub."""
    try:
        return download_via_hf(split)
    except Exception as e_hf:
        print(f"[!] Không thể tải qua Hugging Face ({e_hf}). Đang chuyển sang tải trực tiếp từ GitHub...")
        try:
            return download_via_github(split)
        except Exception as e_gh:
            raise RuntimeError(f"Cả hai phương thức tải đều thất bại. HF: {e_hf} | GitHub: {e_gh}")


def main():
    parser = argparse.ArgumentParser(description="Tải toàn bộ tập dữ liệu GSM8K.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["test", "train", "all"],
        help="Chọn tập cần tải: 'test' (1,319 câu), 'train' (7,473 câu), hoặc 'all' (mặc định: 'all')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DATA_DIR,
        help=f"Thư mục lưu trữ file kết quả (mặc định: {DATA_DIR})",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    splits_to_download = ["test", "train"] if args.split == "all" else [args.split]

    for split in splits_to_download:
        print(f"\n==========================================")
        print(f"  Bắt đầu xử lý tập: {split.upper()}")
        print(f"==========================================")
        data = fetch_gsm8k_split(split)
        out_file = os.path.join(args.output_dir, f"gsm8k_{split}_full.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Đã lưu thành công {len(data)} câu hỏi vào: {out_file}")


if __name__ == "__main__":
    main()
