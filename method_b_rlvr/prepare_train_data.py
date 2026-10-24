"""Module chuẩn bị tập dữ liệu huấn luyện RLVR (GSM8K Train Split).
Đảm bảo nghiêm ngặt nguyên tắc Zero Data Leakage:
Đối chiếu và loại trừ tuyệt đối bất kỳ câu hỏi nào trùng lặp với 100 câu kiểm thử trong common/data/gsm8k_test_100.json.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Set

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
from common.metrics import normalize_numeric_str
from common.model_loader import format_reasoning_prompt
from common.seed import set_seed

DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "data", "gsm8k_train_split.json"
)
VAL_FILEPATH = os.path.join(
    os.path.dirname(__file__), "..", "common", "data", "gsm8k_val_200.json"
)


def _normalize_text_for_comparison(text: str) -> str:
    """Chuẩn hóa chuỗi văn bản phục vụ so sánh kiểm tra rò rỉ dữ liệu."""
    if not text:
        return ""
    # Chuyển về chữ thường, loại bỏ dấu câu và khoảng trắng thừa
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_test_leakage_filter(test_filepath: str = None) -> Set[str]:
    """Tạo tập hợp các câu hỏi test đã chuẩn hóa để phát hiện trùng lặp."""
    test_samples = load_gsm8k_test(filepath=test_filepath)
    filter_set = set()
    for sample in test_samples:
        q = sample.get("question", "")
        norm_q = _normalize_text_for_comparison(q)
        if norm_q:
            filter_set.add(norm_q)
    print(f"[DataLeakageCheck] Đã nạp {len(filter_set)} câu hỏi kiểm thử làm màng lọc đối chiếu.")
    return filter_set


def extract_gsm8k_ground_truth(solution: str) -> str:
    """Trích xuất đáp án số nguyên bản từ chuỗi lời giải GSM8K (sau '####')."""
    if "####" in solution:
        ans_part = solution.split("####")[-1].strip()
        norm = normalize_numeric_str(ans_part)
        if norm:
            return norm
    # Fallback trích xuất số cuối cùng
    numbers = re.findall(r"-?\d+(?:[\.,]\d+)*", solution)
    if numbers:
        return normalize_numeric_str(numbers[-1])
    return ""


def prepare_rlvr_dataset(
    num_samples: int = 400,
    output_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = 42,
    test_filepath: str = None,
) -> List[Dict[str, Any]]:
    """Tải tập GSM8K train, lọc rò rỉ dữ liệu, định dạng prompt và lưu trữ ngoại tuyến."""
    set_seed(seed)
    leakage_filter = load_test_leakage_filter(test_filepath)
    # Loại cả tập validation dùng để chọn siêu tham số RPDI-EE (common/make_val_split.py)
    if os.path.exists(VAL_FILEPATH):
        leakage_filter |= load_test_leakage_filter(VAL_FILEPATH)

    print("[DataPrep] Đang tải tập dữ liệu gốc GSM8K (split='train') từ Hugging Face datasets...")
    try:
        from datasets import load_dataset
        raw_dataset = load_dataset("gsm8k", "main", split="train")
    except Exception as e:
        print(f"[Lỗi] Không thể tải dataset qua Hugging Face: {e}")
        print("Vui lòng đảm bảo đã kết nối Internet hoặc đã cài đặt thư viện 'datasets'.")
        raise

    print(f"[DataPrep] Tổng số mẫu trong GSM8K Train gốc: {len(raw_dataset)}")

    processed_samples = []
    skipped_leakage_count = 0

    # Xáo trộn với seed cố định để đảm bảo tính tái lập (reproducibility)
    shuffled_dataset = raw_dataset.shuffle(seed=seed)

    for idx, item in enumerate(shuffled_dataset):
        if len(processed_samples) >= num_samples:
            break

        question = item["question"].strip()
        solution = item["answer"].strip()

        # Kiểm tra chống rò rỉ dữ liệu (Zero Data Leakage)
        norm_q = _normalize_text_for_comparison(question)
        if norm_q in leakage_filter:
            skipped_leakage_count += 1
            print(f"[Cảnh báo Leakage] Bỏ qua mẫu trùng lặp kiểm thử: '{question[:60]}...'")
            continue

        ground_truth = extract_gsm8k_ground_truth(solution)
        if not ground_truth:
            continue

        # Tạo prompt chuẩn DeepSeek-R1 CoT
        prompt = format_reasoning_prompt(question)

        processed_samples.append({
            "id": len(processed_samples),
            "original_train_idx": idx,
            "question": question,
            "prompt": prompt,
            "full_solution": solution,
            "ground_truth": ground_truth,
        })

    print(f"[DataPrep] Hoàn tất xử lý:")
    print(f"  - Số mẫu hợp lệ được chọn: {len(processed_samples)}")
    print(f"  - Số mẫu bị loại trừ do nguy cơ rò rỉ kiểm thử: {skipped_leakage_count}")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)

    print(f"[DataPrep] Đã lưu file dữ liệu huấn luyện thành công tại: {output_path}")
    return processed_samples


def main():
    parser = argparse.ArgumentParser(
        description="Trích xuất tập huấn luyện GSM8K phục vụ phương pháp RLVR (GRPO) - Zero Data Leakage."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=400,
        help="Số lượng mẫu huấn luyện cần trích xuất (mặc định: 400 câu; khuyến nghị 300-500).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Đường dẫn lưu file JSON kết quả (mặc định: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed xáo trộn ngẫu nhiên (mặc định: 42).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Đường dẫn tùy chỉnh đến file test 100 câu để lọc leakage.",
    )

    args = parser.parse_args()
    prepare_rlvr_dataset(
        num_samples=args.num_samples,
        output_path=args.output_file,
        seed=args.seed,
        test_filepath=args.test_file,
    )


if __name__ == "__main__":
    main()
