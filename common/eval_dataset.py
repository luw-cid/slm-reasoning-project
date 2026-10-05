"""Module tải tập dữ liệu đánh giá cho bài toán GSM8K.
Tải 100 mẫu kiểm thử đã được lưu trữ ngoại tuyến (offline) sẵn trong thư mục common/data.
"""

import json
import os
from typing import Dict, List, Optional

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "gsm8k_test_100.json"
)


def load_gsm8k_test(
    filepath: Optional[str] = None, limit: Optional[int] = None
) -> List[Dict[str, any]]:
    """Tải tập dữ liệu kiểm thử GSM8K.

    Tham số:
        filepath: Đường dẫn tùy chỉnh tới file JSON. Mặc định là common/data/gsm8k_test_100.json.
        limit: Giới hạn số lượng mẫu tối đa cần trả về (tùy chọn).

    Trả về:
        Danh sách các dictionary chứa thông tin mẫu dữ liệu: [
            {
                "id": int,
                "question": str,
                "full_solution": str,
                "ground_truth": str  # đáp án số chuẩn
            }, ...
        ]
    """
    path = filepath or DEFAULT_DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy tập dữ liệu tại {path}. Vui lòng kiểm tra đường dẫn hoặc script tạo dữ liệu."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    return data


def get_sample_by_id(sample_id: int, filepath: Optional[str] = None) -> Optional[Dict[str, any]]:
    """Lấy một mẫu kiểm thử theo mã định danh (ID)."""
    dataset = load_gsm8k_test(filepath)
    for sample in dataset:
        if sample.get("id") == sample_id:
            return sample
    return None
