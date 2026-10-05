"""Các chỉ số đánh giá và tiện ích trích xuất câu trả lời cho benchmark suy luận.
Trích xuất đáp số toán học (dạng \\boxed{}, mẫu biểu thức cụ thể, hoặc số cuối cùng) và tính toán các chỉ số độ chính xác / hiệu năng.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def normalize_numeric_str(val: str) -> str:
    """Chuẩn hóa chuỗi số phục vụ việc so khớp chính xác hoặc xấp xỉ.
    Loại bỏ dấu phẩy ngăn cách hàng nghìn, ký hiệu tiền tệ, khoảng trắng,
    ký tự escape LaTeX (\\, \\!, \\%, \\text{}, ...) và chuẩn hóa số nguyên / số thực.
    """
    if val is None:
        return ""
    val = str(val).strip()

    # Loại bỏ các lệnh text LaTeX thông dụng (ví dụ: \text{...}, \mathbf{...})
    val = re.sub(r"\\(?:text|mathbf|mathrm|mbox)\{([^}]*)\}", r"\1", val)

    # Loại bỏ ký hiệu tiền tệ, dấu phẩy, khoảng trắng, ký tự escape LaTeX (\, !, %, ~)
    val = re.sub(r"[\$,\s\\!%~#]", "", val)

    # Loại bỏ dấu ngoặc bao quanh và dấu chấm/hai chấm ở cuối
    val = re.sub(r"^[\(\{\[]*(.*?)[\)\}\]]*$", r"\1", val)
    val = val.rstrip(".:;")

    # Trích xuất số thực hoặc nguyên nếu chuỗi vẫn còn dính chữ cái thừa
    num_match = re.search(r"-?\d+(?:\.\d+)?", val)
    if num_match:
        extracted = num_match.group(0)
        try:
            f_val = float(extracted)
            if f_val.is_integer():
                return str(int(f_val))
            return str(f_val)
        except ValueError:
            return extracted

    return val


def _extract_boxed_content(text: str) -> List[str]:
    """Trích xuất nội dung bên trong các thẻ \\boxed{...}, hỗ trợ ngoặc lồng nhau."""
    results = []
    idx = 0
    while True:
        pos = text.find(r"\boxed{", idx)
        if pos == -1:
            break
        start = pos + len(r"\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1])
        idx = pos + len(r"\boxed{")
    return results


def extract_answer_from_response(text: str) -> Optional[str]:
    """Trích xuất đáp án số cuối cùng từ câu trả lời của mô hình ngôn ngữ lớn.

    Chiến lược ưu tiên:
    1. Tìm trong thẻ \\boxed{...} (hỗ trợ ngoặc nhọn lồng nhau như \\boxed{180 \\text{ minutes}})
    2. Tìm theo mẫu '#### <số>' (định dạng chuẩn của tập dữ liệu GSM8K)
    3. Tìm theo các cụm từ tường minh như 'the answer is <số>', 'total is <số>'
    4. Dự phòng: Trích xuất giá trị số xuất hiện cuối cùng trong phần trả lời
    """
    if not text:
        return None

    # Chỉ tìm kiếm sau thẻ </think> nếu có, tránh trích nhầm các con số trung gian trong quá trình nháp
    if "</think>" in text:
        content_to_search = text.split("</think>")[-1]
    else:
        content_to_search = text

    # 1. Thẻ LaTeX \boxed{...}
    boxed_candidates = _extract_boxed_content(content_to_search)
    if boxed_candidates:
        for cand in reversed(boxed_candidates):
            norm = normalize_numeric_str(cand)
            if norm:
                return norm

    # 2. Định dạng chuẩn GSM8K '#### <số>'
    hash_matches = re.findall(r"####\s*(-?\d+(?:[\.,]\d+)*)", content_to_search)
    if hash_matches:
        for match in reversed(hash_matches):
            norm = normalize_numeric_str(match)
            if norm:
                return norm

    # 3. Các mẫu câu kết luận đáp án tường minh (yêu cầu phải có chữ số)
    statement_patterns = [
        r"(?:the|final)?\s*answer\s*(?:is|equals|:|=)\s*[:\$]?\s*(-?\d+(?:[\.,]\d+)*)",
        r"(?:total|result)\s*(?:is|equals|:|=)\s*[:\$]?\s*(-?\d+(?:[\.,]\d+)*)",
        r"(?:therefore|thus|hence)\s*,?\s*(?:the\s+answer\s+is\s+|the\s+total\s+is\s+|it\s+is\s+)?[:\$]?\s*(-?\d+(?:[\.,]\d+)*)",
    ]
    for pattern in statement_patterns:
        matches = re.findall(pattern, content_to_search, re.IGNORECASE)
        if matches:
            for match in reversed(matches):
                norm = normalize_numeric_str(match)
                if norm:
                    return norm

    # 4. Dự phòng: Tìm tất cả các số (nguyên hoặc thập phân) và lấy số cuối cùng
    all_numbers = re.findall(r"-?\d+(?:\.\d+)?", content_to_search.replace(",", ""))
    if all_numbers:
        return normalize_numeric_str(all_numbers[-1])

    return None


def is_answer_correct(prediction: Optional[str], ground_truth: str, tolerance: float = 1e-4) -> bool:
    """Kiểm tra xem câu trả lời dự đoán của mô hình có khớp với đáp án chuẩn (ground truth) hay không."""
    if prediction is None:
        return False

    norm_pred = normalize_numeric_str(prediction)
    norm_gt = normalize_numeric_str(ground_truth)

    if norm_pred == norm_gt:
        return True

    # So sánh dạng số thực với sai số cho phép
    try:
        p_float = float(norm_pred)
        g_float = float(norm_gt)
        return abs(p_float - g_float) < tolerance
    except ValueError:
        return False


def split_think_and_answer(response_text: str) -> Tuple[str, str]:
    """Tách nội dung sinh của mô hình thành 2 phần: quá trình suy nghĩ (<think>...</think>) và câu trả lời cuối cùng."""
    think_text = ""
    answer_text = response_text

    if "<think>" in response_text:
        parts = response_text.split("<think>", 1)[1]
        if "</think>" in parts:
            think_text, answer_text = parts.split("</think>", 1)
        else:
            think_text = parts
            answer_text = ""
    elif "</think>" in response_text:
        think_text, answer_text = response_text.split("</think>", 1)

    return think_text.strip(), answer_text.strip()


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tính toán các chỉ số tổng hợp benchmark từ danh sách kết quả các lượt chạy.

    Mỗi phần tử kết quả cần có các trường:
    - is_correct: bool (đúng/sai)
    - latency: float (thời gian tính bằng giây)
    - total_tokens: int (tổng số token sinh)
    - think_tokens: int (số token trong phần suy nghĩ)
    - answer_tokens: int (số token trong phần trả lời)
    - early_exited: bool (có kích hoạt dừng sớm hay không - tùy chọn)
    """
    if not results:
        return {}

    n = len(results)
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    accuracy = (correct_count / n) * 100.0

    avg_latency = sum(r.get("latency", 0.0) for r in results) / n
    avg_total_tokens = sum(r.get("total_tokens", 0) for r in results) / n
    avg_think_tokens = sum(r.get("think_tokens", 0) for r in results) / n
    avg_answer_tokens = sum(r.get("answer_tokens", 0) for r in results) / n

    total_tokens_generated = sum(r.get("total_tokens", 0) for r in results)
    total_time_spent = sum(r.get("latency", 0.0) for r in results)
    tps = total_tokens_generated / total_time_spent if total_time_spent > 0 else 0.0

    exit_count = sum(1 for r in results if r.get("early_exited", False))
    exit_rate = (exit_count / n) * 100.0

    return {
        "num_samples": n,
        "correct": correct_count,
        "accuracy_pct": round(accuracy, 2),
        "avg_latency_s": round(avg_latency, 2),
        "avg_total_tokens": round(avg_total_tokens, 1),
        "avg_think_tokens": round(avg_think_tokens, 1),
        "avg_answer_tokens": round(avg_answer_tokens, 1),
        "tokens_per_second": round(tps, 2),
        "early_exit_count": exit_count,
        "early_exit_rate_pct": round(exit_rate, 2),
    }
