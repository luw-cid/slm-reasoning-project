"""Chấm đáp án MATH/MATH500: lấy nội dung \\boxed{} cuối cùng và so khớp sau chuẩn hóa chuỗi LaTeX.

Chuẩn hóa theo hàm ``strip_string`` / ``is_equiv`` của bộ MATH (Hendrycks et al., 2021), thêm so khớp số
(thập phân, phân số đơn giản) và bỏ ``\\left``/``\\right``, ``\\dfrac``, đơn vị ``\\text{}`` cuối.
"""
import re
from fractions import Fraction
from typing import Optional


def last_boxed(text: str) -> Optional[str]:
    """Nội dung của \\boxed{...} (hoặc \\fbox{...}) cuối cùng, hỗ trợ ngoặc lồng nhau."""
    if not text:
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])  # \boxed 5
        return m.group(1) if m else None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def _fix_fracs(s: str) -> str:
    subs = s.split("\\frac")
    out = subs[0]
    for sub in subs[1:]:
        out += "\\frac"
        if sub.startswith("{") or len(sub) < 2:
            out += sub
            continue
        a, b, rest = sub[0], sub[1], sub[2:]
        out += f"{{{a}}}{{{b}}}{rest}" if b != "{" else f"{{{a}}}{b}{rest}"
    return out


def _fix_a_slash_b(s: str) -> str:
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}" if m else s


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def strip_string(s: str) -> str:
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    # "x = 5" -> "5" (chỉ khi vế trái ngắn)
    parts = s.split("=")
    if len(parts) == 2 and len(parts[0].strip()) <= 2:
        s = parts[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    s = s.rstrip(".")
    # đơn vị chữ ở cuối: "5inches" -> "5"
    m = re.fullmatch(r"(-?[\d.,]+)[a-zA-Z]+", s)
    if m:
        s = m.group(1)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def _as_number(s: str) -> Optional[Fraction]:
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    try:
        if m:
            v = Fraction(int(m.group(2)), int(m.group(3)))
            return -v if m.group(1) else v
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def is_equiv(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    a, b = strip_string(pred), strip_string(gt)
    if a == b:
        return True
    na, nb = _as_number(a), _as_number(b)
    return na is not None and nb is not None and abs(float(na - nb)) < 1e-6
