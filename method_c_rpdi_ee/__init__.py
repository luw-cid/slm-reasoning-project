"""Module hiện thực phương pháp RPDI-EE (Reasoning Path Deviation Index - Early Exit).
Bao gồm RPDILogitsProcessor và các hàm thực thi sinh câu trả lời so sánh đối chứng.
"""

from .logits_processor import RPDILogitsProcessor
from .runner import generate_standard, generate_rpdi

__all__ = [
    "RPDILogitsProcessor",
    "generate_standard",
    "generate_rpdi",
]
