"""Module hiện thực phương pháp RPDI-EE (Reasoning Path Deviation Index - Early Exit).
Bao gồm RPDILogitsProcessor và các hàm thực thi sinh câu trả lời so sánh đối chứng.

Import lười (PEP 562): các module con không cần torch (vllm_engine.rpdi_core, analyze_results, ...)
vẫn import được trên máy không có torch/transformers.
"""

from importlib import import_module

_LAZY = {
    "RPDILogitsProcessor": ".logits_processor",
    "generate_standard": ".runner",
    "generate_rpdi": ".runner",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
