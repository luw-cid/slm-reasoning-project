"""Lõi quyết định phanh của RPDI-EE, viết bằng Python thuần (không cần torch) để test được trên CPU.

Hai chế độ:

- ``paper``: bám đúng Algorithm 1 của bài gốc (Guan et al., 2026). Tại bước i (đã sinh t_i), nếu
  i >= W và t_i thuộc tập biên B thì tính GTF = S_global / i, LTF = S_local / W (tổng W entropy gần
  nhất), RPDI = LTF / GTF; phanh khi RPDI > lambda bằng cách chèn đúng một token ``</think>``.
  Việc kiểm tra diễn ra TRƯỚC khi nạp entropy của token kế tiếp.
  Riêng tập biên ``wait`` (điểm chuyển ý "Wait" của DEER): xét token SẮP sinh (argmax, vì giải mã
  greedy); nếu phanh thì ``</think>`` thay cho token "Wait", giống cách DEER dừng trước "Wait".

- ``legacy``: sao chép chính xác ``method_c_rpdi_ee/logits_processor.py::RPDILogitsProcessor`` (bản HF
  đã dùng cho các kết quả 768 token): nạp entropy của phân phối kế tiếp trước, rồi xét token liền
  trước có phải biên dấu câu không (step >= W); phanh bằng chuỗi ``"</think>\\n"``.

``RPDIMonitor.step`` được gọi một lần cho mỗi token sinh ra, trả về id token cần ép (hoặc None).
"""

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

BOUNDARY_PRESETS = ("wait", "newline", "punct")
LEGACY_PUNCT_SYMBOLS = (".", "\n", ";", "?", "!", ",")

Stats = Callable[[], Tuple[float, int]]


def build_boundary_ids(tokenizer, preset: str) -> FrozenSet[int]:
    """Tập id token biên theo preset.

    - ``wait``: token decode ra đúng "Wait" (bỏ khoảng trắng hai đầu), vd "Wait" và " Wait".
    - ``newline``: mọi token có ký tự xuống dòng khi decode.
    - ``punct``: tập dấu câu của bản HF cũ, kể cả biến thể có khoảng trắng đứng trước.
    """
    if preset == "punct":
        ids = set()
        for sym in LEGACY_PUNCT_SYMBOLS:
            ids.update(tokenizer.encode(sym, add_special_tokens=False))
            spaced = tokenizer.encode(" " + sym, add_special_tokens=False)
            if spaced:
                ids.add(spaced[-1])
        return frozenset(ids)
    if preset == "wait":
        match = lambda s: s.strip() == "Wait"  # noqa: E731
    elif preset == "newline":
        match = lambda s: "\n" in s  # noqa: E731
    else:
        raise ValueError(f"Preset biên không hợp lệ: {preset!r} (chọn trong {BOUNDARY_PRESETS})")
    return frozenset(i for i in range(len(tokenizer)) if match(tokenizer.decode([i])))


@dataclass(frozen=True)
class MethodSpec:
    key: str
    kind: str  # "standard" | "rpdi" | "rpdi_legacy"
    W: int
    lam: Optional[float]
    boundary: str


def parse_method(spec: str) -> MethodSpec:
    """``standard`` | ``rpdi:W:lambda[:wait|newline|punct]`` | ``rpdi_legacy:W:lambda``."""
    parts = spec.strip().split(":")
    name = parts[0]
    if name == "standard" and len(parts) == 1:
        return MethodSpec("standard", "standard", 0, None, "")
    try:
        if name == "rpdi" and len(parts) in (3, 4):
            W, lam = int(parts[1]), float(parts[2])
            boundary = parts[3] if len(parts) == 4 else "wait"
            if boundary not in BOUNDARY_PRESETS:
                raise ValueError(boundary)
            return MethodSpec(f"rpdi_W{W}_l{lam}_{boundary}", "rpdi", W, lam, boundary)
        if name == "rpdi_legacy" and len(parts) == 3:
            W, lam = int(parts[1]), float(parts[2])
            return MethodSpec(f"rpdi_legacy_W{W}_l{lam}", "rpdi_legacy", W, lam, "punct")
    except ValueError:
        pass
    raise ValueError(f"Phương pháp không hợp lệ: {spec!r}")


class RPDIMonitor:
    """Trạng thái giám sát của một request."""

    def __init__(
        self,
        *,
        W: int,
        lam: Optional[float],
        mode: str,
        boundary: str,
        boundary_ids: FrozenSet[int],
        think_close_id: int,
        close_ids: Sequence[int],
    ):
        if mode not in ("paper", "legacy"):
            raise ValueError(f"mode không hợp lệ: {mode!r}")
        self.W = W
        self.lam = lam
        self.mode = mode
        self.boundary = boundary
        self.boundary_ids = boundary_ids
        self.think_close_id = think_close_id
        self.close_ids = list(close_ids)

        self.entropies: List[float] = []
        self.S_global = 0.0
        self.S_local = 0.0
        self.early_exited = False
        self.exit_step = -1
        self.rpdi_at_exit = 0.0
        self.closed = False
        self._forced_idx: Optional[int] = None

    # ------------------------------------------------------------------ helpers
    def _append(self, h: float) -> None:
        self.entropies.append(h)
        self.S_global += h
        self.S_local += h
        n = len(self.entropies)
        if n > self.W:
            self.S_local -= self.entropies[n - self.W - 1]

    def _rpdi(self, i: int) -> float:
        gtf = self.S_global / i
        return (self.S_local / self.W) / gtf if gtf > 0 else 0.0

    def _start_exit(self, rpdi: float, think_tokens: int) -> int:
        self.early_exited = True
        self.exit_step = think_tokens
        self.rpdi_at_exit = rpdi
        self._forced_idx = 1
        return self.close_ids[0]

    # --------------------------------------------------------------------- step
    def step(self, prompt_ids: Sequence[int], output_ids: Sequence[int], stats: Stats) -> Optional[int]:
        if self.closed:
            return None

        # Đang ép chuỗi đóng thẻ
        if self._forced_idx is not None:
            if self._forced_idx < len(self.close_ids):
                tok = self.close_ids[self._forced_idx]
                self._forced_idx += 1
                return tok
            self.closed = True
            return None

        last = output_ids[-1] if len(output_ids) else (prompt_ids[-1] if len(prompt_ids) else None)
        if last == self.think_close_id:
            # Mô hình tự đóng </think>: ngừng giám sát, không phanh trong pha trả lời
            self.closed = True
            return None

        if self.mode == "legacy":
            h, _ = stats()
            self._append(h)
            step = len(self.entropies)
            if self.lam is not None and last in self.boundary_ids and step >= self.W:
                rpdi = self._rpdi(step)
                if rpdi > self.lam:
                    return self._start_exit(rpdi, think_tokens=step - 1)
            return None

        # mode == "paper"
        h, argmax = stats()
        i = len(self.entropies)
        if self.lam is not None and i >= self.W and i > 0:
            candidate = argmax if self.boundary == "wait" else (output_ids[-1] if len(output_ids) else None)
            if candidate in self.boundary_ids:
                rpdi = self._rpdi(i)
                if rpdi > self.lam:
                    return self._start_exit(rpdi, think_tokens=i)
        self._append(h)
        return None


def monitor_for(spec: MethodSpec, tokenizer, boundary_cache: Dict[str, FrozenSet[int]]) -> RPDIMonitor:
    """Tạo monitor cho một request theo MethodSpec (``standard`` = chỉ ghi entropy, không phanh)."""
    think_close = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_close) != 1:
        raise RuntimeError(f"'</think>' phải là một token, nhận được {think_close}")
    close_id = think_close[0]

    if spec.kind == "standard":
        return RPDIMonitor(W=0, lam=None, mode="paper", boundary="", boundary_ids=frozenset(),
                           think_close_id=close_id, close_ids=[close_id])

    if spec.boundary not in boundary_cache:
        boundary_cache[spec.boundary] = build_boundary_ids(tokenizer, spec.boundary)
    if spec.kind == "rpdi_legacy":
        return RPDIMonitor(W=spec.W, lam=spec.lam, mode="legacy", boundary=spec.boundary,
                           boundary_ids=boundary_cache[spec.boundary], think_close_id=close_id,
                           close_ids=tokenizer.encode("</think>\n", add_special_tokens=False))
    return RPDIMonitor(W=spec.W, lam=spec.lam, mode="paper", boundary=spec.boundary,
                       boundary_ids=boundary_cache[spec.boundary], think_close_id=close_id,
                       close_ids=[close_id])
