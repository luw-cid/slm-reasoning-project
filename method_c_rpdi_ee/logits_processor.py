"""RPDI-EE (Reasoning Path Deviation Index - Early Exit) LogitsProcessor.
Giám sát entropy suy luận tại các token biên câu trong quá trình mô hình sinh chuỗi tư duy (<think>).
Cưỡng bức chuyển sang pha trả lời sớm (đóng </think>) nếu chỉ số RPDI vượt ngưỡng lambda.
"""

from typing import List, Optional, Set
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


DEFAULT_BOUNDARY_SYMBOLS = {".", "\n", ";", "?", "!", ","}


def build_boundary_ids(tokenizer, boundary_symbols: Optional[Set[str]] = None) -> Set[int]:
    """Tập ID token biên câu (kể cả biến thể có khoảng trắng đứng trước)."""
    if boundary_symbols is None:
        boundary_symbols = DEFAULT_BOUNDARY_SYMBOLS
    boundary_ids = set()
    for sym in boundary_symbols:
        boundary_ids.update(tokenizer.encode(sym, add_special_tokens=False))
        ids_space = tokenizer.encode(" " + sym, add_special_tokens=False)
        if ids_space:
            boundary_ids.add(ids_space[-1])
    return boundary_ids


class RPDILogitsProcessor(LogitsProcessor):
    """LogitsProcessor giám sát entropy cục bộ/toàn cục trong quá trình sinh thẻ <think>.

    Thuộc tính:
        W: Kích thước cửa sổ trượt để tính độ trôi chảy suy luận cục bộ (LTF).
        lambda_th: Ngưỡng phát hiện lệch hướng suy luận (RPDI = LTF / GTF).
        min_steps: Số token tối thiểu cần sinh trước khi cho phép kích hoạt dừng sớm.
        early_exited: Cờ đánh dấu thuật toán có kích hoạt dừng sớm hay không.
        force_exit_step: Bước token kích hoạt dừng sớm (-1 nếu không kích hoạt).
        rpdi_at_exit: Giá trị chỉ số RPDI tại thời điểm kích hoạt phanh.
        think_tokens_count: Số token đã sinh trong pha suy nghĩ trước khi đóng </think>.
    """

    def __init__(
        self,
        tokenizer,
        W: int = 16,
        lambda_th: float = 1.5,
        min_steps: Optional[int] = None,
        boundary_symbols: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.W = W
        self.lambda_th = lambda_th
        self.min_steps = min_steps if min_steps is not None else W

        # 1. Tiền xử lý ID của các token biên câu trên GPU để so khớp vector song song O(1)
        self.boundary_tensor = torch.tensor(
            sorted(build_boundary_ids(tokenizer, boundary_symbols)), dtype=torch.long
        )

        # 2. Token IDs cho chuỗi đóng thẻ: "</think>\n"
        self.close_tokens: List[int] = tokenizer.encode("</think>\n", add_special_tokens=False)
        self.close_len = len(self.close_tokens)
        # ID của thẻ '</think>' (token đơn trong tokenizer DeepSeek-R1-Distill) để nhận biết mô hình tự đóng thẻ
        think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.close_think_id: Optional[int] = think_close_ids[0] if len(think_close_ids) == 1 else None

        # Theo dõi trạng thái
        self.force_exit_step = -1
        self.forced_token_idx = 0
        self.early_exited = False
        self.rpdi_at_exit = 0.0

        # Lịch sử entropy & tổng tích lũy trượt O(1)
        self.entropy_history: List[float] = []
        self.S_global: float = 0.0
        self.S_local: float = 0.0
        self.step_count: int = 0
        self.think_tokens_count: int = 0
        self.closed_think: bool = False

    def reset(self) -> None:
        """Đặt lại trạng thái nội bộ cho một lượt sinh mới."""
        self.force_exit_step = -1
        self.forced_token_idx = 0
        self.early_exited = False
        self.rpdi_at_exit = 0.0
        self.entropy_history.clear()
        self.S_global = 0.0
        self.S_local = 0.0
        self.step_count = 0
        self.think_tokens_count = 0
        self.closed_think = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Đồng bộ thiết bị của tensor token biên với input_ids nếu cần
        if self.boundary_tensor.device != input_ids.device:
            self.boundary_tensor = self.boundary_tensor.to(input_ids.device)

        # Nếu đã hoàn thành việc ép đóng thẻ, giữ nguyên điểm số tự nhiên
        if self.closed_think:
            return scores

        # 1. Nếu đang trong quá trình ép sinh các token của chuỗi '</think>\n'
        if self.force_exit_step >= 0:
            if self.forced_token_idx < self.close_len:
                target_token_id = self.close_tokens[self.forced_token_idx]
                next_scores = torch.full_like(scores, float("-inf"))
                next_scores[:, target_token_id] = 0.0
                self.forced_token_idx += 1
                return next_scores
            else:
                self.closed_think = True
                return scores

        # Nếu mô hình đã tự đóng </think>, dừng giám sát để không phanh nhầm trong pha trả lời
        if self.close_think_id is not None and input_ids[0, -1].item() == self.close_think_id:
            self.closed_think = True
            return scores

        # 2. Tính toán entropy token: H(t_i) = - sum(p * log p)
        # Sử dụng scores của phần tử đầu tiên trong batch
        logits = scores[0]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy_val = -torch.sum(probs * log_probs, dim=-1)

        H_ti = entropy_val.item()
        self.entropy_history.append(H_ti)
        self.step_count += 1
        self.think_tokens_count += 1

        self.S_global += H_ti
        self.S_local += H_ti
        if self.step_count > self.W:
            self.S_local -= self.entropy_history[self.step_count - self.W - 1]

        # Kiểm tra token vừa được sinh ở bước trước
        last_token = input_ids[0, -1]

        # 3. Kiểm tra xem có phải token biên câu hay không trực tiếp trên GPU
        is_boundary_gpu = torch.any(self.boundary_tensor == last_token)

        if is_boundary_gpu.item():
            if self.step_count >= self.min_steps and self.step_count >= self.W:
                GTF = self.S_global / self.step_count
                LTF = self.S_local / self.W
                RPDI = LTF / GTF if GTF > 0 else 0.0

                if RPDI > self.lambda_th:
                    self.early_exited = True
                    self.force_exit_step = self.step_count
                    self.rpdi_at_exit = RPDI
                    self.forced_token_idx = 0

                    # Cưỡng bức sinh token đầu tiên của chuỗi '</think>\n' ngay lập tức
                    target_token_id = self.close_tokens[0]
                    next_scores = torch.full_like(scores, float("-inf"))
                    next_scores[:, target_token_id] = 0.0
                    self.forced_token_idx = 1
                    return next_scores

        return scores


class ScheduledExitLogitsProcessor(LogitsProcessor):
    """Baseline đối chứng: ép đóng </think> tại một bước định trước, KHÔNG dùng tín hiệu entropy.

    - align_to_boundary=False: cắt cứng đúng tại bước exit_at (Fixed Hard Truncation).
    - align_to_boundary=True: phanh tại token biên câu đầu tiên kể từ bước exit_at
      (dùng cho baseline phanh ngẫu nhiên, cùng cơ chế chèn thẻ như RPDI-EE).
    Nếu mô hình tự đóng </think> trước đó thì không can thiệp.
    """

    def __init__(
        self,
        tokenizer,
        exit_at: int,
        align_to_boundary: bool = False,
        boundary_symbols: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.exit_at = exit_at
        self.align_to_boundary = align_to_boundary
        self.boundary_ids = build_boundary_ids(tokenizer, boundary_symbols)
        self.close_tokens: List[int] = tokenizer.encode("</think>\n", add_special_tokens=False)
        think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.close_think_id: Optional[int] = think_close_ids[0] if len(think_close_ids) == 1 else None

        self.step_count = 0
        self.forced_token_idx = -1
        self.closed_think = False
        self.early_exited = False
        self.force_exit_step = -1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.closed_think:
            return scores

        # Đang ép sinh chuỗi '</think>\n'
        if self.forced_token_idx >= 0:
            if self.forced_token_idx < len(self.close_tokens):
                next_scores = torch.full_like(scores, float("-inf"))
                next_scores[:, self.close_tokens[self.forced_token_idx]] = 0.0
                self.forced_token_idx += 1
                return next_scores
            self.closed_think = True
            return scores

        last_token = input_ids[0, -1].item()
        if self.close_think_id is not None and last_token == self.close_think_id:
            self.closed_think = True
            return scores

        self.step_count += 1
        if self.step_count >= self.exit_at and (
            not self.align_to_boundary or last_token in self.boundary_ids
        ):
            self.early_exited = True
            self.force_exit_step = self.step_count
            next_scores = torch.full_like(scores, float("-inf"))
            next_scores[:, self.close_tokens[0]] = 0.0
            self.forced_token_idx = 1
            return next_scores

        return scores
