"""Vá tương thích PEFT trên Google Colab.

Colab cài sẵn torchao 0.10.0. Một số phiên bản PEFT kiểm tra torchao khi gắn LoRA và báo ImportError
nếu torchao < 0.16.0, kể cả khi mô hình không dùng torchao (PEFT bản mới đã đổi thành cảnh báo).
Dự án không dùng torchao, nên tắt nhánh này để LoRA được gắn bằng dispatcher thông thường.
"""


def disable_incompatible_torchao() -> None:
    """Gọi trước khi tạo/nạp LoRA (get_peft_model, GRPOTrainer(peft_config=...), PeftModel.from_pretrained)."""
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_lora_torchao
    except ImportError:
        return

    try:
        peft_import_utils.is_torchao_available()
    except ImportError as exc:
        print(f"[PEFT] Bỏ qua torchao không tương thích ({exc}).")
        peft_import_utils.is_torchao_available = lambda: False
        peft_lora_torchao.is_torchao_available = lambda: False
