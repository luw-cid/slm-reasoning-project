"""Module tải Model và Tokenizer cho DeepSeek-R1-Distill-Qwen-1.5B.
Hỗ trợ phân bổ GPU/CPU, độ chính xác bfloat16/float16, và chuẩn hóa định dạng prompt.
"""

from typing import Optional, Tuple
import os

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def format_reasoning_prompt(question: str) -> str:
    """Định dạng prompt đầu vào để kích hoạt chuỗi suy luận (chain-of-thought) cho DeepSeek-R1.

    Định dạng:
    User: {question}
    Assistant: <think>
    """
    return f"User: {question.strip()}\nAssistant: <think>"


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    device_map: str = "auto",
    torch_dtype: Optional[str] = "auto",
    hf_token: Optional[str] = None,
) -> Tuple[any, any]:
    """Tải mô hình và tokenizer đã được huấn luyện sẵn với kiểu dữ liệu tiết kiệm bộ nhớ.

    Tham số:
        model_id: ID kho lưu trữ mô hình trên Hugging Face.
        device_map: Chiến lược ánh xạ thiết bị ('auto', 'cuda', 'cpu').
        torch_dtype: Kiểu dữ liệu độ chính xác ('auto', 'bfloat16', 'float16').
        hf_token: Token Hugging Face tùy chọn.

    Trả về:
        Tuple (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = hf_token or os.environ.get("HF_TOKEN")

    print(f"[ModelLoader] Đang tải tokenizer từ '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    # Xác định kiểu dữ liệu (độ chính xác) tối ưu cho phần cứng
    if torch_dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[ModelLoader] Đang tải trọng số mô hình với torch_dtype={dtype}, device_map={device_map}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        token=token,
    )

    model.eval()
    return model, tokenizer
