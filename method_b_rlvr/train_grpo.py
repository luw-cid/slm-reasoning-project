"""Script huấn luyện mô hình suy luận SLM bằng thuật toán GRPO (Group Relative Policy Optimization) kết hợp LoRA.
Áp dụng trên mô hình: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Tối ưu hóa chạy trơn tru trên Google Colab (GPU T4/A100/L4) và môi trường cục bộ.
"""

import argparse
import json
import os
import sys
from typing import Optional

# Hỗ trợ ký tự tiếng Việt an toàn trên console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.model_loader import DEFAULT_MODEL_ID
from common.seed import set_seed
from method_b_rlvr.prepare_train_data import DEFAULT_OUTPUT_PATH, prepare_rlvr_dataset
from method_b_rlvr.reward_funcs import get_reward_functions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện DeepSeek-R1-1.5B bằng RLVR qua GRPO + LoRA."
    )
    # Cấu hình mô hình và dữ liệu
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"ID mô hình base từ Hugging Face (mặc định: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Đường dẫn file dữ liệu JSON đã chuẩn bị (nếu chưa có sẽ tự động tạo).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=400,
        help="Số lượng mẫu huấn luyện nếu cần tự sinh (mặc định: 400).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/grpo_deepseek_r1_lora",
        help="Thư mục lưu checkpoints trung gian trong quá trình huấn luyện.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="results/rlvr_adapter",
        help="Thư mục lưu trọng số LoRA adapter cuối cùng sau khi hoàn tất.",
    )

    # Siêu tham số thuật toán GRPO
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Tốc độ học (Learning Rate) cho tối ưu hóa chính sách (mặc định: 5e-6).",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Số lượng lời giải G sinh ra cho mỗi prompt trong nhóm GRPO (mặc định: 4).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Kích thước batch trên mỗi GPU (mặc định: 1 để an toàn VRAM).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Số bước tích lũy gradient (mặc định: 4).",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=256,
        help="Độ dài token prompt tối đa (mặc định: 256).",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=512,
        help="Độ dài token sinh hoàn thiện tối đa cho mỗi generation (mặc định: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Nhiệt độ sampling trong GRPO để khám phá các chuỗi suy luận đa dạng (mặc định: 0.8).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Số epoch huấn luyện (mặc định: 1).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Số bước huấn luyện tối đa (-1 để chạy theo epoch).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=25,
        help="Tần suất lưu checkpoint trung gian (mặc định: 25).",
    )

    # Cấu hình LoRA (PEFT)
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Thứ hạng r của LoRA (mặc định: 16).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Hệ số lora_alpha (mặc định: 32).",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Tỷ lệ dropout của LoRA (mặc định: 0.05).",
    )

    # Quản lý bộ nhớ phần cứng
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=False,
        help="Kích hoạt định dạng QLoRA 4-bit (khuyến nghị khi chạy trên GPU T4 của Google Colab).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên toàn cục (mặc định: 42).",
    )
    parser.add_argument(
        "--include_brevity_reward",
        action="store_true",
        default=False,
        help="Bật thêm hàm phần thưởng khuyến khích suy nghĩ súc tích (mặc định: False - chuẩn DeepSeek-R1).",
    )

    return parser.parse_args()


def load_or_create_dataset(args: argparse.Namespace):
    """Đảm bảo tập dữ liệu huấn luyện đã sẵn sàng."""
    from datasets import Dataset

    if not os.path.exists(args.data_file):
        print(f"[Dữ liệu] Chưa tìm thấy file dữ liệu tại '{args.data_file}'. Đang tự động tạo mới...")
        prepare_rlvr_dataset(num_samples=args.num_samples, output_path=args.data_file, seed=args.seed)

    print(f"[Dữ liệu] Đang nạp tập dữ liệu từ '{args.data_file}'...")
    with open(args.data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Chuẩn bị dataset Hugging Face với 2 cột bắt buộc cho GRPO: 'prompt' và 'ground_truth'
    data_dict = {
        "prompt": [item["prompt"] for item in raw_data],
        "ground_truth": [item["ground_truth"] for item in raw_data],
        "question": [item.get("question", "") for item in raw_data],
    }
    hf_dataset = Dataset.from_dict(data_dict)
    print(f"[Dữ liệu] Đã nạp thành công {len(hf_dataset)} mẫu dữ liệu huấn luyện.")
    return hf_dataset


def train_grpo():
    args = parse_args()
    set_seed(args.seed)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, TaskType
    from trl import GRPOTrainer, GRPOConfig

    print("==========================================================================================")
    print("                    KHỞI CHẠY HUẤN LUYỆN RLVR (GRPO + LoRA) - SLM REASONING               ")
    print(f" - Mô hình base: {args.model_id}")
    print(f" - Số lượng mẫu: {args.num_samples} | Số generation mỗi nhóm G: {args.num_generations}")
    print(f" - Tốc độ học (LR): {args.learning_rate} | Chế độ 4-bit: {args.use_4bit}")
    print(f" - Thư mục lưu adapter: {args.adapter_dir}")
    print("==========================================================================================")

    # 1. Chuẩn bị tập dữ liệu
    train_dataset = load_or_create_dataset(args)

    # 2. Tải Tokenizer
    print(f"[ModelLoader] Đang nạp tokenizer từ '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Cấu hình kiểu dữ liệu và lượng tử hóa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    model_kwargs = {
        "device_map": "auto" if torch.cuda.is_available() else None,
    }

    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        print("[ModelLoader] Kích hoạt chế độ QLoRA 4-bit (BitsAndBytesConfig)...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else torch.float16

    print(f"[ModelLoader] Đang nạp mô hình nền '{args.model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # 4. Cấu hình LoRA Adapter (PEFT)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

    # 5. Cấu hình hệ hàm phần thưởng
    reward_functions = get_reward_functions(include_brevity=args.include_brevity_reward)
    print(f"[Reward] Đã nạp {len(reward_functions)} hàm phần thưởng:")
    for fn in reward_functions:
        print(f"  - {fn.__name__}")

    # 6. Thiết lập GRPOConfig
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.adapter_dir, exist_ok=True)

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        warmup_ratio=0.1,
        remove_unused_columns=False,
    )

    # 7. Khởi tạo GRPOTrainer
    print("[Trainer] Khởi tạo GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 8. Bắt đầu tiến trình huấn luyện
    print("\n🚀 [Trainer] BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN GRPO...")
    trainer.train()

    # 9. Lưu Adapter và Tokenizer cuối cùng
    print(f"\n[Lưu trữ] Đang lưu trọng số LoRA adapter vào '{args.adapter_dir}'...")
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    # Ghi lại tệp thông số cấu hình huấn luyện phục vụ viết báo cáo luận văn
    training_info = {
        "base_model": args.model_id,
        "algorithm": "GRPO (Group Relative Policy Optimization) with LoRA",
        "num_samples": len(train_dataset),
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "use_4bit": args.use_4bit,
        "adapter_path": args.adapter_dir,
    }
    with open(os.path.join(args.adapter_dir, "training_info.json"), "w", encoding="utf-8") as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)

    print(f"✅ [Hoàn tất] Huấn luyện GRPO thành công! Adapter đã sẵn sàng tại: {args.adapter_dir}")


if __name__ == "__main__":
    train_grpo()
