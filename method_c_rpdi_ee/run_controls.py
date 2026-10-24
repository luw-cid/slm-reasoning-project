"""Thực nghiệm đối chứng cho RPDI-EE trên tập held-out (mặc định câu 100–704 của GSM8K test).

Các phương pháp (chỉ định qua --methods, phân tách bằng dấu phẩy):
    standard            Sinh tự nhiên, không can thiệp.
    rpdi:W:lambda       RPDI-EE với cửa sổ W và ngưỡng lambda (vd: rpdi:16:1.5, rpdi:512:2.0).
    hardtrunc:N         Cắt cứng mù tại đúng N token suy nghĩ.
    random              Phanh tại biên câu đầu tiên sau một bước N ngẫu nhiên, N lấy từ phân phối
                        bước phanh thực tế của RPDI-EE (--random_ref). Kiểm tra tín hiệu entropy có
                        đóng góp gì so với phanh "mù" cùng phân phối thời điểm hay không.
    nothinking          Bỏ qua pha suy nghĩ (Ma et al., 2025): chèn sẵn đoạn suy nghĩ giả và </think>.

Kết quả ghi nối tiếp vào file JSONL (mỗi dòng = 1 câu x 1 phương pháp), tự động bỏ qua các cặp
đã chạy để có thể tiếp tục khi Colab bị ngắt.

Ví dụ:
    python method_c_rpdi_ee/run_controls.py --methods hardtrunc:130,random --output_file results/controls_768.jsonl
    python method_c_rpdi_ee/run_controls.py --max_new_tokens 2048 --end 300 \\
        --methods standard,rpdi:16:1.5,rpdi:512:2.0 --output_file results/budget_2048.jsonl

Mô hình sau RLVR (gắn adapter LoRA; tên phương pháp có tiền tố 'rlvr_' để gộp được với kết quả mô hình gốc):
    python method_c_rpdi_ee/run_controls.py --adapter_dir /content/drive/MyDrive/slm_rlvr/grpo_run1/checkpoint-300 \\
        --end 300 --methods standard,rpdi:16:1.5,hardtrunc:130 --output_file results/rlvr_eval.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.eval_dataset import load_gsm8k_test
from common.metrics import extract_answer_from_response, is_answer_correct
from common.model_loader import DEFAULT_MODEL_ID, format_reasoning_prompt

NOTHINKING_PREFILL = "\nOkay, I think I have finished thinking.\n</think>\n\n"
DEFAULT_DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "common", "data", "gsm8k_test_full.json")
DEFAULT_RANDOM_REF = os.path.join(os.path.dirname(__file__), "..", "results", "method_c_results_705.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thực nghiệm đối chứng RPDI-EE (cắt cứng, phanh ngẫu nhiên, NoThinking, ngân sách).")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--data_file", type=str, default=DEFAULT_DATA_FILE)
    parser.add_argument("--start", type=int, default=100, help="Chỉ số câu bắt đầu (mặc định 100: bỏ 100 câu đã dùng để chọn W, lambda).")
    parser.add_argument("--end", type=int, default=705, help="Chỉ số câu kết thúc (không bao gồm).")
    parser.add_argument("--methods", type=str, default="hardtrunc:130,random")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--random_ref", type=str, default=DEFAULT_RANDOM_REF,
                        help="File kết quả RPDI-EE dùng làm phân phối bước phanh cho phương pháp 'random'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Thư mục adapter LoRA (checkpoint GRPO hoặc adapter cuối); được gộp vào mô hình gốc trước khi sinh.")
    parser.add_argument("--method_prefix", type=str, default=None,
                        help="Tiền tố tên phương pháp trong kết quả (mặc định 'rlvr_' khi có --adapter_dir, ngược lại rỗng).")
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def parse_method(spec: str) -> Tuple[str, Dict[str, Any]]:
    """Chuyển chuỗi mô tả phương pháp thành (khóa duy nhất, tham số)."""
    parts = spec.strip().split(":")
    name = parts[0]
    if name == "rpdi":
        W, lam = int(parts[1]), float(parts[2])
        return f"rpdi_W{W}_l{lam}", {"type": "rpdi", "W": W, "lambda_th": lam}
    if name == "hardtrunc":
        N = int(parts[1])
        return f"hardtrunc_N{N}", {"type": "hardtrunc", "N": N}
    if name in ("standard", "random", "nothinking"):
        return name, {"type": name}
    raise ValueError(f"Phương pháp không hợp lệ: {spec}")


def load_rpdi_exit_steps(path: str) -> List[int]:
    """Lấy phân phối bước phanh của RPDI-EE, chỉ tính các lần phanh trong pha suy nghĩ.

    File 705 câu được sinh trước khi sửa lỗi phanh nhầm trong pha trả lời, nên loại các ca
    có bước phanh vượt quá độ dài suy nghĩ tự nhiên của baseline.
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)["sample_records"]
    steps = []
    for rec in records:
        std, rp = rec["standard"], rec["rpdi_ee"]
        if not rp.get("early_exited"):
            continue
        std_exhausted = std["total_tokens"] >= 768
        if std_exhausted or rp["exit_step"] <= std["think_tokens"] + 2:
            steps.append(int(rp["exit_step"]))
    if not steps:
        raise RuntimeError(f"Không tìm thấy bước phanh hợp lệ trong {path}")
    return steps


def load_done(path: str) -> Set[Tuple[int, str, int]]:
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["sample_id"], r["method"], r["max_new_tokens"]))
    return done


def run_one(model, tokenizer, question: str, cfg: Dict[str, Any], max_new_tokens: int,
            exit_at: Optional[int]) -> Dict[str, Any]:
    import torch
    from transformers import LogitsProcessorList

    from method_c_rpdi_ee.logits_processor import RPDILogitsProcessor, ScheduledExitLogitsProcessor

    prompt = format_reasoning_prompt(question)
    processor = None
    if cfg["type"] == "nothinking":
        prompt = prompt + NOTHINKING_PREFILL
    elif cfg["type"] == "rpdi":
        processor = RPDILogitsProcessor(tokenizer=tokenizer, W=cfg["W"], lambda_th=cfg["lambda_th"])
    elif cfg["type"] == "hardtrunc":
        # Ép đóng thẻ ở bước N+1 => đúng N token suy nghĩ (khớp simulate_hard_truncation)
        processor = ScheduledExitLogitsProcessor(tokenizer, exit_at=cfg["N"] + 1, align_to_boundary=False)
    elif cfg["type"] == "random":
        processor = ScheduledExitLogitsProcessor(tokenizer, exit_at=exit_at, align_to_boundary=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "do_sample": False,
    }
    if processor is not None:
        gen_kwargs["logits_processor"] = LogitsProcessorList([processor])

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    latency = time.time() - start

    generated = outputs[0][prompt_len:].tolist()
    total_tokens = len(generated)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    close_id = close_ids[0] if len(close_ids) == 1 else None
    if cfg["type"] == "nothinking":
        think_tokens = 0
    elif close_id is not None and close_id in generated:
        think_tokens = generated.index(close_id)
    else:
        think_tokens = total_tokens

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    pred = extract_answer_from_response(response_text)

    return {
        "predicted_answer": pred,
        "latency": round(latency, 3),
        "total_tokens": total_tokens,
        "think_tokens": think_tokens,
        "answer_tokens": total_tokens - think_tokens,
        "early_exited": bool(getattr(processor, "early_exited", False)),
        "exit_step": int(getattr(processor, "force_exit_step", -1)),
        "scheduled_exit_at": exit_at,
        # Bộ xử lý đã sửa ngừng giám sát khi mô hình tự sinh </think>, nên mọi lần phanh đều trong pha suy nghĩ
        "brake_in_think_phase": bool(getattr(processor, "early_exited", False)),
        "response_text": tokenizer.decode(generated, skip_special_tokens=False),
    }


def main() -> None:
    args = parse_args()
    prefix = args.method_prefix if args.method_prefix is not None else ("rlvr_" if args.adapter_dir else "")
    methods = [(prefix + key, cfg) for key, cfg in (parse_method(m) for m in args.methods.split(",") if m.strip())]
    dataset = load_gsm8k_test(filepath=args.data_file)[args.start:args.end]

    exit_steps: List[int] = []
    if any(cfg["type"] == "random" for _, cfg in methods):
        exit_steps = load_rpdi_exit_steps(args.random_ref)
        print(f"[*] Phân phối bước phanh tham chiếu: {len(exit_steps)} giá trị từ {args.random_ref}")

    done = load_done(args.output_file)
    todo = [(s, key, cfg) for s in dataset for key, cfg in methods
            if (s["id"], key, args.max_new_tokens) not in done]
    print(f"[*] {len(dataset)} câu x {len(methods)} phương pháp | Còn {len(todo)} lượt chạy (đã có {len(done)}).")
    if not todo:
        return

    from common.model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, torch_dtype=args.torch_dtype)
    if args.adapter_dir:
        from peft import PeftModel

        from common.peft_compat import disable_incompatible_torchao

        disable_incompatible_torchao()
        print(f"[*] Gắn và gộp adapter LoRA từ {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir).merge_and_unload()
        model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    t0 = time.time()
    with open(args.output_file, "a", encoding="utf-8") as fout:
        for i, (sample, key, cfg) in enumerate(todo, 1):
            exit_at = None
            if cfg["type"] == "random":
                # Seed theo từng câu để kết quả tái lập được và không phụ thuộc thứ tự chạy
                exit_at = random.Random(args.seed * 1_000_003 + sample["id"]).choice(exit_steps)
            res = run_one(model, tokenizer, sample["question"], cfg, args.max_new_tokens, exit_at)
            res["is_correct"] = is_answer_correct(res["predicted_answer"], sample["ground_truth"])
            record = {
                "sample_id": sample["id"],
                "method": key,
                "max_new_tokens": args.max_new_tokens,
                "ground_truth": sample["ground_truth"],
                "adapter_dir": args.adapter_dir,
                **res,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            elapsed = time.time() - t0
            print(f"[{i}/{len(todo)}] id={sample['id']} {key}: {'ĐÚNG' if res['is_correct'] else 'SAI'} "
                  f"| {res['total_tokens']} tok (think {res['think_tokens']}) | ETA {elapsed / i * (len(todo) - i) / 3600:.2f}h")


if __name__ == "__main__":
    main()
