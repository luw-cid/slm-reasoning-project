"""Chẩn đoán chênh lệch giữa mô phỏng offline (step2: 78%) và chạy thật (lần chạy 705 câu: 73%)
trên cùng 100 câu đầu với RPDI(W=16, lambda=1.5).

Phân tích offline (không cần GPU) cho thấy điểm phanh của mô phỏng và chạy thật trùng nhau ở
93/100 câu; 7 câu lệch là do lỗi phanh trong pha trả lời (đã sửa). Vậy phần chênh lệch accuracy
còn lại nằm ở pha sinh câu trả lời. Script này kiểm chứng trên GPU:

    A. Sinh tiếp từ prefix (cách của step2): prefill [prompt + think[:k] + '</think>\\n'] rồi generate.
    B. Sinh trong một lệnh generate() duy nhất với RPDILogitsProcessor (cách của runner, đã sửa lỗi).

Với mỗi câu có phanh, so sánh đáp án và vị trí token đầu tiên mà hai cách bắt đầu khác nhau.
Nếu hai cách khác nhau dù cùng điểm phanh => sai khác số học (bf16, prefill vs. decode từng bước),
và kết quả mô phỏng không thể dùng thay chạy thật.
"""

import argparse
import json
import os
import sys
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.metrics import extract_answer_from_response, is_answer_correct
from common.model_loader import DEFAULT_MODEL_ID, load_model_and_tokenizer
from method_c_rpdi_ee.ablation.step2_ablate_grid import simulate_rpdi_exit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chẩn đoán chênh lệch mô phỏng offline và chạy thật của RPDI-EE.")
    parser.add_argument("--traces_file", type=str, default=os.path.join(RESULTS_DIR, "baseline_traces.json"))
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--W", type=int, default=16)
    parser.add_argument("--lambda_th", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=os.path.join(RESULTS_DIR, "diagnose_sim_vs_online.json"))
    return parser.parse_args()


def first_divergence(a, b) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    import torch
    from transformers import LogitsProcessorList

    from method_c_rpdi_ee.logits_processor import RPDILogitsProcessor

    args = parse_args()
    with open(args.traces_file, "r", encoding="utf-8") as f:
        traces = json.load(f)
    boundary_ids = set(traces["metadata"]["boundary_token_ids"])
    max_new_tokens = traces["metadata"]["max_new_tokens"]
    samples = traces["traces"][: args.limit] if args.limit else traces["traces"]

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, torch_dtype=args.torch_dtype)
    close_tokens = tokenizer.encode("</think>\n", add_special_tokens=False)

    rows = []
    acc_sim = acc_online = 0
    for idx, s in enumerate(samples):
        k = simulate_rpdi_exit(s["entropies"], s["generated_token_ids"], boundary_ids,
                               s["natural_think_end_idx"], args.W, args.lambda_th)
        prompt_ids = s["prompt_token_ids"]

        # B. Chạy thật trong một lệnh generate()
        proc = RPDILogitsProcessor(tokenizer=tokenizer, W=args.W, lambda_th=args.lambda_th)
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            out_b = model.generate(input_ids=inp, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                   eos_token_id=tokenizer.eos_token_id,
                                   logits_processor=LogitsProcessorList([proc]))
        gen_b = out_b[0][len(prompt_ids):].tolist()
        pred_b = extract_answer_from_response(tokenizer.decode(out_b[0], skip_special_tokens=False))
        ok_b = is_answer_correct(pred_b, s["ground_truth"])

        if k is None:
            # Không phanh: mô phỏng dùng nguyên kết quả baseline
            gen_a = s["generated_token_ids"]
            ok_a = s["standard_metrics"]["is_correct"]
            pred_a = s["standard_metrics"]["predicted_answer"]
        else:
            # A. Sinh tiếp từ prefix như step2
            prefix = prompt_ids + s["generated_token_ids"][:k] + close_tokens
            remaining = max(0, max_new_tokens - k - len(close_tokens))
            inp_a = torch.tensor([prefix], dtype=torch.long, device=model.device)
            with torch.no_grad():
                out_a = model.generate(input_ids=inp_a, max_new_tokens=remaining, do_sample=False, use_cache=True,
                                       eos_token_id=tokenizer.eos_token_id)
            gen_a = s["generated_token_ids"][:k] + close_tokens + out_a[0][len(prefix):].tolist()
            pred_a = extract_answer_from_response(tokenizer.decode(out_a[0], skip_special_tokens=False))
            ok_a = is_answer_correct(pred_a, s["ground_truth"])

        acc_sim += ok_a
        acc_online += ok_b
        div = first_divergence(gen_a, gen_b)
        rows.append({
            "sample_id": s["sample_id"],
            "sim_think_tokens": k,
            "online_exit_step": proc.force_exit_step if proc.early_exited else None,
            "first_divergence_token": div,
            "sim_pred": pred_a, "online_pred": pred_b,
            "sim_correct": bool(ok_a), "online_correct": bool(ok_b),
        })
        flag = "" if div is None else f" | lệch từ token {div}"
        print(f"[{idx + 1}/{len(samples)}] id={s['sample_id']} sim={'Đ' if ok_a else 'S'} online={'Đ' if ok_b else 'S'}{flag}")

    n = len(samples)
    summary = {
        "n": n,
        "sim_accuracy_pct": round(100 * acc_sim / n, 2),
        "online_accuracy_pct": round(100 * acc_online / n, 2),
        "identical_outputs": sum(r["first_divergence_token"] is None for r in rows),
        "different_correctness": sum(r["sim_correct"] != r["online_correct"] for r in rows),
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
