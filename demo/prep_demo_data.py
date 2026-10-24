"""Tiền xử lý dữ liệu cho trang demo RPDI-EE. Chạy: python prep.py <repo_root> <tokenizer.json> <out.json>"""
import json
import sys

from tokenizers import Tokenizer

repo, tok_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo)
# Lấy nguyên văn hàm simulate_rpdi_exit từ repo (không import package vì package cần torch)
import ast  # noqa: E402
from typing import List, Optional, Set  # noqa: E402,F401

_src = open(f"{repo}/method_c_rpdi_ee/ablation/step2_ablate_grid.py", encoding="utf-8").read()
_fn = next(n for n in ast.parse(_src).body if isinstance(n, ast.FunctionDef) and n.name == "simulate_rpdi_exit")
exec(compile(ast.Module(body=[_fn], type_ignores=[]), "step2", "exec"))

tok = Tokenizer.from_file(tok_path)
load = lambda p: json.load(open(f"{repo}/{p}", encoding="utf-8"))

traces = load("results/baseline_traces.json")
boundary = traces["metadata"]["boundary_token_ids"]
real = {r["sample_id"]: r for r in load("results/method_c_results_705.json")["sample_records"]}

trace_out = []
checks = []
for t in traces["traces"]:
    sid = t["sample_id"]
    end = t["natural_think_end_idx"]
    n = end if end is not None else len(t["entropies"])
    ids = t["generated_token_ids"][:n]
    ent = t["entropies"][:n]
    r = real[sid]
    trace_out.append({
        "id": sid,
        "q": t["question"],
        "gt": t["ground_truth"],
        "end": end,  # None = cạn ngân sách 768 token trong pha suy nghĩ
        "tok": [tok.decode([i]) for i in ids],
        "b": [1 if i in boundary else 0 for i in ids],
        "H": [round(x, 3) for x in ent],
        "std": {"pred": r["standard"]["predicted_answer"], "ok": r["standard"]["is_correct"],
                "think": r["standard"]["think_tokens"], "total": r["standard"]["total_tokens"]},
        "rpdi": {"pred": r["rpdi_ee"]["predicted_answer"], "ok": r["rpdi_ee"]["is_correct"],
                 "think": r["rpdi_ee"]["think_tokens"], "exited": r["rpdi_ee"]["early_exited"],
                 "total": r["rpdi_ee"]["total_tokens"]},
    })
    # Điểm phanh tham chiếu từ Python (đã làm tròn entropy như dữ liệu nhúng) để đối chiếu với JS
    for W, lam in [(16, 1.5), (32, 2.0), (8, 1.2)]:
        checks.append([sid, W, lam, simulate_rpdi_exit(trace_out[-1]["H"], t["generated_token_ids"],
                                                       set(boundary), end, W, lam)])

questions = {q["id"]: q["question"] for q in load("common/data/gsm8k_test_full.json")}
methods = [("standard", "standard"), ("hardtrunc", "hardtrunc"), ("random", "random"), ("rpdi", "rpdi")]
ctrl = {}
for key, fname in methods:
    for line in open(f"{repo}/results/rpdi_results/controls_{fname}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        c = ctrl.setdefault(r["sample_id"], {"id": r["sample_id"], "q": questions[r["sample_id"]], "gt": r["ground_truth"]})
        c[key] = {"pred": r["predicted_answer"], "ok": r["is_correct"], "think": r["think_tokens"],
                  "ans": r["answer_tokens"], "total": r["total_tokens"],
                  "exit": r["exit_step"] if r["early_exited"] else None, "text": r["response_text"]}

json.dump({"traces": trace_out, "controls": [ctrl[k] for k in sorted(ctrl)], "checks": checks},
          open(out_path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("traces", len(trace_out), "controls", len(ctrl))
