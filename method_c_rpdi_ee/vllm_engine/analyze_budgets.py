"""Phân tích kết quả vLLM theo ngân sách token (không cần GPU).

Ba phần:
    1. Bảng accuracy / token / tỷ lệ cạn ngân sách / tỷ lệ phanh theo từng ngân sách.
    2. Kiểm tra pipeline: so kết quả vLLM với kết quả HF đã có trên cùng câu hỏi và cùng ngân sách
       (accuracy lệch <= --tolerance điểm thì ĐẠT; kèm tỷ lệ trùng văn bản sinh ra).
    3. So sánh chính "target@B1 vs reference@B2": chênh lệch theo cặp, KTC bootstrap 90%/95%,
       McNemar, kết luận theo biên δ khai báo trước (--delta; thiếu thì bỏ qua phần này).

Ví dụ:
    python method_c_rpdi_ee/vllm_engine/analyze_budgets.py --budgets_file OUT/budgets.jsonl \\
        --hf_reference results/rpdi_results/controls_standard.jsonl results/rpdi_results/controls_rpdi.jsonl \\
        --validate standard=standard rpdi_legacy_W16_l1.5=rpdi_W16_l1.5 \\
        --compare rpdi_legacy_W16_l1.5@768:rpdi_W512_l2.0_wait@16384 --delta 3
"""

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from method_c_rpdi_ee.analyze_results import mcnemar_exact_p, wilson_ci  # noqa: E402

Index = Dict[Tuple[str, int], Dict[int, Dict[str, Any]]]

VERDICT_TEXT = {
    "better": "Hơn (cận dưới KTC 95% > 0)",
    "slightly_worse": "Kém hơn một chút, nằm trong biên δ (KTC 95% nằm trong (−δ, 0))",
    "worse": "Kém hơn, chưa chứng minh được nằm trong biên δ (cận dưới KTC 95% ≤ −δ)",
    "equivalent": "Tương đương trong biên ±δ (KTC 90% nằm trong (−δ, +δ), tương đương kiểm định TOST α = 0,05)",
    "noninferior": "Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương",
    "inconclusive": "Chưa kết luận được (khoảng tin cậy quá rộng so với δ)",
}
VERDICT_SHORT = {
    "better": "Hơn",
    "slightly_worse": "Kém hơn một chút (trong δ)",
    "worse": "Kém hơn (chưa chứng minh trong δ)",
    "equivalent": "Tương đương (±δ)",
    "noninferior": "Không kém hơn quá δ",
    "inconclusive": "Chưa kết luận",
}


# ----------------------------------------------------------------------------- thống kê
def paired_bootstrap_ci(a: Sequence[int], b: Sequence[int], n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Chênh lệch accuracy (a − b, điểm phần trăm) và KTC bootstrap percentile theo cặp."""
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("Hai mảng phải cùng độ dài và khác rỗng")
    d = [int(x) - int(y) for x, y in zip(a, b)]
    diff = 100.0 * sum(d) / n
    rng = random.Random(seed)
    stats = sorted(100.0 * sum(rng.choices(d, k=n)) / n for _ in range(n_boot))

    def q(p: float) -> float:
        return stats[min(n_boot - 1, max(0, int(round(p * (n_boot - 1)))))]

    return {"diff": diff, "ci90": (q(0.05), q(0.95)), "ci95": (q(0.025), q(0.975)), "n_boot": n_boot}


def classify(ci90: Tuple[float, float], ci95: Tuple[float, float], delta: float) -> str:
    """Kết luận so với tham chiếu theo biên δ khai báo trước (xem VERDICT_TEXT)."""
    lo90, hi90 = ci90
    lo95, hi95 = ci95
    if lo95 > 0:
        return "better"
    if hi95 < 0:
        return "slightly_worse" if lo95 > -delta else "worse"
    if lo90 > -delta and hi90 < delta:
        return "equivalent"
    if lo95 > -delta:
        return "noninferior"
    return "inconclusive"


def _acc(recs: Sequence[Dict[str, Any]]) -> float:
    return 100.0 * sum(bool(r["is_correct"]) for r in recs) / len(recs)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compare(target: Dict[int, Dict[str, Any]], reference: Dict[int, Dict[str, Any]], delta: float,
            n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    ids = sorted(set(target) & set(reference))
    a = [int(bool(target[i]["is_correct"])) for i in ids]
    b = [int(bool(reference[i]["is_correct"])) for i in ids]
    boot = paired_bootstrap_ci(a, b, n_boot=n_boot, seed=seed)
    t_better = sum(1 for x, y in zip(a, b) if x and not y)
    r_better = sum(1 for x, y in zip(a, b) if y and not x)
    tok_t = _mean([target[i]["total_tokens"] for i in ids])
    tok_r = _mean([reference[i]["total_tokens"] for i in ids])
    return {
        "n": len(ids),
        "acc_target": _acc([target[i] for i in ids]),
        "acc_reference": _acc([reference[i] for i in ids]),
        "diff": boot["diff"],
        "ci90": boot["ci90"],
        "ci95": boot["ci95"],
        "target_better": t_better,
        "reference_better": r_better,
        "mcnemar_p": mcnemar_exact_p(t_better, r_better),
        "delta": delta,
        "verdict": classify(boot["ci90"], boot["ci95"], delta),
        "avg_tokens_target": tok_t,
        "avg_tokens_reference": tok_r,
        "token_ratio": tok_t / tok_r if tok_r else None,
    }


def validate(vllm: Dict[int, Dict[str, Any]], hf: Dict[int, Dict[str, Any]], tolerance: float = 2.0) -> Dict[str, Any]:
    """So kết quả vLLM với HF trên các câu chung (cùng ngân sách)."""
    ids = sorted(set(vllm) & set(hf))
    if not ids:
        return {"n": 0, "passed": False}
    acc_v = _acc([vllm[i] for i in ids])
    acc_h = _acc([hf[i] for i in ids])
    sha = lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()  # noqa: E731
    with_text = [i for i in ids if "response_text" in hf[i] and "response_sha1" in vllm[i]]
    return {
        "n": len(ids),
        "acc_vllm": acc_v,
        "acc_hf": acc_h,
        "diff": acc_v - acc_h,
        "passed": abs(acc_v - acc_h) <= tolerance,
        "tolerance": tolerance,
        "correct_agree_pct": 100.0 * _mean([vllm[i]["is_correct"] == hf[i]["is_correct"] for i in ids]),
        "text_match_pct": 100.0 * _mean([vllm[i]["response_sha1"] == sha(hf[i]["response_text"]) for i in with_text])
        if with_text else None,
        "think_equal_pct": 100.0 * _mean([vllm[i]["think_tokens"] == hf[i]["think_tokens"] for i in ids]),
    }


def iso_budget(index: Index, method: str, ids: Sequence[int], target_acc: float) -> Optional[int]:
    """Ngân sách nhỏ nhất để `method` đạt accuracy >= target_acc trên cùng tập câu."""
    for B in sorted(b for m, b in index if m == method):
        recs = index[(method, B)]
        if all(i in recs for i in ids) and _acc([recs[i] for i in ids]) >= target_acc - 1e-9:
            return B
    return None


def parse_target(s: str) -> Tuple[str, int]:
    if "@" not in s:
        raise ValueError(f"Cần dạng phương_pháp@ngân_sách, nhận được {s!r}")
    method, budget = s.rsplit("@", 1)
    return method, int(budget)


# ----------------------------------------------------------------------------- I/O
def load_index(paths: Sequence[str]) -> Index:
    index: Index = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    index.setdefault((r["method"], int(r["max_new_tokens"])), {})[int(r["sample_id"])] = r
    return index


def summarize(recs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(recs)
    k = sum(bool(r["is_correct"]) for r in recs)
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "acc": 100.0 * k / n,
        "ci95": (100.0 * lo, 100.0 * hi),
        "avg_total_tokens": _mean([r["total_tokens"] for r in recs]),
        "hit_budget_pct": 100.0 * _mean([bool(r.get("hit_budget", r["total_tokens"] >= r["max_new_tokens"])) for r in recs]),
        "exit_pct": 100.0 * _mean([bool(r.get("early_exited")) for r in recs]),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phân tích kết quả RPDI-EE trên vLLM theo ngân sách token.")
    p.add_argument("--budgets_file", nargs="+", required=True, help="budgets.jsonl do run_vllm.py sinh ra")
    p.add_argument("--hf_reference", nargs="*", default=[], help="JSONL của run_controls.py (HF) để kiểm tra pipeline")
    p.add_argument("--validate", nargs="*", default=[], help="Cặp vllm_method=hf_method")
    p.add_argument("--validate_budget", type=int, default=768)
    p.add_argument("--tolerance", type=float, default=2.0, help="Ngưỡng lệch accuracy (điểm) để pipeline ĐẠT")
    p.add_argument("--compare", nargs="*", default=[], help="Dạng target@B:reference@B")
    p.add_argument("--delta", type=float, default=None, help="Biên δ (điểm %%) đã chốt TRƯỚC khi xem kết quả")
    p.add_argument("--baseline", type=str, default="standard", help="Phương pháp dùng cho tỷ lệ phục hồi và ngân sách tương đương")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_md", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args(argv)


def fmt_ci(ci: Tuple[float, float]) -> str:
    return f"[{ci[0]:+.1f}; {ci[1]:+.1f}]"


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = os.path.dirname(os.path.abspath(args.budgets_file[0]))
    out_md = args.output_md or os.path.join(out_dir, "analysis.md")
    out_json = args.output_json or os.path.join(out_dir, "analysis.json")

    index = load_index(args.budgets_file)
    if not index:
        raise SystemExit("Không có bản ghi nào trong --budgets_file")
    methods = sorted({m for m, _ in index})
    budgets = sorted({b for _, b in index})
    common = sorted(set.intersection(*(set(v) for v in index.values())))
    report: Dict[str, Any] = {"files": args.budgets_file, "n_common": len(common), "methods": methods,
                              "budgets": budgets, "table": {}, "validation": [], "comparisons": []}
    lines = [
        "# Kết quả RPDI-EE trên vLLM theo ngân sách token",
        "",
        f"- Số câu chung: **{len(common)}** (id {common[0]}–{common[-1]})" if common else "- Không có câu chung",
        "- Giải mã greedy. Mỗi cặp (câu, phương pháp) sinh một lần ở ngân sách lớn nhất; các ngân sách nhỏ hơn suy ra bằng cách cắt chuỗi token.",
        "",
    ]

    # 1. Bảng theo ngân sách
    for title, cell in (
        ("Accuracy % (token trung bình)", lambda s: f"{s['acc']:.1f} ({s['avg_total_tokens']:.0f})"),
        ("Tỷ lệ cạn ngân sách %", lambda s: f"{s['hit_budget_pct']:.1f}"),
        ("Tỷ lệ phanh trong pha suy nghĩ %", lambda s: f"{s['exit_pct']:.1f}"),
    ):
        lines += [f"## {title}", "", "| Phương pháp | " + " | ".join(str(b) for b in budgets) + " |",
                  "| :--- | " + " | ".join("---:" for _ in budgets) + " |"]
        for m in methods:
            cells = []
            for b in budgets:
                recs = index.get((m, b))
                if recs and all(i in recs for i in common) and common:
                    s = summarize([recs[i] for i in common])
                    report["table"].setdefault(m, {})[b] = s
                    cells.append(cell(s))
                else:
                    cells.append("–")
            lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
        lines.append("")

    # 2. Kiểm tra pipeline
    if args.validate:
        hf_index = load_index(args.hf_reference) if args.hf_reference else {}
        B = args.validate_budget
        lines += [f"## Kiểm tra pipeline: vLLM so với HF ở ngân sách {B}", "",
                  f"Đạt khi accuracy lệch không quá {args.tolerance:g} điểm trên cùng tập câu.", "",
                  "| vLLM | HF | n | Acc vLLM | Acc HF | Δ | Đồng thuận đúng/sai | Trùng văn bản | Trùng số token suy nghĩ | Kết quả |",
                  "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"]
        for pair in args.validate:
            vm, hm = pair.split("=", 1)
            v = validate(index.get((vm, B), {}), hf_index.get((hm, B), {}), tolerance=args.tolerance)
            report["validation"].append({"vllm": vm, "hf": hm, "budget": B, **v})
            if v["n"] == 0:
                lines.append(f"| `{vm}` | `{hm}` | 0 | – | – | – | – | – | – | KHÔNG CÓ DỮ LIỆU |")
                continue
            text = f"{v['text_match_pct']:.1f}%" if v["text_match_pct"] is not None else "–"
            lines.append(f"| `{vm}` | `{hm}` | {v['n']} | {v['acc_vllm']:.1f} | {v['acc_hf']:.1f} | {v['diff']:+.1f} | "
                         f"{v['correct_agree_pct']:.1f}% | {text} | {v['think_equal_pct']:.1f}% | "
                         f"{'ĐẠT' if v['passed'] else 'KHÔNG ĐẠT'} |")
        lines.append("")

    # 3. So sánh chính
    if args.compare and args.delta is None:
        lines += ["## So sánh chính", "", "> Bỏ qua: chưa khai báo `--delta`. Hãy chốt biên δ với thầy trước khi xem phần này.", ""]
    elif args.compare:
        lines += [f"## So sánh chính (biên δ = {args.delta:g} điểm, bootstrap {args.n_boot} lần)", "",
                  "| Target | Reference | n | Acc T | Acc R | Δ (T−R) | KTC 95% | KTC 90% | McNemar p | Token T/R | Phục hồi | `"
                  + args.baseline + "` cần ngân sách | Kết luận |",
                  "| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | :--- |"]
        notes = []
        for spec in args.compare:
            t_spec, r_spec = spec.split(":", 1)
            (tm, tb), (rm, rb) = parse_target(t_spec), parse_target(r_spec)
            target, ref = index.get((tm, tb), {}), index.get((rm, rb), {})
            if not target or not ref:
                lines.append(f"| `{tm}@{tb}` | `{rm}@{rb}` | 0 | – | – | – | – | – | – | – | – | – | KHÔNG CÓ DỮ LIỆU |")
                continue
            c = compare(target, ref, delta=args.delta, n_boot=args.n_boot, seed=args.seed)
            ids = sorted(set(target) & set(ref))
            base = index.get((args.baseline, tb), {})
            recovery = None
            if base and all(i in base for i in ids):
                acc_base = _acc([base[i] for i in ids])
                if c["acc_reference"] != acc_base:
                    recovery = (c["acc_target"] - acc_base) / (c["acc_reference"] - acc_base)
            iso = iso_budget(index, args.baseline, ids, c["acc_target"])
            c.update({"target": f"{tm}@{tb}", "reference": f"{rm}@{rb}", "recovery": recovery, "baseline_budget_to_match": iso})
            report["comparisons"].append(c)
            lines.append(
                f"| `{tm}@{tb}` | `{rm}@{rb}` | {c['n']} | {c['acc_target']:.1f} | {c['acc_reference']:.1f} | {c['diff']:+.1f} | "
                f"{fmt_ci(c['ci95'])} | {fmt_ci(c['ci90'])} | {c['mcnemar_p']:.3g} | "
                f"{c['token_ratio']:.2f} | {'–' if recovery is None else f'{100 * recovery:.0f}%'} | "
                f"{iso if iso is not None else '> ' + str(budgets[-1])} | {VERDICT_SHORT[c['verdict']]} |")
            notes.append(f"- `{tm}@{tb}` so với `{rm}@{rb}`: **{VERDICT_TEXT[c['verdict']]}**. "
                         f"Target đúng mà reference sai: {c['target_better']} câu; ngược lại: {c['reference_better']} câu.")
        lines += [""] + notes + ["",
                  "- *Phục hồi* = (Acc T − Acc baseline ở ngân sách của T) / (Acc R − Acc baseline ở ngân sách của T).",
                  f"- *Cần ngân sách* = ngân sách nhỏ nhất để `{args.baseline}` đạt accuracy bằng target trên cùng tập câu.", ""]

    md = "\n".join(lines)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(md)
    print(f"\n[*] Đã ghi {out_md} và {out_json}")


if __name__ == "__main__":
    main()
