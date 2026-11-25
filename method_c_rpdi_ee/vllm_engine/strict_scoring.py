"""Độ nhạy với cách chấm lượt chưa xong (phản biện B2.5), không cần GPU.

Ba cách chấm một bản ghi trong budgets.jsonl:
    lenient   đúng nếu số cuối cùng trích được khớp đáp án (cách chấm trong bài)
    closed    như lenient, nhưng chỉ khi đã đóng </think> trong ngân sách
    finished  như lenient, nhưng chỉ khi lượt sinh kết thúc trước ngân sách (không chạm trần)

Ví dụ:
    python method_c_rpdi_ee/vllm_engine/strict_scoring.py --budgets_file results/gsm8k_500/budgets.jsonl \\
        --output_md results/gsm8k_500/strict_scoring.md
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from method_c_rpdi_ee.analyze_results import mcnemar_exact_p  # noqa: E402
from method_c_rpdi_ee.vllm_engine.analyze_budgets import paired_bootstrap_ci  # noqa: E402

MODES = ("lenient", "closed", "finished")


def score(r: Dict[str, Any], mode: str) -> bool:
    if mode == "lenient":
        return bool(r["is_correct"])
    if mode == "closed":
        return bool(r["is_correct"]) and r["answer_tokens"] > 0
    if mode == "finished":
        return bool(r["is_correct"]) and not r["hit_budget"]
    raise ValueError(mode)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budgets_file", required=True)
    p.add_argument("--methods", default="standard,rpdi_legacy_W16_l1.5")
    p.add_argument("--compare", nargs="*", default=[
        "rpdi_legacy_W16_l1.5@768:standard@768",
        "rpdi_legacy_W16_l1.5@1024:standard@16384",
        "rpdi_legacy_W16_l1.5@16384:standard@16384",
    ])
    p.add_argument("--output_md", default=None)
    args = p.parse_args(argv)

    R: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    with open(args.budgets_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                R[(r["method"], int(r["max_new_tokens"]), int(r["sample_id"]))] = r
    ids = sorted({k[2] for k in R})
    budgets = sorted({k[1] for k in R})
    n = len(ids)
    out: List[str] = [f"# Độ nhạy với cách chấm (n = {n})", "",
                      "| Phương pháp | Ngân sách | lenient | closed | finished |", "| :--- | ---: | ---: | ---: | ---: |"]
    for m in args.methods.split(","):
        for b in budgets:
            if (m, b, ids[0]) in R:
                accs = [100 * sum(score(R[(m, b, i)], md) for i in ids) / n for md in MODES]
                out.append(f"| `{m}` | {b} | " + " | ".join(f"{a:.1f}" for a in accs) + " |")
    out += ["", "| So sánh | Cách chấm | Acc T | Acc R | Δ | KTC 95% (bootstrap) | McNemar p |",
            "| :--- | :--- | ---: | ---: | ---: | :---: | ---: |"]
    for c in args.compare:
        (t, tb), (rf, rb) = [(x.rsplit("@", 1)[0], int(x.rsplit("@", 1)[1])) for x in c.split(":")]
        for md in MODES:
            a = [int(score(R[(t, tb, i)], md)) for i in ids]
            b = [int(score(R[(rf, rb, i)], md)) for i in ids]
            ci = paired_bootstrap_ci(a, b)
            only_a = sum(x and not y for x, y in zip(a, b))
            only_b = sum(y and not x for x, y in zip(a, b))
            out.append(f"| `{t}@{tb}` vs `{rf}@{rb}` | {md} | {100 * sum(a) / n:.1f} | {100 * sum(b) / n:.1f} | "
                       f"{ci['diff']:+.1f} | [{ci['ci95'][0]:+.1f}; {ci['ci95'][1]:+.1f}] | {mcnemar_exact_p(only_a, only_b):.3g} |")
    text = "\n".join(out) + "\n"
    print(text)
    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
