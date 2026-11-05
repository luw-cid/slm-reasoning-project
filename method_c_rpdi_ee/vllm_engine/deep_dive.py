"""Phân tích sâu một lần chạy run_vllm.py: so sánh cặp, cơ chế phanh, vì sao cấu hình bài gốc không phanh.

Không cần GPU hay tokenizer: chỉ đọc raw.jsonl và budgets.jsonl.

    python method_c_rpdi_ee/vllm_engine/deep_dive.py --run_dir results/gsm8k_500 \
        --compare_run results/pilot_200 --output_json results/gsm8k_500/deep_dive.json
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

STD = "standard"
OURS = "rpdi_legacy_W16_l1.5"
PAPER = ("rpdi_W512_l2.0_wait", "rpdi_W512_l2.0_newline")
TOP = 16384


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def quantiles(xs: Sequence[float]) -> Dict[str, float]:
    return {"n": len(xs), "min": min(xs), "p10": pct(xs, .1), "p25": pct(xs, .25), "median": pct(xs, .5),
            "p75": pct(xs, .75), "p90": pct(xs, .9), "p95": pct(xs, .95), "max": max(xs)} if xs else {"n": 0}


def mcnemar_p(b: int, c: int) -> float:
    """McNemar chính xác hai phía (nhị thức trên các cặp bất đồng)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(t: List[bool], r: List[bool], n_boot: int, rng: random.Random) -> Dict[str, Any]:
    d = [int(a) - int(b) for a, b in zip(t, r)]
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    b = sum(1 for a, c in zip(t, r) if a and not c)
    c = sum(1 for a, cc in zip(t, r) if cc and not a)
    return {"n": n, "acc_t": 100 * sum(t) / n, "acc_r": 100 * sum(r) / n, "delta": 100 * sum(d) / n,
            "ci95": [100 * boots[int(.025 * n_boot)], 100 * boots[int(.975 * n_boot) - 1]],
            "ci90": [100 * boots[int(.05 * n_boot)], 100 * boots[int(.95 * n_boot) - 1]],
            "only_t": b, "only_r": c, "mcnemar_p": mcnemar_p(b, c)}


def think_len(r: Dict[str, Any]) -> int:
    return r["close_index"] if r["close_index"] is not None else len(r["output_ids"])


def max_rpdi(h: Sequence[float], W: int) -> Optional[float]:
    """RPDI lớn nhất trên mọi vị trí i >= W (định nghĩa Algorithm 1, không xét tập biên): cận trên."""
    if len(h) < W or W <= 0:
        return None
    pre = [0.0]
    for x in h:
        pre.append(pre[-1] + x)
    best = 0.0
    for i in range(W, len(h) + 1):
        g = pre[i] / i
        if g > 0:
            best = max(best, ((pre[i] - pre[i - W]) / W) / g)
    return best


def first_diff(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--compare_run", default=None, help="Lần chạy khác có câu trùng (đo độ lặp lại)")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_json", default=None)
    args = p.parse_args(argv)
    rng = random.Random(args.seed)

    raw = {(r["method"], r["sample_id"]): r for r in read_jsonl(os.path.join(args.run_dir, "raw.jsonl"))}
    bud = {(r["method"], r["max_new_tokens"], r["sample_id"]): r
           for r in read_jsonl(os.path.join(args.run_dir, "budgets.jsonl"))}
    ids = sorted({s for _, s in raw})
    budgets = sorted({b for _, b, _ in bud})
    ok = lambda m, b, i: bool(bud[(m, b, i)]["is_correct"])  # noqa: E731
    acc = lambda m, b: 100 * sum(ok(m, b, i) for i in ids) / len(ids)  # noqa: E731
    out: Dict[str, Any] = {"run_dir": args.run_dir, "n": len(ids), "id_range": [ids[0], ids[-1]]}

    # 1. So sánh cặp
    pairs = [(OURS, 768, STD, 768), (OURS, 1024, STD, 1024), (OURS, TOP, STD, TOP)]
    pairs += [(OURS, b, ref, TOP) for b in (768, 1024) for ref in (STD,) + PAPER]
    out["comparisons"] = [
        {"target": f"{t}@{bt}", "reference": f"{r}@{br}",
         **paired([ok(t, bt, i) for i in ids], [ok(r, br, i) for i in ids], args.n_boot, rng)}
        for t, bt, r, br in pairs
    ]

    # 2. Ngân sách Standard cần để bằng accuracy của RPDI-EE
    out["std_budget_to_match"] = {
        str(b): next((bb for bb in budgets if acc(STD, bb) >= acc(OURS, b)), None) for b in (768, 1024)
    }
    out["recovery"] = {
        str(b): {r: (acc(OURS, b) - acc(STD, b)) / (acc(r, TOP) - acc(STD, b)) for r in (STD,) + PAPER}
        for b in (768, 1024)
    }

    # 3. Vì sao cấu hình bài gốc không phanh: RPDI lớn nhất trên chuỗi suy nghĩ của standard
    std_h = {i: raw[(STD, i)]["entropies"] for i in ids}
    tl = [think_len(raw[(STD, i)]) for i in ids]
    out["std_think_len"] = quantiles(tl)
    out["std_think_ge_512"] = sum(1 for x in tl if x >= 512)
    sweep = {}
    for W in (512, 256, 128):
        mx = [m for m in (max_rpdi(std_h[i], W) for i in ids) if m is not None]
        sweep[str(W)] = {"n_long_enough": len(mx), "max": max(mx) if mx else None,
                         "p90_of_max": pct(mx, .9) if mx else None,
                         **{f"gt_{th}": sum(1 for m in mx if m > th) for th in (1.3, 1.5, 2.0)}}
    out["max_rpdi_sweep"] = sweep
    out["paper_exit_count"] = {m: sum(1 for i in ids if raw[(m, i)]["forced_close"]) for m in PAPER}

    # 4. Nhiễu nền: chế độ paper (không phanh) so với standard
    noise = {}
    for m in PAPER:
        same = sum(1 for i in ids if raw[(m, i)]["output_ids"] == raw[(STD, i)]["output_ids"])
        fd = [d for d in (first_diff(raw[(m, i)]["output_ids"], raw[(STD, i)]["output_ids"]) for i in ids)
              if d is not None]
        noise[m] = {"identical": same, "first_diff_median": pct(fd, .5) if fd else None,
                    **{f"only_m_vs_only_std@{b}": [sum(1 for i in ids if ok(m, b, i) and not ok(STD, b, i)),
                                                   sum(1 for i in ids if ok(STD, b, i) and not ok(m, b, i))]
                       for b in (768, TOP)}}
    out["noise_paper_vs_std"] = noise

    # 5. Hành vi phanh của RPDI-EE
    ex = [i for i in ids if raw[(OURS, i)]["forced_close"]]
    out["ours_exit_rate"] = 100 * len(ex) / len(ids)
    out["ours_exit_step"] = quantiles([raw[(OURS, i)]["exit_step"] for i in ex])
    ans_len = [len(raw[(OURS, i)]["output_ids"]) - raw[(OURS, i)]["close_index"] - 1 for i in ex]
    out["ours_answer_len_after_exit"] = quantiles(ans_len)
    out["ours_exit_prefix_identical_to_std"] = sum(
        1 for i in ex if raw[(OURS, i)]["output_ids"][: raw[(OURS, i)]["close_index"]]
        == raw[(STD, i)]["output_ids"][: raw[(OURS, i)]["close_index"]])
    out["ours_loop_after_exit"] = [i for i in ex if raw[(OURS, i)]["finish_reason"] == "length"]

    # 6. Câu RPDI-EE sai nhưng standard@16k đúng, chia nhóm theo nguyên nhân
    def loss_breakdown(b: int) -> Dict[str, Any]:
        lost = [i for i in ids if not ok(OURS, b, i) and ok(STD, TOP, i)]
        g = {"truncated_after_exit": [], "finished_after_exit": [], "no_exit_truncated": [], "no_exit_finished": []}
        for i in lost:
            r = bud[(OURS, b, i)]
            key = ("truncated" if r["hit_budget"] else "finished")
            g[("" if r["early_exited"] else "no_exit_") + key + ("_after_exit" if r["early_exited"] else "")].append(i)
        return {"n_lost": len(lost), "n_gained": sum(1 for i in ids if ok(OURS, b, i) and not ok(STD, TOP, i)),
                **{k: {"n": len(v), "correct_at_16k": sum(ok(OURS, TOP, i) for i in v), "ids": v}
                   for k, v in g.items()}}
    out["loss_vs_std16k"] = {str(b): loss_breakdown(b) for b in (768, 1024)}

    # 7. Phanh nhầm: standard đúng nhưng RPDI-EE sai ở cùng ngân sách
    harm = {}
    for b in (768, 1024, TOP):
        hs = [i for i in ids if ok(STD, b, i) and not ok(OURS, b, i)]
        harm[str(b)] = {"n": len(hs), "exited": sum(bud[(OURS, b, i)]["early_exited"] for i in hs),
                        "truncated": sum(bud[(OURS, b, i)]["hit_budget"] for i in hs), "ids": hs}
    out["harm_same_budget"] = harm

    # 8. Lợi thế theo ngân sách
    out["by_budget"] = [{"budget": b, "acc_ours": acc(OURS, b), "acc_std": acc(STD, b),
                         "only_ours": sum(1 for i in ids if ok(OURS, b, i) and not ok(STD, b, i)),
                         "only_std": sum(1 for i in ids if ok(STD, b, i) and not ok(OURS, b, i))} for b in budgets]

    # 9. Ở 16k: nhóm standard lặp đến hết ngân sách so với nhóm bình thường
    loop = [i for i in ids if raw[(STD, i)]["close_index"] is None]
    normal = [i for i in ids if i not in set(loop) and raw[(OURS, i)]["finish_reason"] == "stop"]
    tot = lambda m, xs: sum(len(raw[(m, i)]["output_ids"]) for i in xs)  # noqa: E731
    out["top_budget_groups"] = {
        "std_loops": {"n": len(loop), "token_share_of_std_mean": tot(STD, loop) / len(ids),
                      "ours_exited": sum(raw[(OURS, i)]["forced_close"] for i in loop),
                      "ours_finished": sum(raw[(OURS, i)]["finish_reason"] == "stop" for i in loop),
                      "correct_ours": sum(ok(OURS, TOP, i) for i in loop),
                      "correct_std": sum(ok(STD, TOP, i) for i in loop)},
        "normal": {"n": len(normal), "acc_ours": 100 * sum(ok(OURS, TOP, i) for i in normal) / len(normal),
                   "acc_std": 100 * sum(ok(STD, TOP, i) for i in normal) / len(normal),
                   "only_ours": sum(1 for i in normal if ok(OURS, TOP, i) and not ok(STD, TOP, i)),
                   "only_std": sum(1 for i in normal if ok(STD, TOP, i) and not ok(OURS, TOP, i)),
                   "mean_tokens_std": tot(STD, normal) / len(normal), "mean_tokens_ours": tot(OURS, normal) / len(normal),
                   "mean_think_std": sum(think_len(raw[(STD, i)]) for i in normal) / len(normal),
                   "mean_think_ours": sum(think_len(raw[(OURS, i)]) for i in normal) / len(normal)},
    }

    # 10. Độ lặp lại: các câu trùng với một lần chạy khác
    if args.compare_run:
        raw2 = {(r["method"], r["sample_id"]): r for r in read_jsonl(os.path.join(args.compare_run, "raw.jsonl"))}
        bud2 = {(r["method"], r["max_new_tokens"], r["sample_id"]): r
                for r in read_jsonl(os.path.join(args.compare_run, "budgets.jsonl"))}
        rep = {}
        for m in (STD, OURS) + PAPER:
            common = [i for i in ids if (m, i) in raw2]
            if not common:
                continue
            rep[m] = {"n": len(common),
                      "identical_output": sum(raw[(m, i)]["output_ids"] == raw2[(m, i)]["output_ids"] for i in common),
                      **{f"acc@{b}": [100 * sum(ok(m, b, i) for i in common) / len(common),
                                      100 * sum(bool(bud2[(m, b, i)]["is_correct"]) for i in common) / len(common)]
                         for b in (768, 1024, TOP)}}
        out["repeatability"] = {"compare_run": args.compare_run, "methods": rep}

    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
