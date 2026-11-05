"""Vẽ accuracy theo ngân sách và theo số token trung bình từ analysis.json (analyze_budgets.py).

    python method_c_rpdi_ee/vllm_engine/plot_budgets.py --analysis results/gsm8k_500/analysis.json \
        --output report/gsm8k_500_accuracy_vs_budget.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Bảng màu phân loại đã kiểm tra mù màu (dataviz validate_palette.js, nền sáng)
SERIES = [
    ("rpdi_legacy_W16_l1.5", "RPDI-EE (W=16, λ=1.5)", "#2a78d6", "-", "o"),
    ("standard", "Standard", "#eb6834", "-", "s"),
    ("rpdi_W512_l2.0_wait", "Bài gốc W=512, λ=2.0 (wait)", "#1baf7a", "--", "^"),
    ("rpdi_W512_l2.0_newline", "Bài gốc W=512, λ=2.0 (newline)", "#eda100", ":", "D"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    table = json.load(open(args.analysis, encoding="utf-8"))["table"]
    n = json.load(open(args.analysis, encoding="utf-8"))["n_common"]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    for ax, xkey, xlabel in ((axes[0], "budget", "Ngân sách token (thang log)"),
                             (axes[1], "avg_total_tokens", "Số token trung bình mỗi câu (thang log)")):
        for key, label, color, ls, mk in SERIES:
            rows = sorted(table[key].items(), key=lambda kv: int(kv[0]))
            xs = [int(b) if xkey == "budget" else s["avg_total_tokens"] for b, s in rows]
            ys = [s["acc"] for _, s in rows]
            ax.plot(xs, ys, ls, color=color, lw=2, marker=mk, ms=5, label=label,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3 if key.startswith("rpdi_legacy") else 2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(40, 86)

    budgets = sorted(int(b) for b in table["standard"])
    axes[0].set_xticks(budgets)
    axes[0].set_xticklabels([str(b) for b in budgets], rotation=45)
    for b in (768, 1024):
        axes[0].axvline(b, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
    ref = table["standard"]["16384"]["acc"]
    for ax in axes:
        ax.axhline(ref, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
    ours = table["rpdi_legacy_W16_l1.5"]
    for b, off in (("768", (8, -5)), ("1024", (4, 9))):
        axes[0].annotate(f"{ours[b]['acc']:.1f}% @{b}", (int(b), ours[b]["acc"]), xytext=off,
                         textcoords="offset points", fontsize=9, color=INK)
    axes[0].set_title("(a) Accuracy theo ngân sách", loc="left", fontsize=11)
    axes[1].set_title("(b) Accuracy theo số token thực dùng", loc="left", fontsize=11)
    axes[1].set_xticks([512, 1024, 2048])
    axes[1].set_xticklabels(["512", "1024", "2048"])
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, lw=0.8, ls=(0, (2, 3))))
    labels.append(f"Standard@16384 = {ref:.1f}%")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"RPDI-EE trên GSM8K, {n} câu, DeepSeek-R1-Distill-Qwen-1.5B (vLLM, fp16, greedy)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(args.output, facecolor="white")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
