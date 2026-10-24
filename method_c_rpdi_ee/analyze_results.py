"""Phân tích thống kê kết quả RPDI-EE trên tập held-out (không cần GPU).

Gộp kết quả từ file 705 câu (results/method_c_results_705.json) và các file JSONL của
run_controls.py, rồi tính cho từng phương pháp:
    - Accuracy kèm khoảng tin cậy Wilson 95%
    - Token trung bình, tỷ lệ cạn ngân sách, tỷ lệ phanh (chỉ tính phanh trong pha suy nghĩ)
    - So sánh cặp với phương pháp tham chiếu: số ca giải cứu / phanh nhầm, McNemar chính xác,
      khoảng tin cậy 95% của chênh lệch accuracy
    - Phân tách theo việc baseline có cạn ngân sách hay không

Ví dụ:
    python method_c_rpdi_ee/analyze_results.py
    python method_c_rpdi_ee/analyze_results.py --controls results/controls_768.jsonl --reference rpdi_ee
"""

import argparse
import json
import math
import os
import sys
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phân tích thống kê kết quả RPDI-EE trên tập held-out.")
    parser.add_argument("--results_705", type=str, default=os.path.join(ROOT, "results", "method_c_results_705.json"))
    parser.add_argument("--controls", type=str, nargs="*", default=[], help="Các file JSONL từ run_controls.py")
    parser.add_argument("--min_id", type=int, default=100, help="Bỏ các câu có id < min_id (100 câu dùng để chọn W, lambda).")
    parser.add_argument("--max_id", type=int, default=10**9)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--reference", type=str, default="standard", help="Phương pháp tham chiếu cho so sánh cặp.")
    parser.add_argument("--output_json", type=str, default=os.path.join(ROOT, "results", "heldout_analysis.json"))
    parser.add_argument("--output_md", type=str, default=os.path.join(ROOT, "report", "phan_tich_heldout.md"))
    return parser.parse_args()


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def mcnemar_exact_p(b: int, c: int) -> float:
    """McNemar chính xác hai phía (kiểm định nhị thức trên các cặp bất đồng)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, float(Fraction(2 * tail, 2 ** n)))


def paired_diff_ci(b: int, c: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Chênh lệch accuracy (method - reference) cho dữ liệu cặp và khoảng tin cậy Wald."""
    d = (b - c) / n
    se = math.sqrt(max(0.0, (b + c) / n - d * d) / n)
    return d, d - z * se, d + z * se


def load_records(args: argparse.Namespace) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Trả về {sample_id: {method: record}}."""
    data: Dict[int, Dict[str, Dict[str, Any]]] = {}
    if args.results_705 and os.path.exists(args.results_705) and args.max_new_tokens == 768:
        with open(args.results_705, "r", encoding="utf-8") as f:
            for rec in json.load(f)["sample_records"]:
                sid = rec["sample_id"]
                data.setdefault(sid, {})["standard"] = rec["standard"]
                data[sid]["rpdi_ee"] = rec["rpdi_ee"]
    for path in args.controls:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["max_new_tokens"] != args.max_new_tokens:
                    continue
                data.setdefault(r["sample_id"], {})[r["method"]] = r
    return {sid: v for sid, v in data.items() if args.min_id <= sid < args.max_id}


def is_think_phase_brake(rec: Dict[str, Any], std: Optional[Dict[str, Any]], budget: int) -> bool:
    """Phanh có xảy ra trong pha suy nghĩ không.

    Kết quả cũ (trước khi sửa RPDILogitsProcessor) có thể "phanh" sau khi mô hình đã tự đóng
    </think>. Với giải mã Greedy, tiền tố trước điểm phanh trùng baseline, nên phanh có
    exit_step vượt độ dài suy nghĩ tự nhiên của baseline là phanh trong pha trả lời.
    """
    if not rec.get("early_exited"):
        return False
    if "brake_in_think_phase" in rec:
        # Bản ghi mới tự ghi nhận; độ dài suy nghĩ của 'standard' mô hình gốc không áp dụng cho mô hình RLVR
        return bool(rec["brake_in_think_phase"])
    if std is None or std["total_tokens"] >= budget:
        return True
    return rec.get("exit_step", -1) <= std["think_tokens"] + 2


def summarize(method: str, ids: List[int], data, budget: int) -> Dict[str, Any]:
    recs = [data[i][method] for i in ids]
    n = len(recs)
    k = sum(bool(r["is_correct"]) for r in recs)
    lo, hi = wilson_ci(k, n)
    brakes = sum(is_think_phase_brake(data[i][method], data[i].get("standard"), budget) for i in ids)
    raw_brakes = sum(bool(r.get("early_exited")) for r in recs)
    return {
        "n": n,
        "correct": k,
        "accuracy_pct": round(100 * k / n, 2),
        "acc_ci95_pct": [round(100 * lo, 2), round(100 * hi, 2)],
        "avg_total_tokens": round(sum(r["total_tokens"] for r in recs) / n, 1),
        "avg_think_tokens": round(sum(r["think_tokens"] for r in recs) / n, 1),
        "exhausted_pct": round(100 * sum(r["total_tokens"] >= budget for r in recs) / n, 2),
        "brake_rate_pct": round(100 * brakes / n, 2),
        "raw_early_exit_flag_pct": round(100 * raw_brakes / n, 2),
    }


def compare(method: str, ref: str, ids: List[int], data) -> Dict[str, Any]:
    b = sum(1 for i in ids if data[i][method]["is_correct"] and not data[i][ref]["is_correct"])
    c = sum(1 for i in ids if not data[i][method]["is_correct"] and data[i][ref]["is_correct"])
    d, lo, hi = paired_diff_ci(b, c, len(ids))
    return {
        "method_better": b,
        "reference_better": c,
        "delta_pct": round(100 * d, 2),
        "delta_ci95_pct": [round(100 * lo, 2), round(100 * hi, 2)],
        "mcnemar_exact_p": mcnemar_exact_p(b, c),
    }


def fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def main() -> None:
    args = parse_args()
    budget = args.max_new_tokens
    data = load_records(args)
    methods = sorted({m for v in data.values() for m in v})
    # Chỉ so sánh trên các câu mà mọi phương pháp đều đã có kết quả
    ids = sorted(i for i, v in data.items() if all(m in v for m in methods))
    if not ids:
        raise SystemExit("Không có câu nào có đủ kết quả cho mọi phương pháp.")
    print(f"[*] {len(ids)} câu chung (id {ids[0]}–{ids[-1]}), phương pháp: {', '.join(methods)}")

    out: Dict[str, Any] = {
        "setup": {"min_id": args.min_id, "max_new_tokens": budget, "n_common": len(ids),
                  "id_range": [ids[0], ids[-1]], "reference": args.reference},
        "methods": {m: summarize(m, ids, data, budget) for m in methods},
        "paired_vs_reference": {},
        "by_baseline_exhaustion": {},
    }
    if args.reference in methods:
        for m in methods:
            if m != args.reference:
                out["paired_vs_reference"][m] = compare(m, args.reference, ids, data)

    if "standard" in methods:
        groups = {
            "baseline_finished": [i for i in ids if data[i]["standard"]["total_tokens"] < budget],
            "baseline_exhausted": [i for i in ids if data[i]["standard"]["total_tokens"] >= budget],
        }
        for g, gids in groups.items():
            if gids:
                out["by_baseline_exhaustion"][g] = {
                    "n": len(gids),
                    **{m: round(100 * sum(data[i][m]["is_correct"] for i in gids) / len(gids), 2) for m in methods},
                }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = [
        "# Phân tích thống kê RPDI-EE trên tập held-out",
        "",
        f"- Tập đánh giá: {len(ids)} câu GSM8K test (id {ids[0]}–{ids[-1]}), loại các câu id < {args.min_id} "
        "đã dùng để chọn siêu tham số.",
        f"- Ngân sách: `max_new_tokens = {budget}`, giải mã Greedy.",
        "- Tỷ lệ phanh chỉ tính các lần phanh trong pha suy nghĩ (xem ghi chú cuối).",
        "",
        "## 1. Kết quả từng phương pháp",
        "",
        "| Phương pháp | Đúng | Accuracy | KTC 95% | Token TB | Think TB | Cạn ngân sách | Tỷ lệ phanh |",
        "| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
    ]
    for m, s in out["methods"].items():
        lines.append(
            f"| `{m}` | {s['correct']}/{s['n']} | {s['accuracy_pct']:.2f}% | "
            f"[{s['acc_ci95_pct'][0]:.1f}, {s['acc_ci95_pct'][1]:.1f}] | {s['avg_total_tokens']} | "
            f"{s['avg_think_tokens']} | {s['exhausted_pct']:.2f}% | {s['brake_rate_pct']:.2f}% |"
        )
    if out["paired_vs_reference"]:
        lines += [
            "",
            f"## 2. So sánh cặp với `{args.reference}`",
            "",
            "| Phương pháp | Chỉ phương pháp đúng | Chỉ tham chiếu đúng | Δ Accuracy | KTC 95% của Δ | McNemar p |",
            "| :--- | ---: | ---: | ---: | :---: | ---: |",
        ]
        for m, c in out["paired_vs_reference"].items():
            lines.append(
                f"| `{m}` | {c['method_better']} | {c['reference_better']} | {c['delta_pct']:+.2f} | "
                f"[{c['delta_ci95_pct'][0]:+.1f}, {c['delta_ci95_pct'][1]:+.1f}] | {fmt_p(c['mcnemar_exact_p'])} |"
            )
    if out["by_baseline_exhaustion"]:
        lines += [
            "",
            "## 3. Phân tách theo trạng thái cạn ngân sách của baseline",
            "",
            "| Nhóm | Số câu | " + " | ".join(f"`{m}`" for m in methods) + " |",
            "| :--- | ---: | " + " | ".join("---:" for _ in methods) + " |",
        ]
        names = {"baseline_finished": "Baseline tự đóng `</think>`", "baseline_exhausted": f"Baseline cạn {budget} token"}
        for g, v in out["by_baseline_exhaustion"].items():
            lines.append(f"| {names[g]} | {v['n']} | " + " | ".join(f"{v[m]:.2f}%" for m in methods) + " |")
    if "rpdi_ee" in out["methods"]:
        s = out["methods"]["rpdi_ee"]
        lines += [
            "",
            "## Ghi chú",
            "",
            f"- Kết quả `rpdi_ee` lấy từ lần chạy 705 câu, trước khi sửa lỗi `RPDILogitsProcessor` tiếp tục giám sát "
            f"sau khi mô hình tự đóng `</think>`. Cờ `early_exited` gốc báo {s['raw_early_exit_flag_pct']:.2f}%, "
            f"nhưng chỉ {s['brake_rate_pct']:.2f}% là phanh thật trong pha suy nghĩ; phần còn lại là chèn thêm "
            "`</think>` vào pha trả lời.",
        ]
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[OK] Đã lưu {args.output_json} và {args.output_md}")


if __name__ == "__main__":
    main()
