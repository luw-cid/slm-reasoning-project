"""Chạy Standard và RPDI-EE trên vLLM (V0, logits processor theo từng request) ở ngân sách lớn,
rồi suy ra kết quả ở mọi ngân sách nhỏ hơn bằng cách cắt chuỗi token (xem budgets.py).

Phương pháp (--methods, phân tách bằng dấu phẩy):
    standard                  Sinh tự nhiên; vẫn ghi entropy pha suy nghĩ để phân tích offline.
    rpdi:W:lambda[:B]         RPDI-EE đúng Algorithm 1 của bài gốc; B = wait (mặc định) | newline | punct.
    rpdi_legacy:W:lambda      Bản sao chính xác RPDILogitsProcessor (HF) đã dùng cho kết quả 768 token.

File ghi vào --output_dir (mỗi lượt được ghi ngay khi sinh xong, kèm một dòng log):
    raw.jsonl       mỗi dòng = 1 câu x 1 phương pháp: output_ids, vị trí </think>, entropy pha suy nghĩ
    budgets.jsonl   mỗi dòng = 1 câu x 1 phương pháp x 1 ngân sách, cùng khóa với run_controls.py
    run_info.json   phiên bản thư viện, GPU, tham số, tiến độ
Chạy lại cùng lệnh sẽ bỏ qua các cặp (câu, phương pháp) đã có trong raw.jsonl.

Ví dụ (Colab T4):
    python method_c_rpdi_ee/vllm_engine/run_vllm.py --start 100 --end 300 --output_dir /content/drive/MyDrive/slm_rlvr/vllm_pilot
    python method_c_rpdi_ee/vllm_engine/run_vllm.py --expand_only --output_dir ... --budgets 768,16384
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.eval_dataset import load_gsm8k_test  # noqa: E402
from common.model_loader import DEFAULT_MODEL_ID, format_reasoning_prompt  # noqa: E402
from method_c_rpdi_ee.vllm_engine.budgets import expand_to_budgets  # noqa: E402
from method_c_rpdi_ee.vllm_engine.rpdi_core import MethodSpec, monitor_for, parse_method  # noqa: E402

DEFAULT_DATA_FILE = os.path.join(ROOT, "common", "data", "gsm8k_test_full.json")
DEFAULT_METHODS = "standard,rpdi_legacy:16:1.5,rpdi:512:2.0,rpdi:512:2.0:newline"
DEFAULT_BUDGETS = "512,768,1024,1536,2048,3072,4096,8192,16384"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RPDI-EE trên vLLM: sinh ở ngân sách lớn, suy ra mọi ngân sách nhỏ hơn.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--data_file", type=str, default=DEFAULT_DATA_FILE)
    p.add_argument("--start", type=int, default=100, help="Chỉ số câu bắt đầu (mặc định 100: bỏ 100 câu dùng để chọn W, lambda).")
    p.add_argument("--end", type=int, default=300, help="Chỉ số câu kết thúc (không bao gồm).")
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    p.add_argument("--max_tokens", type=int, default=16384, help="Ngân sách sinh lớn nhất (tổng token suy nghĩ + trả lời).")
    p.add_argument("--budgets", type=str, default=DEFAULT_BUDGETS, help="Các ngân sách suy ra bằng cách cắt chuỗi.")
    p.add_argument("--chunk_size", type=int, default=None,
                   help="Không còn dùng (giữ để lệnh cũ không lỗi): mỗi lượt được ghi file ngay khi sinh xong.")
    p.add_argument("--dtype", type=str, default="float16", help="T4 không hỗ trợ bfloat16.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_num_seqs", type=int, default=128)
    p.add_argument("--enforce_eager", action="store_true", help="Tắt CUDA graph (dùng khi bước capture lỗi).")
    p.add_argument("--expand_only", action="store_true", help="Chỉ tạo lại budgets.jsonl từ raw.jsonl (không cần GPU).")
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args(argv)
    args.method_specs = [parse_method(m) for m in args.methods.split(",") if m.strip()]
    args.budget_list = sorted({int(b) for b in args.budgets.split(",") if b.strip()})
    return args


class Heartbeat:
    """In định kỳ số lượt còn đang sinh và độ dài lượt dài nhất.

    Thanh tiến độ của vLLM chỉ nhích khi một lượt sinh XONG; ở cuối mỗi chunk vài lượt dài (tới 16k
    token) có thể chạy hơn 10 phút mà thanh tiến độ đứng yên. Dòng này cho thấy tiến trình vẫn chạy.
    """

    def __init__(self, interval: float = 60.0, active_window: float = 5.0, clock=time.time, out=print):
        self.interval = interval
        self.active_window = active_window
        self.clock = clock
        self.out = out
        self.last_print = clock()
        self.lengths: Dict[int, int] = {}
        self.seen: Dict[int, float] = {}

    def tick(self, key: int, n_tokens: int) -> None:
        now = self.clock()
        self.lengths[key] = n_tokens
        self.seen[key] = now
        if now - self.last_print >= self.interval:
            active = [k for k, t in self.seen.items() if now - t <= self.active_window]
            longest = max((self.lengths[k] for k in active), default=0)
            self.out(f"[đang sinh] {len(active)} lượt chưa xong, dài nhất {longest} token", flush=True)
            self.last_print = now


class VLLMProcessor:
    """Logits processor theo request cho vLLM V0, chữ ký (prompt_ids, output_ids, logits) -> logits."""

    def __init__(self, monitor, heartbeat: Optional[Heartbeat] = None):
        self.monitor = monitor
        self.heartbeat = heartbeat

    def __call__(self, prompt_ids, output_ids, logits):
        import torch

        if self.heartbeat is not None:
            self.heartbeat.tick(id(self), len(output_ids))

        def stats() -> Tuple[float, int]:
            lp = torch.log_softmax(logits.float(), dim=-1)
            p = lp.exp()
            ent = -torch.where(p > 0, p * lp, torch.zeros_like(p)).sum()
            h, am = torch.stack([ent, lp.argmax().to(ent.dtype)]).tolist()
            return float(h), int(am)

        tok = self.monitor.step(prompt_ids, output_ids, stats)
        if tok is None:
            return logits
        forced = torch.full_like(logits, float("-inf"))
        forced[tok] = 0.0
        return forced


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def gpu_memory_problem(free_bytes: int, total_bytes: int, utilization: float) -> Optional[str]:
    """Thông báo lỗi nếu GPU không còn đủ bộ nhớ trống cho vLLM (thường do một lần chạy khác còn chạy nền)."""
    need = utilization * total_bytes
    if free_bytes >= need:
        return None
    gib = 1024 ** 3
    return (f"GPU chỉ còn trống {free_bytes / gib:.1f}/{total_bytes / gib:.1f} GiB, nhưng vLLM cần "
            f"{need / gib:.1f} GiB (--gpu_memory_utilization {utilization}). Có thể một lần chạy run_vllm.py "
            "khác vẫn đang chạy nền trên GPU này: kiểm tra bằng `!ps aux | grep run_vllm` và `!nvidia-smi`, "
            "đợi nó chạy xong hoặc dừng nó (`!pkill -f run_vllm.py`) rồi chạy lại.")


def think_close_id_of(tokenizer) -> int:
    ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"'</think>' phải là một token, nhận được {ids}")
    return ids[0]


def build_raw(sample: Dict[str, Any], method: str, max_tokens: int, prompt_ids: List[int], token_ids: List[int],
              finish_reason: Optional[str], stop_reason: Any, monitor, close_id: int, eos_id: Optional[int]) -> Dict[str, Any]:
    """Bản ghi raw.jsonl cho một lượt (câu x phương pháp) đã sinh xong."""
    ids = list(token_ids)
    # HF tính cả token EOS vào số token sinh; bảo đảm vLLM cũng vậy
    if finish_reason == "stop" and stop_reason is None and eos_id is not None and (not ids or ids[-1] != eos_id):
        ids.append(eos_id)
    return {
        "sample_id": sample["id"],
        "method": method,
        "max_tokens": max_tokens,
        "ground_truth": sample["ground_truth"],
        "prompt_ids": list(prompt_ids),
        "output_ids": ids,
        "close_index": ids.index(close_id) if close_id in ids else None,
        "forced_close": monitor.early_exited,
        "exit_step": monitor.exit_step,
        "rpdi_at_exit": monitor.rpdi_at_exit,
        "finish_reason": finish_reason,
        "entropies": [round(h, 4) for h in monitor.entropies],
    }


def format_done_line(k: int, total: int, raw: Dict[str, Any], recs: List[Dict[str, Any]], elapsed_s: float) -> str:
    """Một dòng log khi một lượt sinh xong, vd:
    [  12/800] id=102 rpdi_legacy_W16_l1.5    |    604 token, phanh ở token 131 | @768: ĐÚNG | @16384: ĐÚNG | 3.2 phút
    """
    n = len(raw["output_ids"])
    if raw["forced_close"]:
        status = f"phanh ở token {raw['exit_step']}"
    elif raw["close_index"] is not None:
        status = f"tự đóng </think> ở token {raw['close_index']}"
    else:
        status = "không đóng </think>"
    if raw.get("finish_reason") == "length":
        status += ", chạm trần"
    mark = lambda r: "ĐÚNG" if r["is_correct"] else "sai"  # noqa: E731
    parts = [f"[{k:>4}/{total}] id={raw['sample_id']} {raw['method']:<24}", f"{n:>6} token, {status}"]
    parts += [f"@{r['max_new_tokens']}: {mark(r)}" for r in recs if r["max_new_tokens"] in (768, raw["max_tokens"])]
    parts.append(f"{elapsed_s / 60:.1f} phút")
    return " | ".join(parts)


def expand_only(args: argparse.Namespace) -> None:
    raw_path = os.path.join(args.output_dir, "raw.jsonl")
    raws = read_jsonl(raw_path)
    if not raws:
        raise SystemExit(f"Không có dữ liệu trong {raw_path}")
    tokenizer = load_tokenizer(args.model_id)
    close_id = think_close_id_of(tokenizer)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)  # noqa: E731
    out_path = os.path.join(args.output_dir, "budgets.jsonl")
    n = 0
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        for raw in raws:
            for rec in expand_to_budgets(raw, decode, close_id, args.budget_list):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    os.replace(out_path + ".tmp", out_path)
    print(f"[*] Đã ghi {n} bản ghi ngân sách vào {out_path}")


def generate(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "raw.jsonl")
    budgets_path = os.path.join(args.output_dir, "budgets.jsonl")
    info_path = os.path.join(args.output_dir, "run_info.json")

    dataset = load_gsm8k_test(filepath=args.data_file)[args.start:args.end]
    done: Set[Tuple[int, str, int]] = {(r["sample_id"], r["method"], r["max_tokens"]) for r in read_jsonl(raw_path)}
    todo: List[Tuple[Dict[str, Any], MethodSpec]] = [
        (s, spec) for s in dataset for spec in args.method_specs if (s["id"], spec.key, args.max_tokens) not in done
    ]
    print(f"[*] {len(dataset)} câu x {len(args.method_specs)} phương pháp | còn {len(todo)} lượt (đã có {len(done)}).")
    if not todo:
        return

    os.environ.setdefault("VLLM_USE_V1", "0")  # logits processor theo request chỉ có ở engine V0
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    if torch.cuda.is_available():
        problem = gpu_memory_problem(*torch.cuda.mem_get_info(), args.gpu_memory_utilization)
        if problem:
            raise SystemExit(problem)

    tokenizer = load_tokenizer(args.model_id)
    close_id = think_close_id_of(tokenizer)
    eos_id = tokenizer.eos_token_id
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)  # noqa: E731
    prompt_ids = {s["id"]: tokenizer(format_reasoning_prompt(s["question"]))["input_ids"] for s in dataset}
    max_model_len = args.max_tokens + max(len(v) for v in prompt_ids.values()) + 16

    info: Dict[str, Any] = {
        "args": {k: v for k, v in vars(args).items() if k != "method_specs"},
        "methods": [spec.__dict__ for spec in args.method_specs],
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_model_len": max_model_len,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "progress": {},
    }
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info["previous_runs"] = [json.load(f)]
    write_json(info_path, info)
    print(f"[*] vllm {vllm.__version__} | torch {torch.__version__} | transformers {transformers.__version__} | GPU {info['gpu']}")

    llm = LLM(
        model=args.model_id,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        max_seq_len_to_capture=max_model_len,  # mặc định 8192: chuỗi dài hơn sẽ rơi về chế độ eager (chậm)
        # Logits processor cần thấy đủ token đã sinh ở mỗi bước; tắt xử lý output bất đồng bộ cho chắc chắn
        disable_async_output_proc=True,
        seed=0,
    )

    # Đưa toàn bộ lượt vào engine (theo thứ tự câu, mỗi câu đủ các phương pháp) rồi tự chạy từng bước:
    # lượt nào xong được ghi file và in log NGAY, không phải chờ cả đợt như LLM.generate.
    engine = llm.llm_engine
    boundary_cache: Dict[str, frozenset] = {}
    heartbeat = Heartbeat(interval=60.0)
    pending: Dict[str, Tuple[Dict[str, Any], MethodSpec, Any]] = {}
    for n, (s, spec) in enumerate(todo):
        mon = monitor_for(spec, tokenizer, boundary_cache)
        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                logits_processors=[VLLMProcessor(mon, heartbeat)], skip_special_tokens=False)
        engine.add_request(str(n), {"prompt_token_ids": prompt_ids[s["id"]]}, params)
        pending[str(n)] = (s, spec, mon)

    total_done = len(done) + len(todo)
    t_start = time.time()
    n_done, n_tokens = 0, 0
    print(f"[*] Bắt đầu sinh {len(todo)} lượt (tối đa {args.max_num_seqs} lượt chạy song song).", flush=True)
    with open(raw_path, "a", encoding="utf-8") as fraw, open(budgets_path, "a", encoding="utf-8") as fbud:
        while engine.has_unfinished_requests():
            for out in engine.step():
                if not out.finished:
                    continue
                s, spec, mon = pending.pop(out.request_id)
                comp = out.outputs[0]
                raw = build_raw(s, spec.key, args.max_tokens, prompt_ids[s["id"]], list(comp.token_ids),
                                comp.finish_reason, comp.stop_reason, mon, close_id, eos_id)
                recs = expand_to_budgets(raw, decode, close_id, args.budget_list)
                fraw.write(json.dumps(raw, ensure_ascii=False) + "\n")
                for rec in recs:
                    fbud.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fraw.flush()
                fbud.flush()

                n_done += 1
                n_tokens += len(raw["output_ids"])
                elapsed = time.time() - t_start
                print(format_done_line(len(done) + n_done, total_done, raw, recs, elapsed), flush=True)
                if n_done % 25 == 0 or not pending:
                    info["progress"] = {"done": len(done) + n_done, "total": total_done,
                                        "elapsed_seconds": round(elapsed, 1), "generated_tokens": n_tokens}
                    write_json(info_path, info)

    print(f"[*] Hoàn tất. Kết quả: {raw_path} | {budgets_path}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.expand_only:
        expand_only(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
