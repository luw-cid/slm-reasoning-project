"""Callback ghi nhận động học entropy trong quá trình huấn luyện GRPO (phục vụ Gap 1).

Hai callback độc lập:
1. JSONLLogCallback: ghi toàn bộ metric mà Trainer/TRL in ra (reward, độ dài sinh, tỷ lệ bị cắt,
   entropy của policy nếu phiên bản TRL có log) vào file JSONL để vẽ đồ thị sau huấn luyện.
2. RPDIProbeCallback: định kỳ sinh Greedy trên một tập câu probe cố định với ngân sách 768 token
   (đúng thiết lập đánh giá), ghi entropy từng token trong pha <think> và mô phỏng offline RPDI-EE.
   Trả lời câu hỏi: RLVR làm thay đổi tín hiệu entropy mà RPDI-EE dựa vào như thế nào?
   Trace được lưu đúng định dạng results/baseline_traces.json để tái sử dụng pipeline ablation.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList, TrainerCallback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.metrics import extract_answer_from_response, is_answer_correct
from common.model_loader import format_reasoning_prompt
from method_c_rpdi_ee.ablation.step2_ablate_grid import simulate_rpdi_exit
from method_c_rpdi_ee.logits_processor import build_boundary_ids


DEFAULT_PROBE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "common", "data", "gsm8k_val_200.json"
)


class JSONLLogCallback(TrainerCallback):
    """Ghi mỗi lần Trainer log metric thành một dòng JSON trong output_dir/train_log.jsonl."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._warned_no_entropy = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        if "entropy" not in logs and "loss" in logs and not self._warned_no_entropy:
            print(
                "[EntropyLog] Phiên bản TRL hiện tại không log 'entropy' của policy. "
                "Probe RPDI vẫn hoạt động; nâng cấp TRL nếu cần entropy trên rollout huấn luyện."
            )
            self._warned_no_entropy = True
        record = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class BatchEntropyRecorder(LogitsProcessor):
    """Ghi entropy phân bố token kế tiếp cho mọi phần tử trong batch (giữ trên GPU, không đổi scores)."""

    def __init__(self):
        super().__init__()
        self.steps: List[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores.float()
        log_probs = F.log_softmax(logits, dim=-1)
        self.steps.append(-(log_probs.exp() * log_probs).sum(dim=-1))
        return scores

    def as_lists(self) -> List[List[float]]:
        """Trả về entropy dạng [batch][step]."""
        if not self.steps:
            return []
        return torch.stack(self.steps, dim=1).cpu().tolist()


def rpdi_at_boundaries(
    entropies: List[float],
    generated_token_ids: List[int],
    boundary_ids: set,
    think_len: int,
    W: int,
) -> List[float]:
    """Chuỗi giá trị RPDI = LTF/GTF tại các bước kiểm tra biên câu (cùng logic với simulate_rpdi_exit)."""
    values = []
    S_global = 0.0
    S_local = 0.0
    for i in range(1, think_len + 1):
        H_ti = entropies[i - 1]
        S_global += H_ti
        S_local += H_ti
        if i > W:
            S_local -= entropies[i - W - 1]
        if i > 1 and i >= W and generated_token_ids[i - 2] in boundary_ids:
            gtf = S_global / i
            values.append((S_local / W) / gtf if gtf > 0 else 0.0)
    return values


class RPDIProbeCallback(TrainerCallback):
    """Định kỳ đo tín hiệu entropy/RPDI của policy hiện tại trên tập probe cố định (Greedy, 768 token).

    Ghi:
        - output_dir/rpdi_probe.jsonl: một dòng tổng hợp cho mỗi lần probe.
        - output_dir/probe_traces/step_XXXXX.json: trace từng câu (định dạng baseline_traces.json).
    """

    def __init__(
        self,
        tokenizer,
        output_dir: str,
        probe_every: int = 25,
        probe_size: int = 16,
        probe_file: str = DEFAULT_PROBE_FILE,
        max_new_tokens: int = 768,
        W: int = 16,
        lambda_th: float = 1.5,
        batch_size: int = 8,
    ):
        self.tokenizer = tokenizer
        self.probe_every = probe_every
        self.max_new_tokens = max_new_tokens
        self.W = W
        self.lambda_th = lambda_th
        self.batch_size = batch_size

        with open(probe_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)[:probe_size]

        self.summary_path = os.path.join(output_dir, "rpdi_probe.jsonl")
        self.trace_dir = os.path.join(output_dir, "probe_traces")
        os.makedirs(self.trace_dir, exist_ok=True)

        self.boundary_ids = build_boundary_ids(tokenizer)
        think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.close_think_id: Optional[int] = think_close_ids[0] if len(think_close_ids) == 1 else None
        self._last_probed_step = -1

    # ------------------------------------------------------------------ hooks
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Probe tại bước 0: LoRA khởi tạo B=0 nên đây chính là mô hình gốc (mốc so sánh)
        self._run_probe(model, state)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.probe_every > 0 and state.global_step % self.probe_every == 0:
            self._run_probe(model, state)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._run_probe(model, state)

    # ------------------------------------------------------------------ probe
    @torch.no_grad()
    def _run_probe(self, model, state) -> None:
        if model is None or not state.is_world_process_zero:
            return
        if state.global_step == self._last_probed_step:
            return
        self._last_probed_step = state.global_step

        was_training = model.training
        model.eval()
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        start = time.time()

        traces: List[Dict[str, Any]] = []
        try:
            for b in range(0, len(self.samples), self.batch_size):
                traces.extend(self._probe_batch(model, self.samples[b : b + self.batch_size]))
        finally:
            self.tokenizer.padding_side = old_padding_side
            if was_training:
                model.train()

        summary = self._summarize(traces, state.global_step, time.time() - start)
        with open(self.summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        trace_path = os.path.join(self.trace_dir, f"step_{state.global_step:05d}.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "global_step": state.global_step,
                        "max_new_tokens": self.max_new_tokens,
                        "boundary_token_ids": sorted(self.boundary_ids),
                        "decoding": "greedy, batched left-padding",
                    },
                    "traces": traces,
                },
                f,
                ensure_ascii=False,
            )

        print(
            f"[RPDIProbe] step={summary['step']} | acc={summary['accuracy_pct']:.1f}% | "
            f"think TB={summary['avg_think_tokens']:.1f} | tự đóng </think>={summary['closed_think_pct']:.1f}% | "
            f"H think={summary['mean_think_entropy']:.3f} | phanh RPDI={summary['rpdi_brake_pct']:.1f}% "
            f"({summary['probe_seconds']:.0f}s)"
        )

    def _probe_batch(self, model, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompts = [format_reasoning_prompt(s["question"]) for s in batch]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        recorder = BatchEntropyRecorder()
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            logits_processor=LogitsProcessorList([recorder]),
            do_sample=False,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        entropies_batch = recorder.as_lists()

        traces = []
        for row, sample in enumerate(batch):
            gen_ids = outputs[row, prompt_len:].tolist()
            # Cắt tại EOS đầu tiên (giữ EOS) — phần sau là padding của batch
            if self.tokenizer.eos_token_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(self.tokenizer.eos_token_id) + 1]
            entropies = entropies_batch[row][: len(gen_ids)]

            natural_end = None
            if self.close_think_id is not None and self.close_think_id in gen_ids:
                natural_end = gen_ids.index(self.close_think_id)
            think_len = natural_end if natural_end is not None else len(gen_ids)

            response_text = "<think>" + self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            pred = extract_answer_from_response(response_text)
            rpdi_values = rpdi_at_boundaries(entropies, gen_ids, self.boundary_ids, think_len, self.W)
            brake_think_tokens = simulate_rpdi_exit(
                entropies, gen_ids, self.boundary_ids, natural_end, W=self.W, lambda_th=self.lambda_th
            )

            traces.append(
                {
                    "sample_id": sample.get("id"),
                    "ground_truth": sample["ground_truth"],
                    "predicted_answer": pred,
                    "is_correct": is_answer_correct(pred, sample["ground_truth"]),
                    "total_generated": len(gen_ids),
                    "think_tokens": think_len,
                    "natural_think_end_idx": natural_end,
                    "mean_think_entropy": sum(entropies[:think_len]) / think_len if think_len else 0.0,
                    "max_rpdi": max(rpdi_values) if rpdi_values else None,
                    "rpdi_brake_think_tokens": brake_think_tokens,
                    "entropies": entropies,
                    "generated_token_ids": gen_ids,
                    "response_text": response_text,
                }
            )
        return traces

    def _summarize(self, traces: List[Dict[str, Any]], step: int, seconds: float) -> Dict[str, Any]:
        n = len(traces)
        braked = [t["rpdi_brake_think_tokens"] for t in traces if t["rpdi_brake_think_tokens"] is not None]
        max_rpdis = [t["max_rpdi"] for t in traces if t["max_rpdi"] is not None]
        return {
            "step": step,
            "n": n,
            "W": self.W,
            "lambda_th": self.lambda_th,
            "accuracy_pct": 100.0 * sum(t["is_correct"] for t in traces) / n,
            "avg_total_tokens": sum(t["total_generated"] for t in traces) / n,
            "avg_think_tokens": sum(t["think_tokens"] for t in traces) / n,
            "closed_think_pct": 100.0 * sum(t["natural_think_end_idx"] is not None for t in traces) / n,
            "mean_think_entropy": sum(t["mean_think_entropy"] for t in traces) / n,
            "mean_max_rpdi": sum(max_rpdis) / len(max_rpdis) if max_rpdis else None,
            "rpdi_brake_pct": 100.0 * len(braked) / n,
            "avg_rpdi_brake_position": sum(braked) / len(braked) if braked else None,
            "probe_seconds": round(seconds, 1),
        }
