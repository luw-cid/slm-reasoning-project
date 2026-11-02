from method_c_rpdi_ee.vllm_engine.run_vllm import gpu_memory_problem, parse_args

GIB = 1024 ** 3


def test_gpu_memory_ok_on_empty_t4():
    assert gpu_memory_problem(int(14.4 * GIB), int(14.56 * GIB), 0.90) is None


def test_gpu_memory_busy_reports_other_run():
    # Lỗi thật trên Colab: một lần chạy khác chiếm 12.6 GiB, chỉ còn trống ~0.03 GiB
    msg = gpu_memory_problem(int(0.03 * GIB), int(14.56 * GIB), 0.90)
    assert msg is not None and "run_vllm.py" in msg and "13.1" in msg


def test_parse_args_defaults():
    args = parse_args(["--output_dir", "x"])
    assert [s.key for s in args.method_specs] == [
        "standard", "rpdi_legacy_W16_l1.5", "rpdi_W512_l2.0_wait", "rpdi_W512_l2.0_newline"]
    assert args.budget_list[0] == 512 and args.budget_list[-1] == 16384 and 768 in args.budget_list
    assert (args.start, args.end, args.max_tokens) == (100, 300, 16384)


def test_heartbeat_reports_active_runs_and_longest():
    from method_c_rpdi_ee.vllm_engine.run_vllm import Heartbeat

    now = {"t": 0.0}
    lines = []
    hb = Heartbeat(interval=60.0, active_window=5.0, clock=lambda: now["t"],
                   out=lambda msg, flush=False: lines.append(msg))
    hb.tick(1, 100)
    hb.tick(2, 50)
    assert lines == []                 # chưa đủ 60 giây
    now["t"] = 30.0
    hb.tick(2, 4000)                   # lượt 1 đã xong (không còn được gọi)
    now["t"] = 61.0
    hb.tick(3, 9000)
    assert lines == ["[đang sinh] 1 lượt chưa xong, dài nhất 9000 token"]
    now["t"] = 62.0
    hb.tick(3, 9001)
    assert len(lines) == 1             # chưa tới kỳ in tiếp theo


class _Mon:
    def __init__(self, exited=False, exit_step=-1):
        self.early_exited, self.exit_step, self.rpdi_at_exit = exited, exit_step, 2.5 if exited else 0.0
        self.entropies = [0.123456, 1.0]


def test_build_raw_appends_eos_and_finds_close():
    from method_c_rpdi_ee.vllm_engine.run_vllm import build_raw

    s = {"id": 102, "ground_truth": "7"}
    raw = build_raw(s, "standard", 16384, [1, 2], [5, 9, 6], "stop", None, _Mon(), close_id=9, eos_id=0)
    assert raw["output_ids"] == [5, 9, 6, 0] and raw["close_index"] == 1
    assert raw["entropies"] == [0.1235, 1.0] and raw["forced_close"] is False
    raw2 = build_raw(s, "standard", 16384, [1, 2], [5, 6], "length", None, _Mon(), close_id=9, eos_id=0)
    assert raw2["output_ids"] == [5, 6] and raw2["close_index"] is None


def test_format_done_line():
    from method_c_rpdi_ee.vllm_engine.run_vllm import format_done_line

    raw = {"sample_id": 102, "method": "rpdi_legacy_W16_l1.5", "max_tokens": 16384, "output_ids": [0] * 604,
           "forced_close": True, "exit_step": 131, "close_index": 131, "finish_reason": "stop"}
    recs = [{"max_new_tokens": 512, "is_correct": False}, {"max_new_tokens": 768, "is_correct": True},
            {"max_new_tokens": 16384, "is_correct": True}]
    line = format_done_line(12, 800, raw, recs, 192.0)
    assert line.startswith("[  12/800] id=102 rpdi_legacy_W16_l1.5")
    assert "604 token, phanh ở token 131" in line and "@768: ĐÚNG" in line and "@16384: ĐÚNG" in line
    assert "@512" not in line and line.endswith("3.2 phút")

    raw.update(forced_close=False, close_index=None, finish_reason="length", output_ids=[0] * 16384)
    line = format_done_line(13, 800, raw, recs, 60.0)
    assert "không đóng </think>, chạm trần" in line
