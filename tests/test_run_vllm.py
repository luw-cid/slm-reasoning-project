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
