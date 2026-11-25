import pytest

from conftest import FakeTokenizer
from method_c_rpdi_ee.vllm_engine.rpdi_core import (
    ScheduledMonitor,
    MethodSpec,
    RPDIMonitor,
    build_boundary_ids,
    monitor_for,
    parse_method,
)

TOK = FakeTokenizer()
CLOSE = TOK.id_of["</think>"]
NL = TOK.id_of["\n"]
A = TOK.id_of["a"]
DOT = TOK.id_of["."]
WAIT = TOK.id_of["Wait"]
THINK_OPEN = TOK.id_of["<think>"]


def run(monitor, tokens, entropies, argmaxes=None, prompt=(THINK_OPEN,)):
    """Mô phỏng vòng sinh: lần gọi thứ k thấy output_ids = tokens[:k-1] và phân phối có entropy[k-1].

    Trả về danh sách giá trị step() trả về và số lần stats() được gọi.
    """
    calls = {"n": 0}
    returned = []
    for k in range(len(entropies)):

        def stats(k=k):
            calls["n"] += 1
            am = argmaxes[k] if argmaxes is not None else tokens[k]
            return entropies[k], am

        returned.append(monitor.step(list(prompt), tokens[:k], stats))
    return returned, calls["n"]


def paper(W, lam, boundary="newline", ids=None):
    return RPDIMonitor(
        W=W, lam=lam, mode="paper", boundary=boundary,
        boundary_ids=frozenset(ids if ids is not None else {DOT}),
        think_close_id=CLOSE, close_ids=[CLOSE],
    )


def legacy(W, lam, ids=None):
    return RPDIMonitor(
        W=W, lam=lam, mode="legacy", boundary="punct",
        boundary_ids=frozenset(ids if ids is not None else {DOT}),
        think_close_id=CLOSE, close_ids=[CLOSE, NL],
    )


def test_parse_method_keys():
    assert parse_method("standard") == MethodSpec("standard", "standard", 0, None, "")
    assert parse_method("rpdi:512:2.0") == MethodSpec("rpdi_W512_l2.0_wait", "rpdi", 512, 2.0, "wait")
    assert parse_method("rpdi:512:2.0:newline").key == "rpdi_W512_l2.0_newline"
    assert parse_method("rpdi_legacy:16:1.5") == MethodSpec("rpdi_legacy_W16_l1.5", "rpdi_legacy", 16, 1.5, "punct")
    for bad in ("rpdi:16", "rpdi:16:1.5:tab", "hardtrunc", "hardtrunc:x", "random:1", "rpdi_legacy:16:1.5:wait"):
        with pytest.raises(ValueError):
            parse_method(bad)


def test_paper_checks_before_appending():
    # Token thứ 4 là biên. Lần gọi thứ 5: i=4, S_global=12, LTF=(5+5)/2=5, GTF=3 -> RPDI=1.667
    tokens = [A, A, A, DOT, A]
    ents = [1.0, 1.0, 5.0, 5.0, 9.0]
    m = paper(W=2, lam=1.5)
    out, _ = run(m, tokens, ents)
    assert out == [None, None, None, None, CLOSE]
    assert m.early_exited and m.exit_step == 4
    assert m.rpdi_at_exit == pytest.approx(5 / 3)
    assert m.entropies == [1.0, 1.0, 5.0, 5.0]  # entropy của token bị thay bằng </think> không được nạp

    m2 = paper(W=2, lam=1.7)
    out2, _ = run(m2, tokens, ents)
    assert out2 == [None] * 5 and not m2.early_exited and m2.exit_step == -1


def test_paper_requires_i_at_least_W():
    # Biên ở token 1 với W=2: chưa đủ cửa sổ, không được kiểm tra
    m = paper(W=2, lam=0.5)
    out, _ = run(m, [DOT, A, A], [1.0, 1.0, 1.0])
    assert out == [None, None, None]


def test_paper_wait_replaces_token():
    # Chế độ wait: xét token SẮP sinh (argmax), không xét token liền trước
    tokens = [A, A, A, WAIT]
    ents = [1.0, 1.0, 5.0, 2.0]
    m = paper(W=2, lam=1.2, boundary="wait", ids={WAIT})
    out, _ = run(m, tokens, ents)
    # Lần gọi 4: i=3, argmax=Wait, S_global=7, LTF=3, GTF=7/3 -> RPDI=1.286 > 1.2
    assert out == [None, None, None, CLOSE]
    assert m.exit_step == 3


def test_paper_natural_close_stops_monitoring():
    # Mô hình tự sinh </think> ở token 2; sau đó không còn giám sát dù token sau là biên và entropy rất cao
    tokens = [A, CLOSE, DOT, DOT]
    ents = [1.0, 1.0, 50.0, 50.0]
    m = paper(W=1, lam=100.0, ids={DOT, A})
    out, n_stats = run(m, tokens, ents)
    assert out == [None] * 4
    assert m.closed and n_stats == 2 and not m.early_exited


def test_legacy_natural_close_stops_monitoring():
    m = legacy(W=1, lam=100.0, ids={DOT, A})
    tokens = [A, CLOSE, DOT, DOT]
    out, n_stats = run(m, tokens, [1.0, 1.0, 50.0, 50.0])
    assert out == [None] * 4
    assert m.closed and n_stats == 2


def test_legacy_matches_hf_semantics():
    # Legacy: nạp entropy trước, rồi xét biên ở token liền trước (lần gọi đầu xét prompt_ids[-1]).
    # Lần gọi 4: output=[a,a,.] -> nạp H=5 (step=4), token trước là '.', S_global=1+1+5+5=12,
    # LTF=(5+5)/2=5, GTF=3 -> RPDI=1.667 > 1.5 -> ép '</think>' rồi '\n'.
    tokens = [A, A, DOT, CLOSE, NL, A]
    ents = [1.0, 1.0, 5.0, 5.0, 0.0, 0.0]
    m = legacy(W=2, lam=1.5)
    out, n_stats = run(m, tokens, ents)
    assert out == [None, None, None, CLOSE, NL, None]
    assert m.early_exited and m.exit_step == 3
    assert m.entropies == [1.0, 1.0, 5.0, 5.0]
    assert n_stats == 4
    assert m.closed


def test_legacy_first_call_checks_prompt_token():
    # W=1: lần gọi đầu đã có step=1 >= W, token liền trước là prompt_ids[-1] ('.')
    m = legacy(W=1, lam=0.5)
    out, _ = run(m, [A, A], [1.0, 1.0], prompt=(A, DOT))
    assert out[0] == CLOSE and m.exit_step == 0


def test_lam_none_never_exits():
    m = paper(W=0, lam=None, ids=set())
    out, _ = run(m, [A, DOT, A, DOT], [1.0, 9.0, 1.0, 9.0])
    assert out == [None] * 4
    assert m.entropies == [1.0, 9.0, 1.0, 9.0] and m.exit_step == -1


def test_boundary_presets():
    assert build_boundary_ids(TOK, "wait") == frozenset({TOK.id_of["Wait"], TOK.id_of[" Wait"]})
    assert build_boundary_ids(TOK, "newline") == frozenset(
        {TOK.id_of["\n"], TOK.id_of["\n\n"], TOK.id_of[".\n\n"]}
    )
    punct = build_boundary_ids(TOK, "punct")
    for sym in (".", " .", ",", " ,", ";", "?", "!", "\n"):
        assert TOK.id_of[sym] in punct
    with pytest.raises(ValueError):
        build_boundary_ids(TOK, "tab")


def test_monitor_for_uses_spec():
    cache = {}
    m = monitor_for(parse_method("rpdi_legacy:16:1.5"), TOK, cache)
    assert m.mode == "legacy" and m.close_ids == [CLOSE, NL] and m.W == 16 and m.lam == 1.5
    m = monitor_for(parse_method("rpdi:512:2.0"), TOK, cache)
    assert m.mode == "paper" and m.close_ids == [CLOSE] and WAIT in m.boundary_ids
    m = monitor_for(parse_method("standard"), TOK, cache)
    assert m.lam is None and m.mode == "paper"


# ------------------------------------------------------------------ đối chứng: cắt cứng, phanh ngẫu nhiên
NO_STATS = lambda: (_ for _ in ()).throw(AssertionError("không được tính entropy"))  # noqa: E731


def drive(monitor, tokens, prompt=(THINK_OPEN,)):
    """Lần gọi thứ k thấy output_ids = tokens[:k]; trả về các giá trị step() trả về."""
    return [monitor.step(list(prompt), tokens[:k], NO_STATS) for k in range(len(tokens) + 1)]


def test_parse_control_methods():
    assert parse_method("hardtrunc:130") == MethodSpec("hardtrunc_N130", "hardtrunc", 130, None, "")
    assert parse_method("random") == MethodSpec("random", "random", 0, None, "punct")
    assert parse_method("random_matched") == MethodSpec("random_matched", "random_matched", 0, None, "punct")


def test_hardtrunc_forces_close_after_exactly_N_think_tokens():
    m = ScheduledMonitor(exit_at=3, align=False, boundary_ids=frozenset({DOT}),
                         think_close_id=CLOSE, close_ids=[CLOSE, NL])
    out = drive(m, [A, A, A, CLOSE, NL, A])
    # Gọi lần 4 (đã có 3 token suy nghĩ): ép </think>, lần 5 ép xuống dòng, sau đó thả tự do
    assert out == [None, None, None, CLOSE, NL, None, None]
    assert m.early_exited and m.exit_step == 3
    assert m.entropies == []


def test_scheduled_natural_close_first_means_no_exit():
    m = ScheduledMonitor(exit_at=3, align=False, boundary_ids=frozenset(),
                         think_close_id=CLOSE, close_ids=[CLOSE, NL])
    out = drive(m, [A, CLOSE, A, A, A])
    assert out == [None] * 6 and not m.early_exited and m.exit_step == -1


def test_random_waits_for_boundary_after_exit_at():
    m = ScheduledMonitor(exit_at=2, align=True, boundary_ids=frozenset({DOT}),
                         think_close_id=CLOSE, close_ids=[CLOSE, NL])
    # Token 2 không phải biên, token 3 là "." -> phanh ở lần gọi sau token 3
    out = drive(m, [A, A, A, DOT, CLOSE, NL])
    assert out == [None, None, None, None, CLOSE, NL, None]
    assert m.exit_step == 4


def test_scheduled_without_exit_at_never_exits():
    m = ScheduledMonitor(exit_at=None, align=True, boundary_ids=frozenset({DOT}),
                         think_close_id=CLOSE, close_ids=[CLOSE, NL])
    assert drive(m, [DOT] * 5) == [None] * 6 and not m.early_exited


def test_monitor_for_controls():
    cache = {}
    m = monitor_for(parse_method("hardtrunc:130"), TOK, cache)
    assert isinstance(m, ScheduledMonitor) and m.exit_at == 130 and not m.align
    assert m.close_ids == [CLOSE, NL]
    m = monitor_for(parse_method("random"), TOK, cache, exit_at=77)
    assert m.exit_at == 77 and m.align and DOT in m.boundary_ids
    m = monitor_for(parse_method("random_matched"), TOK, cache, exit_at=None)
    assert m.exit_at is None
