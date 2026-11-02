import hashlib

from conftest import FakeTokenizer
from method_c_rpdi_ee.vllm_engine.budgets import expand_to_budgets

TOK = FakeTokenizer()
CLOSE = TOK.id_of["</think>"]
IDS = TOK.id_of


def raw_record(output_ids, close_index, forced=False, exit_step=-1, max_tokens=16):
    return {
        "sample_id": 7,
        "method": "rpdi_W2_l1.5_wait",
        "max_tokens": max_tokens,
        "ground_truth": "7",
        "prompt_ids": [IDS["a"], IDS["<think>"]],
        "output_ids": output_ids,
        "close_index": close_index,
        "forced_close": forced,
        "exit_step": exit_step,
        "rpdi_at_exit": 2.5 if forced else 0.0,
    }


ANSWERED = [IDS[t] for t in ["a", "b", "c", "a", "b", "c", "</think>", "\\boxed{", "7", "}", "<eos>"]]


def by_budget(records):
    return {r["max_new_tokens"]: r for r in records}


def test_expand_truncates_and_scores():
    recs = by_budget(expand_to_budgets(raw_record(ANSWERED, 6), TOK.decode, CLOSE, [4, 12]))
    r4, r12 = recs[4], recs[12]
    assert (r4["total_tokens"], r4["think_tokens"], r4["answer_tokens"]) == (4, 4, 0)
    assert r4["is_correct"] is False and r4["hit_budget"] is True
    assert (r12["total_tokens"], r12["think_tokens"], r12["answer_tokens"]) == (11, 6, 5)
    assert r12["is_correct"] is True and r12["predicted_answer"] == "7" and r12["hit_budget"] is False
    assert r12["response_sha1"] == hashlib.sha1(TOK.decode(ANSWERED).encode("utf-8")).hexdigest()
    for r in (r4, r12):
        assert r["sample_id"] == 7 and r["method"] == "rpdi_W2_l1.5_wait" and r["ground_truth"] == "7"
        assert r["engine"] == "vllm" and r["source_max_tokens"] == 16


def test_expand_exit_after_budget_not_counted():
    raw = raw_record(ANSWERED, 6, forced=True, exit_step=6)
    recs = by_budget(expand_to_budgets(raw, TOK.decode, CLOSE, [4, 8]))
    assert recs[4]["early_exited"] is False and recs[4]["exit_step"] == -1 and recs[4]["rpdi_at_exit"] == 0.0
    assert recs[8]["early_exited"] is True and recs[8]["exit_step"] == 6 and recs[8]["rpdi_at_exit"] == 2.5


def test_expand_short_output_all_budgets():
    short = [IDS[t] for t in ["a", "b", "c", "a", "<eos>"]]
    recs = by_budget(expand_to_budgets(raw_record(short, None), TOK.decode, CLOSE, [5, 8]))
    assert recs[5]["total_tokens"] == 5 and recs[5]["hit_budget"] is True
    assert recs[8]["total_tokens"] == 5 and recs[8]["hit_budget"] is False
    assert recs[8]["think_tokens"] == 5 and recs[8]["early_exited"] is False


def test_expand_skips_budgets_above_run():
    recs = expand_to_budgets(raw_record(ANSWERED, 6, max_tokens=8), TOK.decode, CLOSE, [4, 8, 16])
    assert sorted(r["max_new_tokens"] for r in recs) == [4, 8]
