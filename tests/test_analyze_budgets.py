import hashlib

import pytest

from method_c_rpdi_ee.vllm_engine.analyze_budgets import (
    classify,
    compare,
    iso_budget,
    paired_bootstrap_ci,
    parse_target,
    validate,
)


def rec(sid, correct, tokens=100, sha="x", think=50, method="m", budget=768):
    return {"sample_id": sid, "is_correct": correct, "total_tokens": tokens, "think_tokens": think,
            "early_exited": False, "hit_budget": False, "response_sha1": sha, "method": method,
            "max_new_tokens": budget}


def test_bootstrap_identical_is_zero():
    out = paired_bootstrap_ci([1, 0, 1, 1], [1, 0, 1, 1], n_boot=500, seed=0)
    assert out["diff"] == 0.0 and out["ci90"] == (0.0, 0.0) and out["ci95"] == (0.0, 0.0)


def test_bootstrap_deterministic():
    a, b = [1, 0] * 10, [0, 0] * 10
    first = paired_bootstrap_ci(a, b, n_boot=1000, seed=3)
    assert first == paired_bootstrap_ci(a, b, n_boot=1000, seed=3)
    assert first["diff"] == pytest.approx(50.0)
    lo95, hi95 = first["ci95"]
    lo90, hi90 = first["ci90"]
    assert lo95 <= lo90 <= 50.0 <= hi90 <= hi95


def test_classify_cases():
    assert classify((2, 4), (1, 5), 3) == "better"
    assert classify((-2, -1), (-2.5, -0.5), 3) == "slightly_worse"
    assert classify((-5, -2), (-6, -1), 3) == "worse"
    assert classify((-1, 1), (-1.5, 1.5), 3) == "equivalent"
    assert classify((-1, 4), (-2, 5), 3) == "noninferior"
    assert classify((-5, 1), (-6, 2), 3) == "inconclusive"


def test_compare_end_to_end():
    t = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    r = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0]
    target = {i: rec(i, bool(c), tokens=100) for i, c in enumerate(t)}
    ref = {i: rec(i, bool(c), tokens=300) for i, c in enumerate(r)}
    out = compare(target, ref, delta=3.0, n_boot=500, seed=0)
    assert out["n"] == 10 and out["target_better"] == 2 and out["reference_better"] == 1
    assert out["diff"] == pytest.approx(10.0)
    assert out["acc_target"] == pytest.approx(70.0) and out["acc_reference"] == pytest.approx(60.0)
    assert out["token_ratio"] == pytest.approx(1 / 3)
    assert out["verdict"] in {"better", "slightly_worse", "worse", "equivalent", "noninferior", "inconclusive"}


def test_validate_text_and_accuracy():
    text_a, text_b = "abc", "abd"
    sha = lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()  # noqa: E731
    vllm = {1: rec(1, True, sha=sha(text_a), think=10), 2: rec(2, False, sha=sha(text_a), think=10)}
    hf = {1: {**rec(1, True, think=10), "response_text": text_a}, 2: {**rec(2, True, think=12), "response_text": text_b}}
    out = validate(vllm, hf, tolerance=2.0)
    assert out["n"] == 2
    assert out["acc_vllm"] == 50.0 and out["acc_hf"] == 100.0 and out["passed"] is False
    assert out["correct_agree_pct"] == 50.0 and out["text_match_pct"] == 50.0 and out["think_equal_pct"] == 50.0


def test_parse_target_and_iso_budget():
    assert parse_target("rpdi_legacy_W16_l1.5@768") == ("rpdi_legacy_W16_l1.5", 768)
    with pytest.raises(ValueError):
        parse_target("standard")
    index = {
        ("standard", 768): {1: rec(1, False), 2: rec(2, False)},
        ("standard", 1536): {1: rec(1, True), 2: rec(2, False)},
        ("standard", 4096): {1: rec(1, True), 2: rec(2, True)},
    }
    assert iso_budget(index, "standard", [1, 2], 50.0) == 1536
    assert iso_budget(index, "standard", [1, 2], 100.0) == 4096
    assert iso_budget(index, "standard", [1, 2], 101.0) is None
