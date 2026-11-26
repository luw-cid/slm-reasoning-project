import json
import os

import pytest

from common.math_grader import is_equiv, last_boxed

DATA = os.path.join(os.path.dirname(__file__), "..", "common", "data", "math500_test.json")


@pytest.mark.parametrize("pred,gt,ok", [
    (r"\dfrac{14}{3}", r"\frac{14}{3}", True),
    (r"14/3", r"\frac{14}{3}", True),
    (r"90", r"90^\circ", True),
    (r"3\sqrt{13}", r"3\sqrt{13}", True),
    (r"11\sqrt{2}", r"11\sqrt2", True),
    (r"(3, \frac{\pi}{2})", r"\left( 3, \frac{\pi}{2} \right)", True),
    (r"5", r"x=5", True),
    (r"Evelyn", r"\text{Evelyn}", True),
    (r"0.5", r"\frac12", True),
    (r"2,220", r"2220", True),
    (r"4.0", r"4", True),
    (r"6-5i", r"6 - 5i", True),
    (r"\frac{1}{3}", r"0.3333", False),
    (r"\frac{14}{3}", r"\frac{3}{14}", False),
    (None, r"4", False),
])
def test_is_equiv(pred, gt, ok):
    assert is_equiv(pred, gt) is ok


def test_last_boxed_nested_and_after_think():
    text = r"<think> try \boxed{1} </think> so the answer is \boxed{\frac{\sqrt{3}}{2}}."
    assert last_boxed(text) == r"\frac{\sqrt{3}}{2}"
    assert last_boxed("no box here") is None
    assert last_boxed(r"</think> \boxed{12") is None  # cut by the budget


def test_every_math500_answer_matches_itself():
    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)
    bad = [x["ground_truth"] for x in data
           if not is_equiv(last_boxed("</think> so \\boxed{" + x["ground_truth"] + "}."), x["ground_truth"])]
    assert not bad
