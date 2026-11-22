import random

from _factories import FakeRetriever, make_results
from sia.evaluation.gate import (auc, blocked, percentile, retained, score_questions,
                                 split_half, threshold_for_retention)


def test_auc_perfect_chance_and_reversed():
    assert auc([0.8, 0.9], [0.1, 0.2]) == 1.0
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auc([0.1], [0.9]) == 0.0


def test_threshold_keeps_the_requested_share_of_answerable_questions():
    scores = [i / 100 for i in range(50, 100)]          # 50 answerable questions
    for target in (1.0, 0.98, 0.9):
        tau = threshold_for_retention(scores, target)
        assert retained(scores, tau) >= target
    assert threshold_for_retention(scores, 1.0) == 0.5
    assert threshold_for_retention(scores, 0.9) == 0.55   # 5 of 50 may be lost


def test_blocked_and_retained_use_a_strict_less_than_gate():
    assert blocked([0.4, 0.6, 0.7], 0.6) == 1 / 3          # 0.6 itself passes the gate
    assert retained([0.4, 0.6, 0.7], 0.6) == 2 / 3


def test_split_half_is_deterministic_and_disjoint():
    a1, b1 = split_half(list(range(10)), random.Random(1))
    a2, b2 = split_half(list(range(10)), random.Random(1))
    assert (a1, b1) == (a2, b2) and sorted(a1 + b1) == list(range(10))


def test_score_questions_extracts_features():
    rows = score_questions(FakeRetriever(make_results(5, top_score=0.8)),
                           [{"id": "Q1", "question": "q?", "category": "near_miss", "doc": None}])
    r = rows[0]
    assert r["id"] == "Q1" and r["category"] == "near_miss"
    assert r["top1"] == 0.8 and abs(r["margin"] - 0.02) < 1e-9 and abs(r["mean3"] - 0.78) < 1e-9
    assert r["top1_title"] == "Guide 1" and r["top1_loc"] == "p. 1"


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([], 50) != percentile([], 50)
