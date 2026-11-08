"""test_gate.py -- tests for the pure decide() function (synthetic numbers only).
Thresholds are chosen so boundary arithmetic is exact in floating point."""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eta_mlflow import decide  # noqa: E402

PROD = dict(mae=10.0, mae_rush=12.0, mae_rain=11.0, p95_ms=2.0)


def cand(**kw):
    return {**dict(mae=5.0, mae_rush=6.0, mae_rain=5.5, p95_ms=3.0), **kw}


def test_clear_improvement_promotes():
    r = decide(PROD, cand())
    assert r["promote"] is True
    assert set(r["criteria"]) == {"overall_gain", "latency", "rush_slice", "rain_slice"}
    assert all(r["criteria"].values())


def test_gain_boundary_inside_exact_and_outside():
    # min_rel_gain=0.25 -> needs cand.mae <= 7.5
    assert decide(PROD, cand(mae=7.49), min_rel_gain=0.25)["promote"] is True
    assert decide(PROD, cand(mae=7.5), min_rel_gain=0.25)["promote"] is True
    r = decide(PROD, cand(mae=7.51), min_rel_gain=0.25)
    assert r["promote"] is False and r["criteria"]["overall_gain"] is False


def test_no_improvement_or_worse_is_rejected():
    assert decide(PROD, cand(mae=10.0))["promote"] is False
    assert decide(PROD, cand(mae=11.0))["promote"] is False


def test_latency_boundary():
    assert decide(PROD, cand(p95_ms=10.0))["promote"] is True
    r = decide(PROD, cand(p95_ms=10.01))
    assert r["promote"] is False and r["criteria"]["latency"] is False
    assert decide(PROD, cand(p95_ms=4.0), max_p95_ms=3.9)["promote"] is False


def test_rush_slice_regression_blocks_even_with_big_overall_gain():
    # max_slice_regression=0.5 -> rush limit = 12 * 1.5 = 18
    assert decide(PROD, cand(mae_rush=18.0), max_slice_regression=0.5)["promote"] is True
    r = decide(PROD, cand(mae_rush=18.1), max_slice_regression=0.5)
    assert r["promote"] is False
    assert r["criteria"]["rush_slice"] is False and r["criteria"]["overall_gain"] is True


def test_rain_slice_regression_blocks():
    # rain limit = 11 * 1.5 = 16.5
    assert decide(PROD, cand(mae_rain=16.5), max_slice_regression=0.5)["promote"] is True
    r = decide(PROD, cand(mae_rain=16.6), max_slice_regression=0.5)
    assert r["promote"] is False and r["criteria"]["rain_slice"] is False


def test_small_slice_regression_within_default_tolerance_is_ok():
    # +2% on rush is below the default 5%
    assert decide(PROD, cand(mae_rush=12.24))["promote"] is True
    assert decide(PROD, cand(mae_rush=12.8))["promote"] is False  # +6.7% > 5%


def test_strict_gain_threshold_rejects_good_model():
    # candidate is 50% better, but min_rel_gain=0.60 demands more
    r = decide(PROD, cand(), min_rel_gain=0.60)
    assert r["promote"] is False
    assert r["criteria"]["overall_gain"] is False and r["criteria"]["latency"] is True


def test_decide_is_pure_inputs_not_mutated():
    p, c = copy.deepcopy(PROD), cand()
    c0 = copy.deepcopy(c)
    decide(p, c)
    assert p == PROD and c == c0
