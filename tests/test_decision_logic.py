"""
test_decision_logic.py
-----------------------
Unit tests for the deterministic decision engine (Stages 1-4).
These test the LOGIC in isolation with synthetic metric values -- no data,
no model training -- so they run instantly and pin down every rule.

Run:  python -m pytest tests/ -v      (or)      python tests/test_decision_logic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config_loader import load_configs
from engine.decision_logic import (
    interpret_metric, aggregate_category, run_decision, MetricResult,
)

cfg = load_configs()


# ---- Stage 1: interpreter ---------------------------------------------------

def test_higher_better_bands():
    # tstr: strong>=0.95, weak<0.90
    assert interpret_metric("tstr", 0.97, cfg).strength == "strong"
    assert interpret_metric("tstr", 0.92, cfg).strength == "moderate"
    assert interpret_metric("tstr", 0.80, cfg).strength == "weak"


def test_lower_better_bands():
    # membership_inference: strong<=0.05, weak>0.10
    assert interpret_metric("membership_inference", 0.02, cfg).strength == "strong"
    assert interpret_metric("membership_inference", 0.07, cfg).strength == "moderate"
    assert interpret_metric("membership_inference", 0.20, cfg).strength == "weak"


# ---- Stage 2: aggregation (graduated) --------------------------------------

def _mk(strength):
    return MetricResult("k", "K", "utility", 0.0, strength, strength != "weak",
                        "higher_better", 0.9, 0.8, "")


def test_aggregation_all_strong_is_high():
    cat = aggregate_category("utility", [_mk("strong"), _mk("strong"), _mk("strong")])
    assert cat.confidence == "high"


def test_aggregation_one_moderate_still_high():
    cat = aggregate_category("utility", [_mk("strong"), _mk("strong"), _mk("moderate")])
    assert cat.confidence == "high"


def test_aggregation_two_moderate_is_medium():
    cat = aggregate_category("utility", [_mk("strong"), _mk("moderate"), _mk("moderate")])
    assert cat.confidence == "medium"


def test_aggregation_one_weak_is_medium():
    cat = aggregate_category("utility", [_mk("strong"), _mk("strong"), _mk("weak")])
    assert cat.confidence == "medium"


def test_aggregation_two_weak_is_low():
    cat = aggregate_category("utility", [_mk("weak"), _mk("weak"), _mk("strong")])
    assert cat.confidence == "low"


# ---- Stage 3-4: policy engine + recommendation -----------------------------

def _strong_credit_values():
    """Metric values that make every Credit category High."""
    return {
        "tstr": 0.99, "auc_gap": 0.0, "decision_agreement": 0.99,
        "demographic_parity": 0.0, "equal_opportunity": 0.0, "subgroup_performance": 1.0,
        "membership_inference": 0.0, "nn_distance_ratio": 1.2, "duplicate_rate": 0.0,
        "brier_score": 0.05, "expected_calibration_error": 0.0,
    }


def test_all_high_approves():
    dec = run_decision("credit_underwriting", _strong_credit_values(), cfg)
    assert dec.recommendation == "Approve"
    assert dec.overall_confidence == "high"


def test_fairness_medium_caps_to_conditional():
    v = _strong_credit_values()
    # push fairness to Medium: two moderate metrics
    v["demographic_parity"] = 0.08   # moderate
    v["equal_opportunity"] = 0.08    # moderate
    dec = run_decision("credit_underwriting", v, cfg)
    fairness = next(c for c in dec.categories if c.category == "fairness")
    assert fairness.confidence == "medium"
    assert dec.recommendation == "Conditional Approval"


def test_privacy_low_forces_reject():
    v = _strong_credit_values()
    # two weak privacy metrics -> privacy Low -> hard flag -> Reject
    v["membership_inference"] = 0.30
    v["duplicate_rate"] = 0.30
    dec = run_decision("credit_underwriting", v, cfg)
    assert dec.hard_flag is not None
    assert dec.recommendation == "Reject"


def test_utility_low_hard_floor_reject():
    v = _strong_credit_values()
    v["tstr"] = 0.5
    v["auc_gap"] = 0.4
    dec = run_decision("credit_underwriting", v, cfg)
    assert dec.overall_confidence == "low"
    assert dec.recommendation == "Reject"


# ---- simple runner ----------------------------------------------------------

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}  {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
