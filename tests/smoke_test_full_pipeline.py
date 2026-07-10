"""Smoke test: run the full engine on each synthetic variant."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from engine.config_loader import load_configs
from engine.metrics import compute_metrics
from engine.decision_logic import run_decision

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
cfg = load_configs()
print(f"config version: {cfg['config_version']}\n")

real = pd.read_csv(SAMPLES / "real_german_credit.csv")
task = "credit_underwriting"
categories = cfg["tasks"][task]["required_categories"]

variants = {
    "high_fidelity": "synthetic_high_fidelity.csv",
    "fairness_degraded": "synthetic_fairness_degraded.csv",
    "privacy_leaky": "synthetic_privacy_leaky.csv",
    "low_utility": "synthetic_low_utility.csv",
}

expected = {
    "high_fidelity": "Approve",
    "fairness_degraded": "Conditional Approval",
    "privacy_leaky": "Reject",
    "low_utility": "Reject",
}

for name, fname in variants.items():
    synth = pd.read_csv(SAMPLES / fname)
    vals = compute_metrics(real, synth, target="target",
                           sensitive="sex", categories=categories)
    dec = run_decision(task, vals, cfg)
    cat_str = " | ".join(f"{c.category}:{c.confidence}" for c in dec.categories)
    ok = "OK" if dec.recommendation == expected[name] else "XX MISMATCH"
    print(f"[{ok}] {name:20s} -> {dec.recommendation:22s} "
          f"(overall={dec.overall_confidence})")
    print(f"        {cat_str}")
    print(f"        metrics: {vals}")
    print(f"        reason: {dec.reason[:100]}")
    print()
