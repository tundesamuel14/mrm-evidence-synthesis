# MRM Evidence Synthesis Framework

**Turn synthetic-data validation metrics into an auditable deployment recommendation.**

Banks already compute dozens of synthetic-data validation metrics — KS, Wasserstein, TSTR, membership inference, fairness, calibration, and more. What they lack is a structured, auditable way to turn all those conflicting numbers into a defensible decision: *given all of this evidence, should this synthetic dataset be deployed for this specific task, and why?*

This project is that missing layer. It is **not** another validation metric. It is a decision-support layer that sits on top of existing metrics, organizes them into evidence categories, applies risk-aware policy, and produces an auditable recommendation — while keeping human validators as the final decision-makers.

> This is an independent prototype built to explore Model Risk Management (MRM) workflows for synthetic data. It is not affiliated with or endorsed by any financial institution.

---

## The idea in one picture

```
raw metrics  ->  evidence strength  ->  category confidence  ->  overall confidence  ->  recommendation
  (Stage 1)         (Stage 1)              (Stage 2)              (Stage 3, risk-aware)     (Stage 4)
                                                                                              |
                                                                                          (Stage 5)
                                                                                       decision logged
                                                                                    for audit + future ML
```

Every step is **deterministic**, **configurable**, and **logged**. No black boxes.

---

## What it does

1. **Load** a real dataset and a synthetic dataset to be validated.
2. **Select** the downstream task (Credit Underwriting, Fraud Detection, AML, Stress Testing). The task sets the **risk tier** and the **required evidence categories**.
3. **Run real validation metrics** — TSTR, AUC gap, fairness (demographic parity, equal opportunity), privacy (membership inference, duplicate rate, nearest-neighbour distance), calibration (Brier, ECE), and more. These are genuinely computed from your data, not mocked.
4. **Organize** the metrics into evidence categories, each with a confidence level (High / Medium / Low) — *interpreted*, never averaged.
5. **Apply risk-aware policy** (an implementation of SR 11-7's proportionality principle): for a high-risk task, a non-negotiable category below High caps the overall confidence; a privacy failure forces rejection.
6. **Recommend**: Approve / Conditional Approval / Reject — always with a **reason** and **suggested remediation**.
7. **Generate an auditable MRM report** you can download.

---

## Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. (optional) regenerate the bundled sample datasets
python data/scripts/generate_samples.py

# 3. run the app
streamlit run app.py
```

Then open the local URL Streamlit prints. In the sidebar, pick a bundled synthetic sample (or upload your own real + synthetic CSVs), choose a task, and click **Run validation**.

---

## The bundled demo datasets

The real base is a German-Credit-style consumer-credit dataset. Four synthetic variants are generated *from* it (the exact workflow the tool validates), each engineered to exercise a different branch of the decision logic:

| Synthetic dataset | Engineered flaw | Expected outcome |
|---|---|---|
| `synthetic_high_fidelity` | none — faithful copula synthesis | **Approve** |
| `synthetic_fairness_degraded` | subgroup outcome skew | **Conditional Approval** (fairness caps a high-risk task) |
| `synthetic_privacy_leaky` | ~35% near-copies of real rows | **Reject** (privacy is non-negotiable) |
| `synthetic_low_utility` | feature→target link destroyed | **Reject** (utility/calibration hit the hard floor) |

The generation script (`data/scripts/generate_samples.py`) is fully transparent — every variant's construction is documented in code.

---

## Design principles

- **Configurable, not hardcoded.** Every threshold lives in `config/*.yaml`, owned by a policy team. Changing a number is a config edit, not a code change.
- **Justified, not naked.** Every threshold carries a one-line rationale that travels into the audit report. See `config/metric_thresholds.yaml`.
- **Regulation gives principles; the institution supplies the parameters.** None of the specific numbers come from SR 11-7. SR 11-7 gives the *proportionality principle*; the thresholds here are a configurable *implementation* of it. The software enforces the bank's implementation of SR 11-7, not SR 11-7 itself.
- **Interpreted, not averaged.** Category confidence uses graduated demotion (a single weak metric matters), never a numeric mean that would hide which weakness drove the result.
- **Logged for the right future.** Every decision is recorded with a blank slot for its eventual *production outcome* — because the future ML target is "did the deployed model actually work," not "what did the committee decide." Training on past decisions only imitates the committee; training on outcomes can correct it.

---

## Repository layout

```
.
├── app.py                          # Streamlit web app (the workflow)
├── engine/
│   ├── config_loader.py            # loads configs + a version hash for audit
│   ├── metrics.py                  # Stage 1: REAL metric computations
│   ├── decision_logic.py           # Stages 1-4: interpret, aggregate, policy, recommend
│   ├── logger.py                   # Stage 5: audit log w/ outcome slot
│   └── report.py                   # auditable MRM report generator
├── config/
│   ├── task_templates.yaml         # task -> risk tier + required categories
│   ├── metric_thresholds.yaml      # per-metric bands + rationale
│   └── policy_rules.yaml           # risk-aware aggregation + recommendation map
├── data/
│   ├── scripts/generate_samples.py # transparent sample-data generator
│   └── samples/                    # real + 4 synthetic CSVs
├── tests/
│   └── test_decision_logic.py      # unit tests for the engine
├── requirements.txt
└── LICENSE
```

---

## Testing

```bash
python tests/test_decision_logic.py        # simple runner
# or, if you have pytest:
python -m pytest tests/ -v
```

The tests pin down every rule in the engine (banding, graduated aggregation, risk caps, hard flags) using synthetic metric values, so they run instantly.

---

## Roadmap (beyond the MVP)

The MVP is deliberately deterministic. The architecture is built so each stage can become more intelligent **without discarding the auditable rule engine**:

- **Phase 2 — Knowledge-driven interpreter.** Replace the threshold table with a component that weighs regulation, policy, task, and model type. The stage's input/output contract is unchanged, so nothing downstream moves.
- **Phase 3 — Learn from MRM decisions.** Explainable ML (e.g. SHAP / EBM) to standardize reviewers. Rules stay visible.
- **Phase 4 — Learn from production outcomes.** The real target: did deployment succeed? This can *correct* historical habits rather than copy them. Stage 5's logger exists now precisely so this is possible later.
- **Always — humans decide.** ML suggests rule changes; a human approves; rules update. The system is an MRM co-pilot, not an autonomous governor.

---

## A note on scope

This is a decision-support prototype. Recommendations are **not** automated approvals; a human MRM validator remains the final decision-maker. The metrics, thresholds, and policies here are reasonable, documented defaults intended to demonstrate the workflow — not certified organizational policy.
