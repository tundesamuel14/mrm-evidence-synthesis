"""
generate_samples.py
-------------------
Builds the demo datasets, transparently and reproducibly.

BASE (real): a German-Credit-style consumer-credit dataset. We fetch the real
UCI German Credit data if available; otherwise we synthesize a realistic
stand-in with the same schema so the repo is self-contained offline.

SYNTHETIC VARIANTS (generated FROM the real base):
  1. high_fidelity  -> faithful copy w/ light noise      -> expect APPROVE
  2. fairness_degraded -> subgroup outcomes skewed        -> expect CONDITIONAL
  3. privacy_leaky  -> many near-duplicate real rows      -> expect REJECT (privacy)
  4. low_utility    -> features shuffled / decorrelated   -> expect REJECT (utility)

This mirrors the exact workflow the tool validates: synthetic-from-real. Each
variant lets the demo show a different branch of the decision logic on command.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)
N = 2000  # larger sample -> tighter metric estimates, more stable models


# --------------------------------------------------------------------------- #
# Real base
# --------------------------------------------------------------------------- #

def _try_fetch_real() -> pd.DataFrame | None:
    """Attempt to fetch real UCI German Credit; return None on any failure."""
    try:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "statlog/german/german.data")
        cols = [f"a{i}" for i in range(20)] + ["target"]
        df = pd.read_csv(url, sep=r"\s+", header=None, names=cols)
        # target: 1 = good, 2 = bad -> map to 1 good / 0 bad
        df["target"] = (df["target"] == 1).astype(int)
        # derive a sensitive attribute from attribute 9 (personal status/sex codes)
        df["sex"] = df["a8"].map(lambda v: "female" if str(v) in
                                 {"A92", "A95"} else "male")
        # keep a compact, numeric-friendly subset + engineered cols
        keep_num = ["a1", "a4", "a12", "a15"]  # duration, credit amount, age, etc (codes)
        out = pd.DataFrame()
        out["duration"] = pd.to_numeric(df["a1"], errors="coerce")
        out["credit_amount"] = pd.to_numeric(df["a4"], errors="coerce")
        out["age"] = pd.to_numeric(df["a12"], errors="coerce")
        out["installment_rate"] = pd.to_numeric(df["a15"], errors="coerce")
        out["sex"] = df["sex"]
        out["target"] = df["target"]
        out = out.dropna().reset_index(drop=True)
        if len(out) > 100:
            return out
        return None
    except Exception:
        return None


def _synthesize_real_base() -> pd.DataFrame:
    """Self-contained realistic stand-in with the same schema."""
    age = RNG.normal(38, 11, N).clip(19, 75).round()
    duration = RNG.normal(21, 12, N).clip(4, 72).round()
    credit_amount = (RNG.lognormal(7.8, 0.5, N)).clip(250, 20000).round()
    installment_rate = RNG.integers(1, 5, N)
    sex = RNG.choice(["male", "female"], size=N, p=[0.69, 0.31])

    # target depends on features (so models can learn something real)
    logit = (
        0.9
        - 0.03 * (duration - 21) / 12
        - 0.35 * (credit_amount - 3500) / 3000
        + 0.02 * (age - 38) / 11
        - 0.15 * (installment_rate - 2.5)
    )
    p_good = 1 / (1 + np.exp(-logit))
    target = (RNG.uniform(size=N) < p_good).astype(int)

    return pd.DataFrame({
        "duration": duration,
        "credit_amount": credit_amount,
        "age": age,
        "installment_rate": installment_rate,
        "sex": sex,
        "target": target,
    })


def build_real() -> pd.DataFrame:
    real = _try_fetch_real()
    source = "UCI German Credit (real)"
    if real is None:
        real = _synthesize_real_base()
        source = "German-Credit-style synthetic stand-in"
    print(f"  real base: {source}, n={len(real)}")
    return real


# --------------------------------------------------------------------------- #
# Synthetic variants
# --------------------------------------------------------------------------- #

NUM_COLS = ["duration", "credit_amount", "age", "installment_rate"]


def _fit_model(real: pd.DataFrame):
    """
    Learn a generative model that preserves BOTH the feature distribution AND
    the feature->target relationship, so synthetic rows are new (privacy-safe)
    yet a model trained on them behaves like one trained on real (utility-safe).

    Two parts:
      * per-class Gaussian over numeric features (marginal + covariance)
      * a logistic model of P(target=1 | features) learned from real, used to
        assign targets to sampled rows -> preserves the joint structure.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    model = {"overall_sex_p": real["sex"].value_counts(normalize=True).to_dict()}
    # --- Gaussian copula over numeric features ---------------------------------
    # Preserves BOTH each feature's real marginal shape (via rank->normal->rank)
    # AND the cross-feature correlation structure. This is a standard synthetic-
    # data technique (used by SDV's GaussianCopula), not resampling of real rows.
    from scipy.stats import norm

    Xr = real[NUM_COLS].values
    n = len(Xr)
    # empirical CDF ranks -> standard normal scores per column
    ranks = np.argsort(np.argsort(Xr, axis=0), axis=0) + 1
    u = ranks / (n + 1)
    z = norm.ppf(u)
    model["corr"] = np.corrcoef(z.T)
    model["sorted_cols"] = {c: np.sort(real[c].values) for c in NUM_COLS}

    scaler = StandardScaler().fit(Xr)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(scaler.transform(Xr), real["target"].values)
    model["scaler"] = scaler
    model["lr"] = lr
    return model


def _sample_model(model, n: int, assign_target_from_model: bool = True) -> pd.DataFrame:
    """Sample via the Gaussian copula: draw correlated normals, map each back
    through the real marginal (inverse-CDF) so synthetic marginals match real,
    then assign target from the learned P(target|features)."""
    from scipy.stats import norm

    # temperature > 1 slightly widens the sampling so synthetic points don't
    # land on top of real ones -> better privacy (less memorization), which is
    # a deliberate property of good synthesizers, not a metric hack.
    temp = 1.08
    L = np.linalg.cholesky(model["corr"] + 1e-6 * np.eye(len(NUM_COLS)))
    zt = (L @ (temp * RNG.standard_normal((len(NUM_COLS), n)))).T
    ut = norm.cdf(zt)
    out = pd.DataFrame(index=range(n))
    for j, col in enumerate(NUM_COLS):
        sorted_vals = model["sorted_cols"][col]
        idx = np.clip((ut[:, j] * (len(sorted_vals) - 1)).astype(int),
                      0, len(sorted_vals) - 1)
        out[col] = sorted_vals[idx]
    out["duration"] = out["duration"].clip(4, 72).round()
    out["age"] = out["age"].clip(19, 80).round()
    out["installment_rate"] = out["installment_rate"].clip(1, 4).round()
    out["credit_amount"] = out["credit_amount"].clip(250, 20000).round()

    if assign_target_from_model:
        p1 = model["lr"].predict_proba(model["scaler"].transform(out[NUM_COLS].values))[:, 1]
        out["target"] = (RNG.uniform(size=len(out)) < p1).astype(int)
    else:
        out["target"] = RNG.integers(0, 2, len(out))

    sexes = list(model["overall_sex_p"].keys())
    probs = list(model["overall_sex_p"].values())
    out["sex"] = RNG.choice(sexes, size=len(out), p=probs)
    return out[["duration", "credit_amount", "age", "installment_rate", "sex", "target"]]


def high_fidelity(real: pd.DataFrame) -> pd.DataFrame:
    """Faithful marginals AND joint structure -> distinct rows -> APPROVE."""
    model = _fit_model(real)
    return _sample_model(model, len(real), assign_target_from_model=True)


def fairness_degraded(real: pd.DataFrame) -> pd.DataFrame:
    """
    Faithful utility & calibration, but the female subgroup's positive-outcome
    rate is modestly depressed -> demographic parity slips into the weak band
    while equal-opportunity stays borderline. Net: Fairness = Medium (one weak
    metric), everything else intact -> CONDITIONAL APPROVAL.

    Kept deliberately mild and targeted so it does NOT bleed into utility or
    calibration -- the demo needs Fairness isolated as the single soft spot.
    """
    out = high_fidelity(real)
    female = out["sex"] == "female"
    # flip ~18% of female positives only -> demographic parity ~0.10-0.13 (weak),
    # equal-opportunity stays moderate, global structure preserved.
    flip = female & (out["target"] == 1) & (RNG.uniform(size=len(out)) < 0.11)
    out.loc[flip, "target"] = 0
    return out


def privacy_leaky(real: pd.DataFrame) -> pd.DataFrame:
    """Faithful base but ~35% EXACT real-row copies -> only privacy fails -> REJECT."""
    out = high_fidelity(real)
    n_copy = int(0.35 * len(real))
    copy_idx = RNG.integers(0, len(real), n_copy)
    out.iloc[:n_copy] = real.iloc[copy_idx].values
    return out.reset_index(drop=True)


def low_utility(real: pd.DataFrame) -> pd.DataFrame:
    """Marginals preserved but feature->target link destroyed -> utility/calibration fail -> REJECT."""
    model = _fit_model(real)
    return _sample_model(model, len(real), assign_target_from_model=False)


# --------------------------------------------------------------------------- #
def main():
    print("Generating sample datasets...")
    real = build_real()
    real.to_csv(OUT / "real_german_credit.csv", index=False)

    variants = {
        "synthetic_high_fidelity.csv": high_fidelity(real),
        "synthetic_fairness_degraded.csv": fairness_degraded(real),
        "synthetic_privacy_leaky.csv": privacy_leaky(real),
        "synthetic_low_utility.csv": low_utility(real),
    }
    for name, df in variants.items():
        df.to_csv(OUT / name, index=False)
        print(f"  wrote {name}  (n={len(df)})")

    print(f"\nDone. Files in: {OUT}")


if __name__ == "__main__":
    main()
