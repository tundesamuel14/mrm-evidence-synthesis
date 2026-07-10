"""
decision_logic.py
-----------------
The deterministic decision engine: Stages 1-4 from the specification.

  Stage 1  Interpreter        raw metric value      -> strength (strong/moderate/weak)
  Stage 2  Aggregation        strengths in category -> category confidence (high/med/low)
  Stage 3  Policy Engine      category confidences  -> overall confidence  (risk-aware)
  Stage 4  Recommendation     overall + hard flags  -> Approve/Conditional/Reject

Every function is pure and independently testable. No metric is computed here;
this module only INTERPRETS values that Stage-1 metric code produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---- ordinal helpers --------------------------------------------------------

_STRENGTH_ORDER = {"weak": 0, "moderate": 1, "strong": 2}
_CONF_ORDER = {"low": 0, "medium": 1, "high": 2}
_CONF_FROM_INT = {0: "low", 1: "medium", 2: "high"}


def _min_conf(a: str, b: str) -> str:
    return a if _CONF_ORDER[a] <= _CONF_ORDER[b] else b


# ---- data objects -----------------------------------------------------------

@dataclass
class MetricResult:
    """One computed metric plus its Stage-1 interpretation."""
    key: str
    display_name: str
    category: str
    value: float
    strength: str            # strong | moderate | weak
    passed: bool             # strong or moderate == "acceptable"; weak == not
    direction: str
    strong_at: float
    weak_below: float
    rationale: str
    critical: bool = False


@dataclass
class CategoryResult:
    category: str
    confidence: str          # high | medium | low
    metrics: list = field(default_factory=list)
    reason: str = ""


@dataclass
class Decision:
    task: str
    task_display: str
    risk_tier: str
    categories: list          # list[CategoryResult]
    overall_confidence: str
    recommendation: str
    fired_rule_id: Optional[str]
    reason: str
    hard_flag: Optional[str]
    suggested_actions: list = field(default_factory=list)
    config_version: str = ""


# ---- Stage 1: Evidence Interpreter -----------------------------------------

def interpret_metric(key: str, value: float, cfg: dict) -> MetricResult:
    """Map one raw metric value to an evidence-strength band."""
    spec = cfg["metrics"][key]
    direction = spec["direction"]
    strong_at = spec["strong_at"]
    weak_below = spec["weak_below"]

    if direction == "higher_better":
        if value >= strong_at:
            strength = "strong"
        elif value >= weak_below:
            strength = "moderate"
        else:
            strength = "weak"
    elif direction == "lower_better":
        if value <= strong_at:
            strength = "strong"
        elif value <= weak_below:
            strength = "moderate"
        else:
            strength = "weak"
    else:
        raise ValueError(f"Unknown direction '{direction}' for metric '{key}'")

    return MetricResult(
        key=key,
        display_name=spec.get("display_name", key),
        category=spec["category"],
        value=value,
        strength=strength,
        passed=strength != "weak",
        direction=direction,
        strong_at=strong_at,
        weak_below=weak_below,
        rationale=spec.get("rationale", ""),
        critical=bool(spec.get("critical", False)),
    )


# ---- Stage 2: Category Aggregation (graduated demotion) --------------------

def aggregate_category(category: str, metrics: list) -> CategoryResult:
    """
    Graduated demotion (MVP rule):
        no weak, at most one moderate       -> high
        no weak, two or more moderate       -> medium
        exactly one weak                    -> medium
        two or more weak                    -> low

    Rationale: near-perfect evidence should not be capped by a single borderline
    metric, but two soft spots, or any hard failure, do lower confidence. Never
    averages -- a single weak metric demotes regardless of the others.
    """
    weak = [m for m in metrics if m.strength == "weak"]
    moderate = [m for m in metrics if m.strength == "moderate"]

    n_weak = len(weak)
    n_mod = len(moderate)

    if n_weak >= 2:
        conf = "low"
        reason = f"{n_weak} metrics weak -> Low."
    elif n_weak == 1:
        conf = "medium"
        reason = f"One weak metric ({weak[0].display_name}) -> capped at Medium."
    elif n_mod >= 2:
        conf = "medium"
        reason = f"{n_mod} moderate metrics (no weak) -> Medium."
    elif n_mod == 1:
        conf = "high"
        reason = f"All metrics acceptable; one moderate ({moderate[0].display_name}) tolerated -> High."
    else:
        conf = "high"
        reason = "All metrics strong -> High."

    return CategoryResult(category=category, confidence=conf, metrics=metrics, reason=reason)


# ---- Stage 3: Policy Engine (risk-aware) -----------------------------------

def _cat_map(categories: list) -> dict:
    return {c.category: c for c in categories}


def apply_policy(task_key: str, categories: list, cfg: dict):
    """
    Return (overall_confidence, fired_rule_id, reason, hard_flag_id, hard_flag_reason).
    Rules evaluated top-to-bottom; first cap wins. Hard flags force Reject later.
    """
    task_spec = cfg["tasks"][task_key]
    risk_tier = task_spec["risk_tier"]
    required = task_spec["required_categories"]
    non_neg = task_spec.get("non_negotiable", [])

    cmap = _cat_map(categories)
    req_confs = {c: cmap[c].confidence for c in required if c in cmap}

    # --- hard flags first (checked, but only *reported* here; recommendation
    #     mapper enforces the Reject) ---
    hard_flag_id = None
    hard_flag_reason = None
    for flag in cfg["policy"].get("hard_flags", []):
        applies = True
        if "applies_to_tiers" in flag and risk_tier not in flag["applies_to_tiers"]:
            applies = False
        if "applies_to_tasks" in flag and task_key not in flag["applies_to_tasks"]:
            applies = False
        if not applies:
            continue
        # parse "X == low"
        cond = flag["when"]
        cat_name, _, level = cond.partition("==")
        cat_name = cat_name.strip()
        level = level.strip()
        if cmap.get(cat_name) and cmap[cat_name].confidence == level:
            hard_flag_id = flag["id"]
            hard_flag_reason = flag["reason"]
            break

    # --- risk-tier cap rules ---
    tier_rules = cfg["policy"]["risk_tiers"][risk_tier]["rules"]
    overall = None
    fired = None
    reason = ""

    for rule in tier_rules:
        rid = rule["id"]

        if rid.endswith("low_floor"):
            if any(v == "low" for v in req_confs.values()):
                overall, fired, reason = "low", rid, rule["reason"]
                break

        elif "nonneg_below_high" in rid:
            if any(_CONF_ORDER[cmap[c].confidence] < _CONF_ORDER["high"]
                   for c in non_neg if c in cmap):
                overall = "medium"
                fired, reason = rid, rule["reason"]
                break

        elif "nonneg_below_medium" in rid:
            if any(_CONF_ORDER[cmap[c].confidence] < _CONF_ORDER["medium"]
                   for c in non_neg if c in cmap):
                overall = "medium"
                fired, reason = rid, rule["reason"]
                break

        elif "two_mediums" in rid:
            if sum(1 for v in req_confs.values() if v == "medium") >= 2:
                overall = "medium"
                fired, reason = rid, rule["reason"]
                break

        elif rid.endswith("default"):
            weakest = min(req_confs.values(), key=lambda x: _CONF_ORDER[x])
            overall, fired, reason = weakest, rid, rule["reason"]
            break

    if overall is None:  # safety fallback
        overall = min(req_confs.values(), key=lambda x: _CONF_ORDER[x])
        fired, reason = "fallback", "Overall = weakest required category."

    return overall, fired, reason, hard_flag_id, hard_flag_reason


# ---- Stage 4: Recommendation Mapper ----------------------------------------

def map_recommendation(overall: str, hard_flag_id: Optional[str], cfg: dict) -> str:
    has_flag = hard_flag_id is not None
    for row in cfg["policy"]["recommendation_map"]:
        row_overall = row["overall"]
        row_flag = row["hard_flag"]
        if row_flag and has_flag:
            return row["recommendation"]
        if not row_flag and not has_flag and row_overall == overall:
            return row["recommendation"]
    return "Reject"  # conservative fallback


# ---- Orchestration: run all stages -----------------------------------------

def run_decision(task_key: str, metric_values: dict, cfg: dict) -> Decision:
    """
    metric_values: {metric_key: raw_value} for every metric the task needs.
    Returns a fully-populated Decision object (Stages 1-4).
    """
    task_spec = cfg["tasks"][task_key]
    required = task_spec["required_categories"]

    # Stage 1
    interpreted = {}
    for key, val in metric_values.items():
        if key in cfg["metrics"]:
            interpreted[key] = interpret_metric(key, val, cfg)

    # Stage 2 (only categories this task requires)
    categories = []
    for cat in required:
        cat_metrics = [m for m in interpreted.values() if m.category == cat]
        if cat_metrics:
            categories.append(aggregate_category(cat, cat_metrics))

    # Stage 3
    overall, fired, reason, hard_id, hard_reason = apply_policy(task_key, categories, cfg)

    # Stage 4
    rec = map_recommendation(overall, hard_id, cfg)

    # Suggested actions: from the category(ies) that drove the outcome
    actions = []
    cmap = _cat_map(categories)
    action_lib = cfg["policy"].get("suggested_actions", {})
    driver_cats = []
    if hard_id:  # rejection driver
        for cat in required:
            if cat in cmap and cmap[cat].confidence == "low":
                driver_cats.append(cat)
    if not driver_cats:  # cap / weakest driver
        for cat in required:
            if cat in cmap and cmap[cat].confidence != "high":
                driver_cats.append(cat)
    for cat in driver_cats:
        for a in action_lib.get(cat, []):
            actions.append(f"[{cat}] {a}")

    full_reason = reason
    if hard_reason:
        full_reason = f"{hard_reason} (Overall confidence rule: {reason})"

    return Decision(
        task=task_key,
        task_display=task_spec.get("display_name", task_key),
        risk_tier=task_spec["risk_tier"],
        categories=categories,
        overall_confidence=overall,
        recommendation=rec,
        fired_rule_id=fired,
        reason=full_reason,
        hard_flag=hard_id,
        suggested_actions=actions,
        config_version=cfg.get("config_version", ""),
    )
