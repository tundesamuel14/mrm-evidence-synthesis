"""
logger.py
---------
Stage 5: the Decision Logger.

Does nothing for today's recommendation. It exists so the FUTURE ML phase has
the right raw material. The label we ultimately want is PRODUCTION OUTCOME
(did the deployed model drift / fail / trigger an audit finding?), NOT the past
MRM approve/reject decision. Training on the latter only teaches a model to
imitate the committee, copying its mistakes.

So every record stores the full decision now, plus a set of BLANK
outcome fields to be filled in months later once production data exists.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "decision_log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_decision(decision, metric_values: dict, dataset_name: str) -> str:
    """Persist one decision as a JSON record. Returns the record id."""
    record_id = str(uuid.uuid4())[:8]

    record = {
        "record_id": record_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_version": decision.config_version,

        # --- the decision, fully expanded (recorded now) ---
        "dataset": dataset_name,
        "task": decision.task,
        "risk_tier": decision.risk_tier,
        "metric_values": metric_values,
        "categories": [
            {
                "category": c.category,
                "confidence": c.confidence,
                "reason": c.reason,
                "metrics": [
                    {
                        "key": m.key,
                        "value": m.value,
                        "strength": m.strength,
                        "passed": m.passed,
                    }
                    for m in c.metrics
                ],
            }
            for c in decision.categories
        ],
        "overall_confidence": decision.overall_confidence,
        "fired_rule_id": decision.fired_rule_id,
        "hard_flag": decision.hard_flag,
        "reason": decision.reason,
        "recommendation": decision.recommendation,

        # --- outcome slots (LEFT BLANK -- filled months later) ---
        "production_outcome": {
            "deployment_date": None,
            "incident_observed": None,
            "drift_detected": None,
            "audit_finding": None,
            "retraining_triggered": None,
            "production_success_label": None,   # <- the real ML target, later
        },
    }

    path = LOG_DIR / f"decision_{record_id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    return record_id
