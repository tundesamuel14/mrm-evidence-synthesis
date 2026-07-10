"""
config_loader.py
----------------
Loads the three policy configs and computes a version hash so every logged
decision can be tied to the exact rule set that produced it (reproducibility).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_configs() -> dict:
    """Return all configs plus a short version hash of their combined content."""
    task_path = CONFIG_DIR / "task_templates.yaml"
    metric_path = CONFIG_DIR / "metric_thresholds.yaml"
    policy_path = CONFIG_DIR / "policy_rules.yaml"

    raw_blob = b""
    for p in (task_path, metric_path, policy_path):
        raw_blob += p.read_bytes()
    version = hashlib.sha256(raw_blob).hexdigest()[:10]

    return {
        "tasks": _read(task_path),
        "metrics": _read(metric_path),
        "policy": _read(policy_path),
        "config_version": version,
    }
