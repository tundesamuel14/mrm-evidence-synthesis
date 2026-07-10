"""
report.py
---------
Generates the auditable MRM report (Stage 9 of the workflow) as Markdown.
Readable by an auditor; every metric shows value, threshold, and status, and
every recommendation carries its reason plus suggested actions.
"""

from __future__ import annotations

from datetime import datetime, timezone

_CONF_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}
_STRENGTH_LABEL = {"strong": "Strong", "moderate": "Moderate", "weak": "Weak"}


def _threshold_str(m) -> str:
    op = ">=" if m.direction == "higher_better" else "<="
    return f"{op} {m.strong_at} strong / {op} {m.weak_below} moderate"


def build_report(decision, dataset_name: str, record_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines.append(f"# Model Risk Management — Synthetic Data Validation Report")
    lines.append("")
    lines.append(f"**Record ID:** `{record_id}`  ")
    lines.append(f"**Generated:** {ts}  ")
    lines.append(f"**Config version:** `{decision.config_version}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary block
    lines.append("## Decision Summary")
    lines.append("")
    lines.append(f"- **Dataset:** {dataset_name}")
    lines.append(f"- **Downstream task:** {decision.task_display}")
    lines.append(f"- **Risk tier:** {decision.risk_tier.title()}")
    lines.append(f"- **Overall confidence:** {_CONF_LABEL[decision.overall_confidence]}")
    lines.append(f"- **Recommendation:** **{decision.recommendation}**")
    lines.append("")
    lines.append(f"> **Reason.** {decision.reason}")
    lines.append("")

    # Evidence categories
    lines.append("## Evidence by Category")
    lines.append("")
    for c in decision.categories:
        lines.append(f"### {c.category.replace('_', ' ').title()} — "
                     f"{_CONF_LABEL[c.confidence]}")
        lines.append(f"*{c.reason}*")
        lines.append("")
        lines.append("| Metric | Value | Threshold | Strength |")
        lines.append("|---|---|---|---|")
        for m in c.metrics:
            lines.append(
                f"| {m.display_name} | {m.value} | {_threshold_str(m)} | "
                f"{_STRENGTH_LABEL[m.strength]} |"
            )
        lines.append("")

    # Rationale trail
    lines.append("## Threshold Rationale (Governance Trail)")
    lines.append("")
    lines.append("Each threshold below is a configurable organizational policy, "
                 "not a regulatory constant. Rationale travels with the decision.")
    lines.append("")
    seen = set()
    for c in decision.categories:
        for m in c.metrics:
            if m.key in seen:
                continue
            seen.add(m.key)
            lines.append(f"- **{m.display_name}:** {m.rationale}")
    lines.append("")

    # Suggested actions
    if decision.suggested_actions:
        lines.append("## Suggested Remediation")
        lines.append("")
        for a in decision.suggested_actions:
            lines.append(f"- {a}")
        lines.append("")

    # Policy trail
    lines.append("## Policy Trail")
    lines.append("")
    lines.append(f"- **Rule fired:** `{decision.fired_rule_id}`")
    if decision.hard_flag:
        lines.append(f"- **Hard flag:** `{decision.hard_flag}` (forces Reject)")
    lines.append(f"- **Overall confidence:** {_CONF_LABEL[decision.overall_confidence]}")
    lines.append(f"- **Recommendation:** {decision.recommendation}")
    lines.append("")
    lines.append("---")
    lines.append("*This recommendation is decision support. A human MRM "
                 "validator remains the final decision-maker.*")

    return "\n".join(lines)
