"""
app.py
======
MRM Evidence Synthesis Framework -- MVP web application.

Workflow:
  1. Load real + synthetic datasets (upload or pick a bundled sample)
  2. Select the downstream task
  3. Run real validation metrics
  4. See evidence organised by category, with per-category confidence
  5. See risk-aware overall confidence + recommendation (reason + action)
  6. Download the auditable MRM report

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_configs
from engine.metrics import compute_metrics
from engine.decision_logic import run_decision
from engine.report import build_report
from engine.logger import log_decision

SAMPLES = ROOT / "data" / "samples"

# --------------------------------------------------------------------------- #
# Page setup + styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="MRM Evidence Synthesis",
    page_icon="assets/favicon.png" if (ROOT / "assets/favicon.png").exists() else "◆",
    layout="wide",
)

st.markdown(
    """
    <style>
      /* ---- institutional type + palette ---- */
      html, body, [class*="css"] { font-family: 'Inter', 'Helvetica Neue', sans-serif; }
      .block-container { padding-top: 2.2rem; max-width: 1150px; }

      .masthead {
        border-left: 5px solid #0B3D6B;
        padding: 0.1rem 0 0.1rem 1rem; margin-bottom: 0.2rem;
      }
      .masthead h1 { color:#0B3D6B; font-size:1.7rem; font-weight:700; margin:0; letter-spacing:-0.01em;}
      .masthead p  { color:#6B7280; font-size:0.95rem; margin:0.15rem 0 0 0; }

      .stepbar { color:#6B7280; font-size:0.8rem; text-transform:uppercase;
                 letter-spacing:0.08em; margin:1.4rem 0 0.4rem 0; font-weight:600; }

      .conf-pill { display:inline-block; padding:0.15rem 0.7rem; border-radius:999px;
                   font-size:0.8rem; font-weight:700; }
      .conf-high   { background:#E4F2E7; color:#2E7D32; }
      .conf-medium { background:#FBEEDC; color:#C77700; }
      .conf-low    { background:#F8E1DE; color:#C0392B; }

      .rec-card { border-radius:8px; padding:1.1rem 1.3rem; margin:0.4rem 0 0.6rem 0;
                  border:1px solid #D9DEE5; }
      .rec-approve     { background:#F0F8F1; border-color:#2E7D32; }
      .rec-conditional { background:#FDF6EC; border-color:#C77700; }
      .rec-reject      { background:#FBEDEB; border-color:#C0392B; }
      .rec-card h2 { margin:0; font-size:1.3rem; }
      .rec-card p  { margin:0.3rem 0 0 0; color:#333; font-size:0.92rem; }

      .cat-box { border:1px solid #D9DEE5; border-radius:8px; padding:0.9rem 1.1rem;
                 background:#FAFBFC; height:100%; }
      .cat-box h4 { margin:0 0 0.4rem 0; font-size:0.95rem; color:#0B3D6B; }
      .metric-row { font-size:0.82rem; color:#333; padding:0.12rem 0;
                    display:flex; justify-content:space-between; }
      .m-strong { color:#2E7D32; font-weight:600; }
      .m-moderate { color:#C77700; font-weight:600; }
      .m-weak { color:#C0392B; font-weight:600; }

      .subtle { color:#6B7280; font-size:0.85rem; }
      div[data-testid="stMetricValue"] { font-size:1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

CONF_CLASS = {"high": "conf-high", "medium": "conf-medium", "low": "conf-low"}
CONF_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}
STRENGTH_CLASS = {"strong": "m-strong", "moderate": "m-moderate", "weak": "m-weak"}
REC_CLASS = {
    "Approve": "rec-approve",
    "Conditional Approval": "rec-conditional",
    "Reject": "rec-reject",
}

cfg = load_configs()


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="masthead">
      <h1>MRM Evidence Synthesis Framework</h1>
      <p>Turn synthetic-data validation metrics into an auditable deployment recommendation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='subtle'>Deterministic MVP &nbsp;·&nbsp; config version "
    f"<code>{cfg['config_version']}</code> &nbsp;·&nbsp; "
    f"human validators remain the final decision-makers</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Sidebar: data + task selection
# --------------------------------------------------------------------------- #

SAMPLE_LABELS = {
    "— none —": None,
    "High-fidelity synthetic (expect: Approve)": "synthetic_high_fidelity.csv",
    "Fairness-degraded synthetic (expect: Conditional)": "synthetic_fairness_degraded.csv",
    "Privacy-leaky synthetic (expect: Reject)": "synthetic_privacy_leaky.csv",
    "Low-utility synthetic (expect: Reject)": "synthetic_low_utility.csv",
}

with st.sidebar:
    st.markdown("### 1 · Data")
    data_mode = st.radio(
        "Choose data source",
        ["Use bundled sample", "Upload my own"],
        label_visibility="collapsed",
    )

    real_df = synth_df = None
    dataset_name = "—"

    if data_mode == "Use bundled sample":
        real_path = SAMPLES / "real_german_credit.csv"
        if real_path.exists():
            real_df = pd.read_csv(real_path)
        pick = st.selectbox("Synthetic dataset", list(SAMPLE_LABELS.keys()), index=1)
        fname = SAMPLE_LABELS[pick]
        if fname and (SAMPLES / fname).exists():
            synth_df = pd.read_csv(SAMPLES / fname)
            dataset_name = fname.replace(".csv", "")
        st.caption("Real base: German-Credit-style consumer-credit data.")
    else:
        up_real = st.file_uploader("Real dataset (CSV)", type="csv", key="real")
        up_synth = st.file_uploader("Synthetic dataset (CSV)", type="csv", key="synth")
        if up_real is not None:
            real_df = pd.read_csv(up_real)
        if up_synth is not None:
            synth_df = pd.read_csv(up_synth)
            dataset_name = up_synth.name.replace(".csv", "")

    st.markdown("### 2 · Task")
    task_options = {v["display_name"]: k for k, v in cfg["tasks"].items()}
    task_display = st.selectbox("Downstream task", list(task_options.keys()))
    task_key = task_options[task_display]
    task_spec = cfg["tasks"][task_key]

    tier = task_spec["risk_tier"]
    st.markdown(
        f"**Risk tier:** <span class='conf-pill {CONF_CLASS.get(tier,'conf-medium')}'>"
        f"{tier.title()}</span>",
        unsafe_allow_html=True,
    )
    st.caption("Required evidence: " +
               ", ".join(c.replace("_", " ").title()
                         for c in task_spec["required_categories"]))

    # target + sensitive column pickers
    st.markdown("### 3 · Columns")
    if real_df is not None:
        cols = list(real_df.columns)
        default_t = cols.index("target") if "target" in cols else len(cols) - 1
        target_col = st.selectbox("Target column", cols, index=default_t)
        sens_opts = ["— none —"] + [c for c in cols if c != target_col]
        default_s = sens_opts.index("sex") if "sex" in sens_opts else 0
        sensitive_col = st.selectbox("Sensitive attribute (fairness)", sens_opts, index=default_s)
        sensitive_col = None if sensitive_col == "— none —" else sensitive_col
    else:
        target_col = sensitive_col = None

    run = st.button("Run validation", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #

if real_df is not None and synth_df is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='stepbar'>Real dataset</div>", unsafe_allow_html=True)
        st.dataframe(real_df.head(6), use_container_width=True, height=230)
        st.caption(f"{len(real_df):,} rows · {real_df.shape[1]} columns")
    with c2:
        st.markdown("<div class='stepbar'>Synthetic dataset</div>", unsafe_allow_html=True)
        st.dataframe(synth_df.head(6), use_container_width=True, height=230)
        st.caption(f"{len(synth_df):,} rows · {synth_df.shape[1]} columns")
else:
    st.info("Pick or upload a real and a synthetic dataset in the sidebar, choose a task, "
            "then **Run validation**.")

# --------------------------------------------------------------------------- #
# Run the pipeline
# --------------------------------------------------------------------------- #

if run:
    if real_df is None or synth_df is None:
        st.error("Both a real and a synthetic dataset are required.")
        st.stop()
    if target_col not in real_df.columns or target_col not in synth_df.columns:
        st.error(f"Target column '{target_col}' must be present in both datasets.")
        st.stop()

    required = task_spec["required_categories"]
    with st.spinner("Computing validation metrics (training models on your data)…"):
        try:
            metric_values = compute_metrics(
                real_df, synth_df, target=target_col,
                sensitive=sensitive_col, categories=required,
            )
            decision = run_decision(task_key, metric_values, cfg)
            record_id = log_decision(decision, metric_values, dataset_name)
        except Exception as e:  # keep the demo resilient
            st.error(f"Could not complete validation: {e}")
            st.stop()

    # ---- Recommendation banner ----
    st.markdown("<div class='stepbar'>Deployment recommendation</div>", unsafe_allow_html=True)
    rec = decision.recommendation
    conf = decision.overall_confidence
    st.markdown(
        f"""
        <div class="rec-card {REC_CLASS.get(rec,'')}">
          <h2>{rec}
            <span class="conf-pill {CONF_CLASS[conf]}" style="vertical-align:middle; margin-left:0.5rem;">
              Overall confidence: {CONF_LABEL[conf]}</span>
          </h2>
          <p><strong>Reason.</strong> {decision.reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Evidence by category ----
    st.markdown("<div class='stepbar'>Evidence by category</div>", unsafe_allow_html=True)
    cols = st.columns(len(decision.categories))
    for col, cat in zip(cols, decision.categories):
        rows_html = ""
        for m in cat.metrics:
            rows_html += (
                f"<div class='metric-row'><span>{m.display_name}</span>"
                f"<span class='{STRENGTH_CLASS[m.strength]}'>{m.value}</span></div>"
            )
        with col:
            st.markdown(
                f"""
                <div class="cat-box">
                  <h4>{cat.category.replace('_',' ').title()}
                    <span class="conf-pill {CONF_CLASS[cat.confidence]}" style="float:right;">
                      {CONF_LABEL[cat.confidence]}</span>
                  </h4>
                  {rows_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption("Colour = evidence strength per metric: "
               "green strong · amber moderate · red weak. Category confidence is "
               "interpreted (graduated demotion), never averaged.")

    # ---- Suggested actions ----
    if decision.suggested_actions:
        st.markdown("<div class='stepbar'>Suggested remediation</div>", unsafe_allow_html=True)
        for a in decision.suggested_actions:
            st.markdown(f"- {a}")

    # ---- Policy trail + report ----
    st.markdown("<div class='stepbar'>Audit trail</div>", unsafe_allow_html=True)
    tcol1, tcol2, tcol3 = st.columns(3)
    tcol1.metric("Rule fired", decision.fired_rule_id or "—")
    tcol2.metric("Hard flag", decision.hard_flag or "none")
    tcol3.metric("Record ID", record_id)

    report_md = build_report(decision, dataset_name, record_id)
    with st.expander("View full MRM report (Markdown)"):
        st.markdown(report_md)
    st.download_button(
        "Download MRM report (.md)",
        data=report_md,
        file_name=f"MRM_report_{record_id}.md",
        mime="text/markdown",
        type="primary",
    )
    st.caption("This recommendation is decision support. A human MRM validator "
               "remains the final decision-maker.")
