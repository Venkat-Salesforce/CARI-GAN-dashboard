"""
CARI-GAN Evaluation Dashboard
Compares CTGAN | TabDDPM | CARI-GAN (vs Real Baseline where available)

Run with:
    streamlit run dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CARI-GAN Dashboard",
    layout="wide",
    page_icon="🧬",
)

st.markdown(
    """
    <style>
        .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        h1, h2, h3 {color: #1f2937;}
        [data-testid="stMetricValue"] {font-size: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_COLORS = {
    "Real (Baseline)": "#2196F3",
    "CTGAN": "#FF9800",
    "TabDDPM": "#4CAF50",
    "CARI-GAN": "#9C27B0",
}
MODEL_ORDER = ["Real (Baseline)", "CTGAN", "TabDDPM", "CARI-GAN"]


# ---------------------------------------------------------------------------
# CSV LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


ics_df         = load_csv("eval_ics.csv")
fid_df         = load_csv("eval_fidelity_kl_ks.csv")
fid_corr_df    = load_csv("eval_fidelity_correlation.csv")
ri_df          = load_csv("eval_referential_integrity.csv")
val_df         = load_csv("eval_validation_compliance.csv")
val_avg_df     = load_csv("eval_validation_avg.csv")
priv_df        = load_csv("eval_privacy_mia.csv")
eff_df         = load_csv("eval_compute.csv")
eff_totals_df  = load_csv("eval_compute_totals.csv")
lc_df          = load_csv("eval_lifecycle_consistency.csv")
sig_df         = load_csv("eval_statistical_significance.csv")


# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------
st.title("🧬 CARI-GAN: Evaluation Dashboard")
st.markdown("**Comparing CTGAN | TabDDPM | CARI-GAN vs Real Baseline**")
st.divider()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

all_models_seen = set()
for df in [ics_df, fid_df, ri_df, val_df, priv_df, eff_df]:
    if df is not None and "Model" in df.columns:
        all_models_seen.update(df["Model"].unique())

ordered_models = [m for m in MODEL_ORDER if m in all_models_seen] + \
                 [m for m in all_models_seen if m not in MODEL_ORDER]

selected_models = st.sidebar.multiselect(
    "Filter models",
    options=ordered_models,
    default=ordered_models,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sections**")
st.sidebar.markdown(
    "1. ICS Score\n"
    "2. Statistical Fidelity\n"
    "3. Referential Integrity\n"
    "4. Validation Compliance\n"
    "5. Privacy Risk\n"
    "6. Computational Efficiency\n"
    "7. Raw Data Tables"
)


def filter_models(df):
    if df is None or "Model" not in df.columns:
        return df
    return df[df["Model"].isin(selected_models)]


def missing(name):
    st.warning(f"⚠️ `{name}` not found in this folder.")


# ===========================================================================
# SECTION 1 — ICS SCORE (MAIN KPI)
# ===========================================================================
st.header("📊 Section 1: Integrity Confidence Score (ICS)")
st.markdown("`ICS = 0.35×RI + 0.30×VAL + 0.20×LC + 0.15×FID`")

if ics_df is not None:
    df = filter_models(ics_df)
    if not df.empty:
        cols = st.columns(len(df))
        for col, (_, row) in zip(cols, df.iterrows()):
            col.metric(label=row["Model"], value=f"{row['ICS']:.1f} / 100")

        fig = px.bar(
            df,
            x="Model", y="ICS",
            color="Model",
            text="ICS",
            title="ICS Score by Model",
            color_discrete_map=MODEL_COLORS,
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show ICS sub-component breakdown"):
            sub_cols = ["RI_Score", "VAL_Score", "LC_Score", "FID_Score"]
            available = [c for c in sub_cols if c in df.columns]
            if available:
                long_df = df.melt(
                    id_vars=["Model"],
                    value_vars=available,
                    var_name="Component",
                    value_name="Score",
                )
                fig2 = px.bar(
                    long_df,
                    x="Component", y="Score",
                    color="Model",
                    barmode="group",
                    text="Score",
                    title="ICS Components by Model",
                    color_discrete_map=MODEL_COLORS,
                )
                fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig2.update_layout(yaxis_range=[0, 110])
                st.plotly_chart(fig2, use_container_width=True)
else:
    missing("eval_ics.csv")

st.divider()


# ===========================================================================
# SECTION 2 — STATISTICAL FIDELITY (KL / KS)
# ===========================================================================
st.header("📈 Section 2: Statistical Fidelity")

if fid_df is not None:
    df = filter_models(fid_df)
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                df, x="Model", y="KL_div",
                color="Model", points="all",
                title="KL-Divergence by Model (Lower = Better)",
                color_discrete_map=MODEL_COLORS,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(
                df, x="Model", y="KS_stat",
                color="Model", points="all",
                title="KS Statistic by Model (Lower = Better)",
                color_discrete_map=MODEL_COLORS,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        mean_kl = df.groupby("Model")["KL_div"].mean().reset_index()
        fig = px.bar(
            mean_kl,
            x="Model", y="KL_div",
            color="Model", text="KL_div",
            title="Mean KL-Divergence Across All Columns (Lower = Better)",
            color_discrete_map=MODEL_COLORS,
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
else:
    missing("eval_fidelity_kl_ks.csv")

if fid_corr_df is not None:
    df = filter_models(fid_corr_df)
    if not df.empty:
        with st.expander("Show Correlation Fidelity (Frobenius Norm)"):
            corr_summary = df.groupby("Model")["Corr_Frobenius_Norm"].mean().reset_index()
            fig = px.bar(
                corr_summary,
                x="Model", y="Corr_Frobenius_Norm",
                color="Model", text="Corr_Frobenius_Norm",
                title="Mean Correlation Frobenius Norm (Lower = Better)",
                color_discrete_map=MODEL_COLORS,
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.divider()


# ===========================================================================
# SECTION 3 — REFERENTIAL INTEGRITY
# ===========================================================================
st.header("🔗 Section 3: Referential Integrity (FK Violations)")

if ri_df is not None:
    df = filter_models(ri_df)
    if not df.empty:
        ri_summary = df.groupby("Model")["FK_Violation_Pct"].mean().reset_index()
        ri_summary.columns = ["Model", "Avg_FK_Violation_Pct"]

        col1, col2 = st.columns([1, 1])
        with col1:
            fig = px.bar(
                ri_summary,
                x="Model", y="Avg_FK_Violation_Pct",
                color="Model", text="Avg_FK_Violation_Pct",
                title="Average FK Violation % (Lower = Better)",
                color_discrete_map=MODEL_COLORS,
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                df,
                x="Child_Table", y="FK_Violation_Pct",
                color="Model", barmode="group",
                title="FK Violation % by Child Table",
                color_discrete_map=MODEL_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    missing("eval_referential_integrity.csv")

st.divider()


# ===========================================================================
# SECTION 4 — VALIDATION COMPLIANCE
# ===========================================================================
st.header("✅ Section 4: Validation Rule Compliance")

if val_df is not None:
    df = filter_models(val_df)
    if not df.empty:
        val_summary = df.groupby("Model")["Compliance_Pct"].mean().reset_index()

        fig = px.bar(
            val_summary,
            x="Model", y="Compliance_Pct",
            color="Model", text="Compliance_Pct",
            title="Average Validation Compliance % (Higher = Better)",
            color_discrete_map=MODEL_COLORS,
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show per-rule breakdown"):
            pivot = df.pivot_table(
                index=["Table", "Rule"],
                columns="Model",
                values="Compliance_Pct",
                aggfunc="mean",
            ).reset_index()
            st.dataframe(pivot, use_container_width=True)
else:
    missing("eval_validation_compliance.csv")

st.divider()


# ===========================================================================
# SECTION 5 — PRIVACY (MIA AUC)
# ===========================================================================
st.header("🔒 Section 5: Privacy Risk (Membership Inference)")
st.markdown(
    "**MIA_AUC** = how well an attacker can tell if a record was in training data.  \n"
    "_AUC close to **0.5** = strong privacy. AUC close to **1.0** = severe leakage._"
)

if priv_df is not None:
    df = filter_models(priv_df)
    if not df.empty:
        st.dataframe(df, use_container_width=True)

        priv_summary = df.groupby("Model")["MIA_AUC"].mean().reset_index()
        fig = px.bar(
            priv_summary,
            x="Model", y="MIA_AUC",
            color="Model", text="MIA_AUC",
            title="Mean MIA AUC by Model (Closer to 0.5 = Safer)",
            color_discrete_map=MODEL_COLORS,
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.add_hline(
            y=0.5, line_dash="dash", line_color="green",
            annotation_text="Random Guess (0.5) — Ideal",
            annotation_position="bottom right",
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            df,
            x="Table", y="MIA_AUC",
            color="Model", barmode="group",
            title="MIA AUC by Table & Model",
            color_discrete_map=MODEL_COLORS,
        )
        fig2.add_hline(y=0.5, line_dash="dash", line_color="green")
        fig2.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig2, use_container_width=True)
else:
    missing("eval_privacy_mia.csv")

st.divider()


# ===========================================================================
# SECTION 6 — COMPUTATIONAL EFFICIENCY
# ===========================================================================
st.header("⚡ Section 6: Computational Efficiency")

if eff_df is not None:
    df = filter_models(eff_df)
    if not df.empty:
        totals = df.groupby("Model").agg(
            Train_Sec_Total=("Train_Sec", "sum"),
            Peak_Mem_MB_Max=("Peak_Mem_MB", "max"),
        ).reset_index()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                totals,
                x="Model", y="Train_Sec_Total",
                color="Model", text="Train_Sec_Total",
                title="Total Training Time (seconds)",
                color_discrete_map=MODEL_COLORS,
            )
            fig.update_traces(texttemplate="%{text:.0f}s", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                totals,
                x="Model", y="Peak_Mem_MB_Max",
                color="Model", text="Peak_Mem_MB_Max",
                title="Peak Memory Usage (MB)",
                color_discrete_map=MODEL_COLORS,
            )
            fig.update_traces(texttemplate="%{text:.1f} MB", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show per-table training time breakdown"):
            fig = px.bar(
                df,
                x="Table", y="Train_Sec",
                color="Model", barmode="group",
                title="Training Time per Table",
                color_discrete_map=MODEL_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    missing("eval_compute.csv")

st.divider()


# ===========================================================================
# RAW DATA TABLES
# ===========================================================================
st.header("📋 Raw Evaluation Data")

table_map = {
    "ICS Table":                ics_df,
    "Fidelity (KL / KS)":       fid_df,
    "Fidelity (Correlation)":   fid_corr_df,
    "Referential Integrity":    ri_df,
    "Validation Compliance":    val_df,
    "Validation Averages":      val_avg_df,
    "Privacy (MIA)":            priv_df,
    "Compute":                  eff_df,
    "Compute Totals":           eff_totals_df,
    "Lifecycle Consistency":    lc_df,
    "Statistical Significance": sig_df,
}

for label, df in table_map.items():
    if df is not None:
        with st.expander(f"Show {label}"):
            st.dataframe(df, use_container_width=True)


st.divider()
st.caption(
    "CARI-GAN Evaluation Dashboard · Built with Streamlit + Plotly · "
    "Data sourced from notebook cells 47–72."
)
