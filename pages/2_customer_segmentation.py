import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
)

OUTPUT_DIR = "outputs"


@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_segment_files():
    val_df = load_csv(
        os.path.join(
            OUTPUT_DIR,
            "eval_validation_apriori_train_rules_by_segment.csv",
        )
    )
    test_df = load_csv(
        os.path.join(
            OUTPUT_DIR,
            "eval_test_apriori_train_rules_by_segment.csv",
        )
    )
    return val_df, test_df


def format_pct(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (out[c] * 100).round(2).astype(str) + "%"
    return out


st.title("👥 Customer Segmentation")

val_df, test_df = load_segment_files()

if not val_df.empty:
    val_df["dataset"] = "validation"
if not test_df.empty:
    test_df["dataset"] = "test_final"

if val_df.empty and test_df.empty:
    st.warning("Segment files not found in outputs/")
    st.stop()

combined = pd.concat(
    [x for x in [val_df, test_df] if not x.empty],
    ignore_index=True,
)

st.subheader("Segment KPIs (Final Test)")
kpi_df = test_df.copy() if not test_df.empty else val_df.copy()

order_map = {"rare": 0, "frequent": 1, "heavy": 2}
if "segment" in kpi_df.columns:
    kpi_df["segment_order"] = kpi_df["segment"].map(order_map).fillna(999)
    kpi_df = kpi_df.sort_values("segment_order")
    kpi_df = kpi_df.drop(columns=["segment_order"])

cols = st.columns(3)
for i, seg in enumerate(["rare", "frequent", "heavy"]):
    row = kpi_df[kpi_df["segment"] == seg]
    with cols[i]:
        if row.empty:
            st.metric(f"{seg.title()} hit-rate@10", "N/A")
        else:
            hr = row.iloc[0]["hit_rate_at_k"]
            nb = row.iloc[0]["n_baskets"]
            st.metric(
                f"{seg.title()} hit-rate@10",
                f"{hr * 100:.2f}%",
                help=f"Baskets: {int(nb):,}",
            )

st.divider()

st.subheader("Detailed Segment Metrics")
show_cols = [
    "dataset",
    "segment",
    "n_baskets",
    "mean_precision_at_k",
    "mean_recall_at_k",
    "hit_rate_at_k",
]
show_cols = [c for c in show_cols if c in combined.columns]

table_df = format_pct(
    combined[show_cols],
    ["mean_precision_at_k", "mean_recall_at_k", "hit_rate_at_k"],
)
st.dataframe(table_df, use_container_width=True)

st.divider()

st.subheader("Chart")
metric = st.selectbox(
    "Metric",
    ["hit_rate_at_k", "mean_recall_at_k", "mean_precision_at_k", "n_baskets"],
)

if all(c in combined.columns for c in ["segment", "dataset", metric]):
    chart_df = combined[["segment", "dataset", metric]].copy()
    pivot_df = chart_df.pivot(
        index="segment",
        columns="dataset",
        values=metric,
    )
    pivot_df["segment_order"] = [
        order_map.get(x, 999) for x in pivot_df.index
    ]
    pivot_df = pivot_df.sort_values("segment_order")
    pivot_df = pivot_df.drop(columns=["segment_order"])
    st.bar_chart(pivot_df)
