import os
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Executive Summary",
    page_icon="📊",
    layout="wide",
)

OUTPUT_DIR = "outputs"
DATA_DIR = "data"


@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_data():
    compare_df = load_csv(
        os.path.join(OUTPUT_DIR, "eval_compare_validation_vs_test.csv")
    )
    val_summary = load_csv(
        os.path.join(
            OUTPUT_DIR,
            "eval_validation_apriori_train_rules_summary.csv",
        )
    )
    test_summary = load_csv(
        os.path.join(
            OUTPUT_DIR,
            "eval_test_apriori_train_rules_summary.csv",
        )
    )
    rules_df = load_csv(
        os.path.join(OUTPUT_DIR, "apriori_train_rules_rules.csv")
    )
    selected_df = load_csv(
        os.path.join(
            OUTPUT_DIR,
            "apriori_train_rules_selected_products.csv",
        )
    )
    products_df = load_csv(os.path.join(DATA_DIR, "products.csv"))
    return (
        compare_df,
        val_summary,
        test_summary,
        rules_df,
        selected_df,
        products_df,
    )


def parse_items(s):
    if pd.isna(s):
        return []
    s = str(s).strip()
    if s == "":
        return []
    return [int(x) for x in s.split("|") if x.strip()]


def product_map(products_df):
    if products_df.empty:
        return {}
    tmp = products_df[["product_id", "product_name"]].copy()
    tmp["product_id"] = tmp["product_id"].astype(int)
    return dict(zip(tmp["product_id"], tmp["product_name"]))


def metric_value(df, col):
    if df.empty or col not in df.columns:
        return np.nan
    return df.iloc[0][col]


def prepare_rules(rules_df, pmap):
    if rules_df.empty:
        return pd.DataFrame()

    df = rules_df.copy()
    df["ante_list"] = df["antecedents"].apply(parse_items)
    df["con_list"] = df["consequents"].apply(parse_items)

    df["antecedent_len"] = df["ante_list"].apply(len)
    df["consequent_len"] = df["con_list"].apply(len)

    df["antecedents_names"] = df["ante_list"].apply(
        lambda x: ", ".join([pmap.get(i, str(i)) for i in x])
    )
    df["consequents_names"] = df["con_list"].apply(
        lambda x: ", ".join([pmap.get(i, str(i)) for i in x])
    )
    return df


st.title("📊 Executive Summary")

(
    compare_df,
    val_summary,
    test_summary,
    rules_df,
    selected_df,
    products_df,
) = load_data()

pmap = product_map(products_df)
rules_show = prepare_rules(rules_df, pmap)

n_selected = len(selected_df) if not selected_df.empty else 0
n_rules = len(rules_df) if not rules_df.empty else 0

if not rules_show.empty:
    n_rules_2to1 = len(
        rules_show[
            (rules_show["antecedent_len"] == 2)
            & (rules_show["consequent_len"] == 1)
        ]
    )
else:
    n_rules_2to1 = 0

test_n = metric_value(test_summary, "n_evaluated_baskets")
test_hit = metric_value(test_summary, "hit_rate_at_k")
test_prec = metric_value(test_summary, "mean_precision_at_k")
test_cov = metric_value(test_summary, "coverage_at_k")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Evaluated baskets (test)",
    f"{int(test_n):,}" if pd.notna(test_n) else "N/A",
)
c2.metric(
    "Hit-rate@10",
    f"{test_hit * 100:.2f}%" if pd.notna(test_hit) else "N/A",
)
c3.metric(
    "Precision@10",
    f"{test_prec * 100:.2f}%" if pd.notna(test_prec) else "N/A",
)
c4.metric(
    "Coverage@10",
    f"{test_cov * 100:.2f}%" if pd.notna(test_cov) else "N/A",
)

c5, c6, c7 = st.columns(3)
c5.metric("Selected products", f"{n_selected:,}")
c6.metric("Rules (all)", f"{n_rules:,}")
c7.metric("Rules 2→1", f"{n_rules_2to1:,}")

st.divider()

st.subheader("Validation vs Test")
if compare_df.empty:
    st.warning("Missing file: outputs/eval_compare_validation_vs_test.csv")
else:
    cols = [
        "dataset",
        "n_evaluated_baskets",
        "n_skipped_baskets",
        "mean_precision_at_k",
        "mean_recall_at_k",
        "hit_rate_at_k",
        "coverage_at_k",
    ]
    cols = [c for c in cols if c in compare_df.columns]
    st.dataframe(compare_df[cols], use_container_width=True)

    if (
        "dataset" in compare_df.columns
        and "hit_rate_at_k" in compare_df.columns
    ):
        chart_df = compare_df[["dataset", "hit_rate_at_k"]].copy()
        chart_df = chart_df.set_index("dataset")
        st.bar_chart(chart_df)

st.divider()

st.subheader("Top 2→1 Rules")
if rules_show.empty:
    st.warning("Missing file: outputs/apriori_train_rules_rules.csv")
else:
    top_rules = rules_show.copy()
    top_rules = top_rules[
        (top_rules["antecedent_len"] == 2)
        & (top_rules["consequent_len"] == 1)
    ]

    if "lift" in top_rules.columns:
        top_rules = top_rules[top_rules["lift"] > 1]

    if all(x in top_rules.columns for x in ["confidence", "lift", "support"]):
        top_rules = top_rules.sort_values(
            ["confidence", "lift", "support"],
            ascending=[False, False, False],
        )

    cols = [
        "antecedents_names",
        "consequents_names",
        "support",
        "confidence",
        "lift",
    ]
    cols = [c for c in cols if c in top_rules.columns]
    st.dataframe(top_rules[cols].head(10), use_container_width=True)
