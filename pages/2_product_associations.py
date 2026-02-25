import ast
import pandas as pd
import streamlit as st


@st.cache_data
def load_association_page_data():
    """
    Load Apriori outputs for the associations page.
    """
    baskets_train = pd.read_csv(
        "outputs/baskets_train_rules_ready.csv",
        usecols=["order_id", "user_id", "segment", "items"],
    )
    selected_products = pd.read_csv(
        "outputs/apriori_train_rules_selected_products.csv"
    )
    dep_quota = pd.read_csv(
        "outputs/apriori_train_rules_department_quotas.csv")
    dep_report = pd.read_csv(
        "outputs/apriori_train_rules_department_distribution_before_after.csv"
    )
    itemsets = pd.read_csv("outputs/apriori_train_rules_frequent_itemsets.csv")
    rules_reco = pd.read_csv("outputs/apriori_train_rules_rules_reco.csv")
    return (
        baskets_train,
        selected_products,
        dep_quota,
        dep_report,
        itemsets,
        rules_reco,
    )


@st.cache_data
def parse_items_cached(df):
    """
    Convert items column from string to list.
    """
    out = df.copy()
    out["items"] = out["items"].apply(ast.literal_eval)
    return out


# Page title
st.title("2) Product Associations")

# Load cached data
(
    baskets_train,
    selected_products,
    dep_quota,
    dep_report,
    itemsets,
    rules_reco,
) = load_association_page_data()

# Parse baskets once with cache
baskets_train = parse_items_cached(baskets_train)

# Page introduction
st.write(
    "This page shows frequent bundles and co-purchases found with Apriori")

# Basket summary
st.subheader("Baskets Used for Apriori")

basket_summary = pd.DataFrame(
    [
        {
            "n_baskets": len(baskets_train),
            "n_users": baskets_train["user_id"].nunique(),
            "avg_basket_size": baskets_train["items"].apply(len).mean(),
            "median_basket_size": baskets_train["items"].apply(len).median(),
        }
    ]
)

st.dataframe(basket_summary, width="stretch")

# Department quotas
st.subheader("Selected Products by department")
st.dataframe(dep_quota, width="stretch")

# Department distribution check
st.subheader("Department Distribution Before vs After Selection")
st.dataframe(dep_report, width="stretch")

# Selected products sample
st.subheader("Selected products")
cols_selected = [
    c
    for c in [
        "product_id",
        "product_name",
        "department",
        "aisle",
        "n_occurrences",
        "rank_in_department",
    ]
    if c in selected_products.columns
]
st.dataframe(selected_products[cols_selected].head(50), width="stretch")

# Frequent itemsets summary
st.subheader("Frequent Itemsets")

if "itemset_length" in itemsets.columns:
    itemset_counts = (
        itemsets["itemset_length"].value_counts().sort_index().reset_index()
    )
    itemset_counts.columns = ["itemset_length", "n_itemsets"]
    st.dataframe(itemset_counts, width="stretch")

itemset_cols = [
    c
    for c in ["support", "itemset_length", "itemsets", "itemsets_str"]
    if c in itemsets.columns
]
st.dataframe(itemsets[itemset_cols].head(30), width="stretch")

# Rules summary
st.subheader("Association Rules")

rules_summary = pd.DataFrame(
    [
        {
            "n_rules": len(rules_reco),
            "avg_confidence": rules_reco["confidence"].mean(),
            "avg_lift": rules_reco["lift"].mean(),
            "avg_support": rules_reco["support"].mean(),
        }
    ]
)

st.dataframe(rules_summary, width="stretch")

# Rule types
if "rule_type" in rules_reco.columns:
    st.subheader("Type of Bundles")
    rule_type_counts = rules_reco["rule_type"].value_counts().reset_index()
    rule_type_counts.columns = ["rule_type", "n_rules"]
    st.dataframe(rule_type_counts, width="stretch")

# Top rules
st.subheader("Top Rules")

rule_cols = [
    c
    for c in [
        "antecedents_str",
        "consequents_str",
        "support",
        "confidence",
        "lift",
        "rule_type",
    ]
    if c in rules_reco.columns
]

st.dataframe(rules_reco[rule_cols].head(30), width="stretch")
