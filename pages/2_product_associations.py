import ast
import pandas as pd
import streamlit as st


# Page title
st.title("2) Product Associations")


def parse_items_column(df):
    """
    Convert items column from string to list.
    """
    out = df.copy()
    out["items"] = out["items"].apply(ast.literal_eval)
    return out


# Load notebook outputs
baskets_train = pd.read_csv("../outputs/baskets_train_rules_ready.csv")
selected_products = pd.read_csv(
    "../outputs/apriori_train_rules_selected_products.csv")
dep_quota = pd.read_csv("../outputs/apriori_train_rules_department_quotas.csv")
dep_report = pd.read_csv(
    "../outputs/apriori_train_rules_department_distribution_before_after.csv"
)
itemsets = pd.read_csv("../outputs/apriori_train_rules_frequent_itemsets.csv")
rules_reco = pd.read_csv("../outputs/apriori_train_rules_rules_reco.csv")

# Page introduction
st.write(
    "This page shows frequent bundles and co-purchases found with Apriori."
)

# Basket summary used for Apriori
st.subheader("Baskets Used for Apriori")

baskets_train = parse_items_column(baskets_train)

basket_summary = pd.DataFrame([{
    "n_baskets": len(baskets_train),
    "n_users": baskets_train["user_id"].nunique(),
    "avg_basket_size": baskets_train["items"].apply(len).mean(),
    "median_basket_size": baskets_train["items"].apply(len).median(),
}])

st.dataframe(basket_summary, use_container_width=True)

# Product selection by department quotas
st.subheader("Selected Products by Department (quota logic)")
st.dataframe(dep_quota, use_container_width=True)

# Department coverage check before/after selection
st.subheader("Department Distribution Before vs After Selection")
st.dataframe(dep_report, use_container_width=True)

# Selected products sample
st.subheader("Selected Products (sample)")
cols_selected = [
    c for c in [
        "product_id",
        "product_name",
        "department",
        "aisle",
        "n_occurrences",
        "rank_in_department",
    ]
    if c in selected_products.columns
]
st.dataframe(
    selected_products[cols_selected].head(50),
    use_container_width=True
)

# Frequent itemsets summary
st.subheader("Frequent Itemsets")

if "itemset_length" in itemsets.columns:
    itemset_counts = (
        itemsets["itemset_length"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    itemset_counts.columns = ["itemset_length", "n_itemsets"]
    st.dataframe(itemset_counts, use_container_width=True)

itemset_cols = [
    c for c in ["support", "itemset_length", "itemsets", "itemsets_str"]
    if c in itemsets.columns
]
st.dataframe(itemsets[itemset_cols].head(30), use_container_width=True)

# Association rules summary
st.subheader("Association Rules (co-purchases)")

rules_summary = pd.DataFrame([{
    "n_rules": len(rules_reco),
    "avg_confidence": rules_reco["confidence"].mean(),
    "avg_lift": rules_reco["lift"].mean(),
    "avg_support": rules_reco["support"].mean(),
}])

st.dataframe(rules_summary, use_container_width=True)

# Rule type summary (1->1, 2->1)
if "rule_type" in rules_reco.columns:
    st.subheader("Type of Bundles (Rule Types)")
    rule_type_counts = rules_reco["rule_type"].value_counts().reset_index()
    rule_type_counts.columns = ["rule_type", "n_rules"]
    st.dataframe(rule_type_counts, use_container_width=True)

# Top rules for business reading
st.subheader("Top Rules (sample)")

rule_cols = [
    c for c in [
        "antecedents_str",
        "consequents_str",
        "support",
        "confidence",
        "lift",
        "rule_type",
    ]
    if c in rules_reco.columns
]

st.dataframe(rules_reco[rule_cols].head(30), use_container_width=True)
