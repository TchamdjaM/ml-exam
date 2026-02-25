import pandas as pd
import streamlit as st


# Page title
st.title("1) Customer Segmentation")

# Load notebook outputs
customers = pd.read_csv("../outputs/customer_features_with_segments.csv")
orders_train_rules = pd.read_csv("../outputs/orders_train_rules.csv")
orders_validation = pd.read_csv("../outputs/orders_validation.csv")
orders_test_final = pd.read_csv("../outputs/orders_test_final.csv")

# Page introduction
st.write(
    "This page shows customer segments built from prior shopping history."
)

# Customer sample
st.subheader("Customer Features (sample)")
st.dataframe(customers.head(50), use_container_width=True)

# Segment distribution
st.subheader("Type of Buyer (Segment Distribution)")

segment_counts = customers["segment"].value_counts().reset_index()
segment_counts.columns = ["segment", "n_users"]
total_users = segment_counts["n_users"].sum()
segment_counts["share"] = segment_counts["n_users"] / total_users

st.dataframe(segment_counts, use_container_width=True)

# Segment profile summary
st.subheader("Segment Profile")

# Keep only columns available in the file
possible_cols = [
    "n_prior_orders",
    "total_items_prior",
    "avg_basket_size_prior",
    "basket_size_std_prior",
    "avg_reorder_rate",
]

agg_map = {"user_id": "count"}
rename_map = {"user_id": "n_users"}

for col in possible_cols:
    if col in customers.columns:
        agg_map[col] = "mean"

segment_profile = customers.groupby("segment").agg(agg_map).reset_index()
segment_profile = segment_profile.rename(columns=rename_map)

st.dataframe(segment_profile, use_container_width=True)

# Basket size and variability focus
st.subheader("Basket Size and Variability")

basket_cols = []
for col in ["avg_basket_size_prior", "basket_size_std_prior"]:
    if col in customers.columns:
        basket_cols.append(col)

if len(basket_cols) > 0:
    basket_view = customers[["user_id", "segment"] + basket_cols].copy()
    st.dataframe(basket_view.head(50), use_container_width=True)

# Split size summary
st.subheader("Order Split Sizes")

split_sizes = pd.DataFrame([
    {
        "dataset": "train_rules",
        "n_orders": len(orders_train_rules),
        "n_users": orders_train_rules["user_id"].nunique(),
    },
    {
        "dataset": "validation",
        "n_orders": len(orders_validation),
        "n_users": orders_validation["user_id"].nunique(),
    },
    {
        "dataset": "test_final",
        "n_orders": len(orders_test_final),
        "n_users": orders_test_final["user_id"].nunique(),
    },
])

st.dataframe(split_sizes, use_container_width=True)

# Segment distribution in each split
st.subheader("Segment Distribution by Split")

for split_name, df in [
    ("train_rules", orders_train_rules),
    ("validation", orders_validation),
    ("test_final", orders_test_final),
]:
    st.markdown(f"**{split_name}**")

    tmp = df["segment"].value_counts().reset_index()
    tmp.columns = ["segment", "n_orders"]
    tmp["share"] = tmp["n_orders"] / tmp["n_orders"].sum()

    st.dataframe(tmp, use_container_width=True)
