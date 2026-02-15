# Before start make sure to install streamlit with pip install streamlit

# install library
import streamlit as st

st.set_page_config(page_title="Instacart Recommender", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🛒 Instacart Product Recommender")
st.write(
    """
    This web app recommends grocery products using **association rules**
    learned with the **Apriori** algorithm.

    The goal is to suggest products that are often purchased together.
    """
)

st.markdown("---")

# -----------------------------
# Inputs
# -----------------------------
st.header("1) Inputs")

st.write(
    """
    - Basket items (selected by the user)
    - Number of recommendations (Top-K)

    **Example input:**
    - Basket items: `[bananas, yogurt, milk]`
    - K = 10
    """
)

st.markdown("")

st.markdown("---")

# -----------------------------
# Recommendations
# -----------------------------
st.header("2) Recommendations")

st.write(
    """
    **Example output (format):**
    - `strawberries` (confidence = 0.25, lift = 1.12)
    - `granola` (confidence = 0.22, lift = 1.08)
    """
)

# (empty space reserved for results)
st.markdown("")

st.markdown("---")

# -----------------------------
# Notes / Context
# -----------------------------
st.header("3) Notes (Project context)")

st.write(
    """
    **Model:**
    - Apriori frequent itemsets (max_len = 2)
    - Simple association rules (1 → 1)

    **Data:**
    - Instacart Kaggle dataset
    - Rules learned from PRIOR and evaluated on TRAIN

    **Limitations:**
    - Rule coverage is limited, so some baskets may produce few or no
        recommendations.
    """
)
