import streamlit as st

st.set_page_config(
    page_title="Retail Customer Recommendation",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Retail Customer Recommendation")
st.write(
    "Simple dashboard based on Apriori association rules."
)

st.markdown(
    """
### Pages
- Executive Summary
- Customer Segmentation
- Basket & Association Analysis
- Prediction
"""
)

st.info(
    "Use the left sidebar to open a page."
)
