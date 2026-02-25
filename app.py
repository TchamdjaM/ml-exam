import streamlit as st


# App configuration
st.set_page_config(
    page_title="Retail Customer Recommendation",
    page_icon="🛒",
    layout="wide"
)

# Main title
st.title("🛒 Retail Customer Recommendation")

# Short project description
st.write(
    "This Web application presents the main results of the Python "
    "Machine learning Project using customer segmentation and "
    "association rules."
)

# Available pages
st.markdown(
    """
    ### Pages available
    - **Customer Segmentation**
    - **Product Associations**
    - **Prediction**

    Use the left sidebar to open a page.
    """
)
