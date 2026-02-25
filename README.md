# Retail Customer Recommendation

This project builds a simple product recommendation system using the Instacart dataset.

The work includes:
- data exploration (EDA)
- data quality checks
- customer segmentation
- basket creation
- association rules mining (Apriori)
- recommendation evaluation
- a Streamlit app for demo and visualization

## Project Structure

- `1_eda.ipynb` → exploratory data analysis
- `2_data_check.ipynb` → data validation and consistency checks
- `3_segmentation.ipynb` → customer features, segmentation, and data splits
- `4_creating_baskets.ipynb` → basket creation and product selection
- `5_rules_mining.ipynb` → Apriori frequent itemsets and association rules
- `6_evaluation.ipynb` → recommendation evaluation (Top-K metrics)
- `app.py` → Streamlit main app
- `1_customer_segmentation.py` → Streamlit page: customer segmentation
- `2_product_associations.py` → Streamlit page: product associations
- `3_predictions.py` → Streamlit page: prediction demo

## Main Idea

The project uses past customer baskets to find products that are often bought together.
These associations are then used to recommend products.

## Tools Used

- Python
- Pandas
- mlxtend (Apriori)
- Streamlit
- GitHub (team collaboration and version control)

## Output

The final result is:
1. a complete notebook pipeline for data mining and recommendation
2. a Streamlit app to present segmentation, rules, and prediction results
