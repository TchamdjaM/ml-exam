# Instacart Apriori Recommender

This project builds a simple grocery product recommender using the Instacart Kaggle dataset.
It learns shopping patterns from past orders with the Apriori algorithm (frequent itemsets).
From these itemsets, it generates association rules of the form **A → B** (1 product leads to 1 recommendation).

The approach stays beginner-friendly by limiting itemsets to size 2 and filtering rules with confidence and lift.
Rules are learned from the PRIOR data and evaluated on the TRAIN data with a basic offline evaluation.
The evaluation hides part of a basket and checks if recommended products match hidden items.

Results show the pipeline works, but rule coverage can be limited, so some baskets produce few recommendations.
A small Streamlit web app provides a simple interface to select basket items and display recommendations.
The project uses notebooks for preprocessing, modeling, rule mining, and evaluation, plus an `app.py` for the demo.
Goal: deliver a clear, reproducible, and interpretable recommendation baseline using association rules.
