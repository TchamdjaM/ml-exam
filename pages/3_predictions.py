import ast

import pandas as pd
import streamlit as st


@st.cache_data
def load_prediction_data():
    """
    Load files used for the prediction page.
    """
    rules_reco = pd.read_csv("outputs/apriori_train_rules_rules_reco.csv")
    baskets_test = pd.read_csv(
        "outputs/baskets_test.csv",
        usecols=["order_id", "user_id", "segment", "items"],
    )
    products = pd.read_csv(
        "data/products.csv", usecols=["product_id", "product_name"])
    return rules_reco, baskets_test, products


@st.cache_data
def parse_rules_cached(rules_df):
    """
    Convert rule columns from text to sets.
    """
    out = rules_df.copy()

    def to_set(text):
        if pd.isna(text) or str(text).strip() == "":
            return set()
        return set(str(text).split("|"))

    out["antecedents_set"] = out["antecedents"].apply(to_set)
    out["consequents_set"] = out["consequents"].apply(to_set)
    return out


@st.cache_data
def parse_baskets_cached(baskets_df):
    """
    Convert items column from string to list and keep valid baskets.
    """
    out = baskets_df.copy()
    out["items"] = out["items"].apply(ast.literal_eval)
    out["basket_size"] = out["items"].apply(len)
    out = out[out["basket_size"] >= 2].copy()
    out = out.reset_index(drop=True)
    return out


def score_candidates(observed_items, rules_df):
    """
    Score predicted items from matching rules.
    """
    observed_set = set(str(x) for x in observed_items)
    scores = {}

    for _, row in rules_df.iterrows():
        antecedents = row["antecedents_set"]
        consequents = row["consequents_set"]

        # Keep only subset -> 1 item rules
        if len(consequents) != 1:
            continue

        # A rule matches when antecedents are included in observed items
        if antecedents.issubset(observed_set):
            candidate = list(consequents)[0]

            # Do not recommend an item already in the basket
            if candidate in observed_set:
                continue

            if candidate not in scores:
                scores[candidate] = {
                    "score": 0.0,
                    "max_confidence": 0.0,
                    "max_lift": 0.0,
                    "max_support": 0.0,
                }

            conf = float(row["confidence"])
            lift = float(row["lift"])
            supp = float(row["support"])

            # Simple score: sum of confidence values
            scores[candidate]["score"] += conf
            scores[candidate]["max_confidence"] = max(
                scores[candidate]["max_confidence"], conf
            )
            scores[candidate][
                "max_lift"] = max(scores[candidate]["max_lift"], lift)
            scores[candidate]["max_support"] = max(
                scores[candidate]["max_support"], supp
            )

    rows = []
    for candidate, vals in scores.items():
        rows.append(
            {
                "candidate_product_id": candidate,
                "score": vals["score"],
                "max_confidence": vals["max_confidence"],
                "max_lift": vals["max_lift"],
                "max_support": vals["max_support"],
            }
        )

    if len(rows) == 0:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["score", "max_lift", "max_support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return out


# Page title
st.title("3) Prediction")

# Load cached data
rules_reco_raw, baskets_test_raw, products = load_prediction_data()
rules_reco = parse_rules_cached(rules_reco_raw)
baskets_test = parse_baskets_cached(baskets_test_raw)

# Build product name mapping
products["product_id"] = products["product_id"].astype(str)
product_name_map = dict(zip(products["product_id"], products["product_name"]))

# Page introduction
st.write(
    "This page shows a simple prediction demo using association rules. "
    "The app hides one product from a basket and tries to predict it."
)

# Basket selector (default fixed to 32 for the demo)
st.subheader("Choose a Basket")

default_index = 32
if default_index > len(baskets_test) - 1:
    default_index = 0

basket_index = st.number_input(
    "Basket row index",
    min_value=0,
    max_value=len(baskets_test) - 1,
    value=default_index,
    step=1,
)

row = baskets_test.iloc[int(basket_index)]
items = row["items"]

# Hide the last product to simulate prediction
observed_items = items[:-1]
hidden_item = items[-1]
hidden_item_str = str(hidden_item)

# Basket information
st.subheader("Basket Information")
basket_info = pd.DataFrame(
    [
        {
            "order_id": row["order_id"],
            "user_id": row["user_id"],
            "segment": (
                row["segment"]
                if "segment" in baskets_test.columns
                else ""
            ),
            "basket_size": len(items),
            "observed_size": len(observed_items),
            "hidden_size": 1,
        }
    ]
)
st.dataframe(basket_info, width="stretch")

# Observed basket
st.subheader("Observed Basket")
observed_df = pd.DataFrame({"product_id": [str(x) for x in observed_items]})
observed_df["product_name"] = observed_df["product_id"].map(product_name_map)
st.dataframe(observed_df, width="stretch")

# Hidden product
st.subheader("Hidden product target")
hidden_df = pd.DataFrame({"product_id": [hidden_item_str]})
hidden_df["product_name"] = hidden_df["product_id"].map(product_name_map)
st.dataframe(hidden_df, width="stretch")

# Prediction settings
st.subheader("Prediction Settings")
top_k = st.slider("Top K predictions", min_value=1, max_value=20, value=10)

# Predictions
st.subheader("Predicted Products")
preds = score_candidates(observed_items, rules_reco)
preds_topk = preds.head(top_k).copy()

if len(preds_topk) > 0:
    preds_topk["product_name"] = preds_topk["candidate_product_id"].map(
        product_name_map
    )
    preds_topk["is_hidden_target"] = (
        preds_topk["candidate_product_id"].astype(str) == hidden_item_str
    )

st.dataframe(preds_topk, width="stretch")

# Hit / miss result (safe if no predictions)
if len(preds_topk) == 0:
    st.info("No prediction available for this basket with the current rules.")
else:
    pred_set = set(preds_topk["candidate_product_id"].astype(str).tolist())

    if hidden_item_str in pred_set:
        st.success("Hit: the hidden product is in the top-k predictions.")
    else:
        st.info("Miss: the hidden product is not in the top-k predictions.")
