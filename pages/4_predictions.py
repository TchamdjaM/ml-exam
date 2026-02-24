import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prediction",
    page_icon="🎯",
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
    rules_df = load_csv(
        os.path.join(OUTPUT_DIR, "apriori_train_rules_rules.csv")
    )
    products_df = load_csv(os.path.join(DATA_DIR, "products.csv"))
    aisles_df = load_csv(os.path.join(DATA_DIR, "aisles.csv"))
    dep_df = load_csv(os.path.join(DATA_DIR, "departments.csv"))
    return rules_df, products_df, aisles_df, dep_df


def parse_items(s):
    if pd.isna(s):
        return []
    s = str(s).strip()
    if s == "":
        return []
    return [int(x) for x in s.split("|") if x.strip()]


def build_lookup(products_df, aisles_df, dep_df):
    if products_df.empty:
        return pd.DataFrame()

    df = products_df.copy()
    if not aisles_df.empty and "aisle_id" in df.columns:
        df = df.merge(
            aisles_df[["aisle_id", "aisle"]],
            on="aisle_id",
            how="left",
        )
    if not dep_df.empty and "department_id" in df.columns:
        df = df.merge(
            dep_df[["department_id", "department"]],
            on="department_id",
            how="left",
        )
    return df


def prepare_rules(rules_df):
    if rules_df.empty:
        return pd.DataFrame()

    df = rules_df.copy()
    df["ante_list"] = df["antecedents"].apply(parse_items)
    df["con_list"] = df["consequents"].apply(parse_items)

    df["antecedent_len"] = df["ante_list"].apply(len)
    df["consequent_len"] = df["con_list"].apply(len)

    df = df[df["consequent_len"] == 1].copy()
    return df


def build_product_maps(lookup_df):
    if lookup_df.empty:
        return {}, {}, {}

    tmp = lookup_df.copy()
    tmp["product_id"] = tmp["product_id"].astype(int)

    id_to_name = dict(zip(tmp["product_id"], tmp["product_name"]))
    name_to_id = dict(zip(tmp["product_name"], tmp["product_id"]))

    if "department" in tmp.columns:
        id_to_dep = dict(zip(tmp["product_id"], tmp["department"]))
    else:
        id_to_dep = {}

    return id_to_name, name_to_id, id_to_dep


def recommend_from_rules(
    basket_ids,
    rules_df,
    id_to_name,
    id_to_dep,
    top_k=10,
):
    rows = []

    basket_set = set(basket_ids)

    for _, r in rules_df.iterrows():
        ante = set(r["ante_list"])
        con = r["con_list"][0]

        if not ante.issubset(basket_set):
            continue

        if con in basket_set:
            continue

        rows.append(
            {
                "recommended_product_id": con,
                "recommended_product": id_to_name.get(con, str(con)),
                "department": id_to_dep.get(con, ""),
                "matched_antecedent_size": len(ante),
                "matched_antecedent": ", ".join(
                    [id_to_name.get(x, str(x)) for x in sorted(list(ante))]
                ),
                "support": r.get("support", None),
                "confidence": r.get("confidence", None),
                "lift": r.get("lift", None),
            }
        )

    if len(rows) == 0:
        return pd.DataFrame()

    rec_df = pd.DataFrame(rows)

    rec_df = rec_df.sort_values(
        ["confidence", "lift", "support", "matched_antecedent_size"],
        ascending=[False, False, False, False],
    )

    rec_df = rec_df.drop_duplicates(
        subset=["recommended_product_id"],
        keep="first",
    )

    return rec_df.head(top_k)


st.title("🎯 Prediction")
st.write(
    "Choose products already in the basket, then click Predict."
)

rules_df, products_df, aisles_df, dep_df = load_data()

if rules_df.empty:
    st.warning("Missing file: outputs/apriori_train_rules_rules.csv")
    st.stop()

lookup_df = build_lookup(products_df, aisles_df, dep_df)
rules_ready = prepare_rules(rules_df)
id_to_name, name_to_id, id_to_dep = build_product_maps(lookup_df)

if lookup_df.empty:
    st.warning("Missing product files in data/")
    st.stop()

product_names = sorted(lookup_df["product_name"].dropna().unique().tolist())

left, right = st.columns([2, 1])

with left:
    selected_products = st.multiselect(
        "Products in current basket",
        options=product_names,
        default=[],
    )

with right:
    top_k = st.slider("Top K", 1, 20, 10)

predict_btn = st.button("Predict", type="primary")

if predict_btn:
    if len(selected_products) == 0:
        st.warning("Please choose at least 1 product.")
    else:
        basket_ids = [
            name_to_id[p] for p in selected_products if p in name_to_id
        ]

        recs = recommend_from_rules(
            basket_ids=basket_ids,
            rules_df=rules_ready,
            id_to_name=id_to_name,
            id_to_dep=id_to_dep,
            top_k=top_k,
        )

        st.subheader("Current Basket")
        st.write(", ".join(selected_products))

        st.subheader("Recommended Products")
        if recs.empty:
            st.info("No recommendation found with current rules.")
        else:
            show_cols = [
                "recommended_product",
                "department",
                "matched_antecedent_size",
                "matched_antecedent",
                "confidence",
                "lift",
                "support",
            ]
            show_cols = [c for c in show_cols if c in recs.columns]
            st.dataframe(recs[show_cols], use_container_width=True)

            st.caption(
                "Recommendations are generated from Apriori rules "
                "(subset → item)."
            )
