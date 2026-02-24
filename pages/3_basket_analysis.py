import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Basket & Association Analysis",
    page_icon="🔗",
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
    itemsets_df = load_csv(
        os.path.join(OUTPUT_DIR, "apriori_train_rules_frequent_itemsets.csv")
    )
    selected_df = load_csv(
        os.path.join(OUTPUT_DIR, "apriori_train_rules_selected_products.csv")
    )
    products_df = load_csv(os.path.join(DATA_DIR, "products.csv"))
    aisles_df = load_csv(os.path.join(DATA_DIR, "aisles.csv"))
    dep_df = load_csv(os.path.join(DATA_DIR, "departments.csv"))
    return rules_df, itemsets_df, selected_df, products_df, aisles_df, dep_df


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


def prepare_rules(rules_df, lookup_df):
    if rules_df.empty:
        return pd.DataFrame()

    pmap = {}
    dmap = {}
    if not lookup_df.empty:
        tmp = lookup_df.copy()
        tmp["product_id"] = tmp["product_id"].astype(int)
        pmap = dict(zip(tmp["product_id"], tmp["product_name"]))
        if "department" in tmp.columns:
            dmap = dict(zip(tmp["product_id"], tmp["department"]))

    df = rules_df.copy()
    df["ante_list"] = df["antecedents"].apply(parse_items)
    df["con_list"] = df["consequents"].apply(parse_items)

    df["antecedent_len"] = df["ante_list"].apply(len)
    df["consequent_len"] = df["con_list"].apply(len)
    df["rule_type"] = (
        df["antecedent_len"].astype(str)
        + "->"
        + df["consequent_len"].astype(str)
    )

    df["antecedents_names"] = df["ante_list"].apply(
        lambda x: ", ".join([pmap.get(i, str(i)) for i in x])
    )
    df["consequents_names"] = df["con_list"].apply(
        lambda x: ", ".join([pmap.get(i, str(i)) for i in x])
    )

    df["consequent_product_id"] = df["con_list"].apply(
        lambda x: x[0] if len(x) == 1 else None
    )
    df["consequent_department"] = df["consequent_product_id"].map(dmap)
    return df


def prepare_itemsets(itemsets_df, lookup_df):
    if itemsets_df.empty:
        return pd.DataFrame()

    pmap = {}
    if not lookup_df.empty:
        tmp = lookup_df.copy()
        tmp["product_id"] = tmp["product_id"].astype(int)
        pmap = dict(zip(tmp["product_id"], tmp["product_name"]))

    df = itemsets_df.copy()
    if "itemsets" in df.columns:
        df["item_list"] = df["itemsets"].apply(parse_items)
        df["itemset_names"] = df["item_list"].apply(
            lambda x: ", ".join([pmap.get(i, str(i)) for i in x])
        )
        df["itemset_length"] = df["item_list"].apply(len)
    return df


st.title("🔗 Basket & Association Analysis")

(
    rules_df,
    itemsets_df,
    selected_df,
    products_df,
    aisles_df,
    dep_df,
) = load_data()

if rules_df.empty:
    st.warning("Missing file: outputs/apriori_train_rules_rules.csv")
    st.stop()

lookup_df = build_lookup(products_df, aisles_df, dep_df)
rules_show = prepare_rules(rules_df, lookup_df)
itemsets_show = prepare_itemsets(itemsets_df, lookup_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rules", f"{len(rules_show):,}")

n_1to1 = len(rules_show[rules_show["rule_type"] == "1->1"])
n_2to1 = len(rules_show[rules_show["rule_type"] == "2->1"])
c2.metric("Rules 1→1", f"{n_1to1:,}")
c3.metric("Rules 2→1", f"{n_2to1:,}")
c4.metric(
    "Selected products",
    f"{len(selected_df):,}" if not selected_df.empty else "N/A",
)

st.divider()

st.subheader("Filter Rules")

a, b, c, d = st.columns(4)

rule_type_values = ["All"] + sorted(
    rules_show["rule_type"].dropna().unique().tolist()
)
selected_rule_type = a.selectbox("Rule type", rule_type_values)

min_conf = b.slider("Min confidence", 0.0, 1.0, 0.10, 0.01)
min_lift = c.slider("Min lift", 0.0, 5.0, 1.0, 0.1)
sort_by = d.selectbox("Sort by", ["confidence", "lift", "support"])

dept_values = ["All"]
if "consequent_department" in rules_show.columns:
    vals = rules_show["consequent_department"].dropna().unique().tolist()
    dept_values += sorted(vals)

selected_dept = st.selectbox("Consequent department", dept_values)

filt = rules_show.copy()

if selected_rule_type != "All":
    filt = filt[filt["rule_type"] == selected_rule_type]

filt = filt[filt["confidence"] >= min_conf]
filt = filt[filt["lift"] >= min_lift]

if selected_dept != "All":
    filt = filt[filt["consequent_department"] == selected_dept]

filt = filt.sort_values(
    [sort_by, "lift", "support"],
    ascending=[False, False, False],
)

st.write(f"Filtered rules: **{len(filt):,}**")

rule_cols = [
    "antecedents_names",
    "consequents_names",
    "consequent_department",
    "support",
    "confidence",
    "lift",
    "rule_type",
]
rule_cols = [c for c in rule_cols if c in filt.columns]
st.dataframe(filt[rule_cols], use_container_width=True)

st.divider()

st.subheader("Frequent Itemsets")
if itemsets_show.empty:
    st.info("No itemsets file found.")
else:
    lens = sorted(itemsets_show["itemset_length"].dropna().unique().tolist())
    selected_len = st.selectbox("Itemset length", lens)
    tmp = itemsets_show[itemsets_show["itemset_length"] == selected_len].copy()
    tmp = tmp.sort_values("support", ascending=False)

    cols = ["itemset_names", "support", "itemset_length"]
    cols = [c for c in cols if c in tmp.columns]
    st.dataframe(tmp[cols].head(50), use_container_width=True)
