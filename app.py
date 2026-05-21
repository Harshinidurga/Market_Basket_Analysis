import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "step" not in st.session_state:
    st.session_state.step = 1

st.title("🛒 Market Basket Analysis")

# STEP 1
if st.session_state.step == 1:
    st.header("Step 1: Upload Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file, sep=";")
        st.session_state.df = df
        st.dataframe(df.head())

        if st.button("Next"):
            st.session_state.step = 2
            st.rerun()

# STEP 2
elif st.session_state.step == 2:
    st.header("Step 2: Select Columns")

    df = st.session_state.df
    tcol = st.selectbox("Transaction Column", df.columns)
    icol = st.selectbox("Item Column", df.columns)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.step = 1
            st.rerun()

    with col2:
        if st.button("Next"):
            st.session_state.tcol = tcol
            st.session_state.icol = icol
            st.session_state.step = 3
            st.rerun()

# STEP 3
elif st.session_state.step == 3:
    st.header("Step 3: Data Processing")

    df = st.session_state.df
    data = df[[st.session_state.tcol, st.session_state.icol]].dropna()
    data.columns = ["Transaction", "Item"]

    transactions = data.groupby("Transaction")["Item"].apply(list)
    st.session_state.transactions = transactions

    st.success("✅ Data Processing Completed")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.step = 2
            st.rerun()

    with col2:
        if st.button("Next"):
            st.session_state.step = 4
            st.rerun()

# STEP 4
elif st.session_state.step == 4:
    st.header("Step 4: Frequent Itemsets")

    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05)

    te = TransactionEncoder()
    te_array = te.fit(st.session_state.transactions).transform(st.session_state.transactions)
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent = apriori(basket_df, min_support=min_support, use_colnames=True)
    frequent["support_count"] = (frequent["support"] * len(basket_df)).astype(int)

    st.session_state.frequent = frequent
    st.dataframe(frequent)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.step = 3
            st.rerun()

    with col2:
        if st.button("Next"):
            st.session_state.step = 5
            st.rerun()

# STEP 5
elif st.session_state.step == 5:
    st.header("Step 5: Association Rules")

    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)

    rules = association_rules(
        st.session_state.frequent,
        metric="confidence",
        min_threshold=min_conf
    )

    st.dataframe(rules)

    # Chart
    st.subheader("Top Items Chart")
    top10 = st.session_state.frequent.head(10)
    fig, ax = plt.subplots()
    ax.barh(top10["itemsets"].astype(str), top10["support"])
    st.pyplot(fig)

    # Download
    csv = rules.to_csv(index=False)

    if st.download_button("Download Rules CSV", csv, "rules.csv"):
        st.success("Download completed")
        st.balloons()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.step = 4
            st.rerun()

    with col2:
        if st.button("🏠 Back to Home"):
            st.session_state.step = 1
            st.rerun()
