import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📂 Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Income Distribution")

    if "Income" in df.columns:
        fig1 = px.histogram(
            df,
            x="Income",
            title="Income Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Spending Score Distribution")

    if "SpendingScore" in df.columns:
        fig2 = px.histogram(
            df,
            x="SpendingScore",
            title="Spending Score Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Matrix")

    numeric_df = df.select_dtypes(include=["number"])

    if not numeric_df.empty:
        corr = numeric_df.corr()

        fig3 = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap"
        )

        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Please upload a CSV file to begin analysis.")