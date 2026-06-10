import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Customer Data Analysis")

df = pd.read_csv("datasets/customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df)

st.subheader("Statistics")
st.dataframe(df.describe())

# Age Distribution
fig1 = px.histogram(
    df,
    x="Age",
    title="Age Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

# Spending Score Distribution
fig2 = px.histogram(
    df,
    x="SpendingScore",
    title="Spending Score Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

# Income vs Spending
fig3 = px.scatter(
    df,
    x="Income",
    y="SpendingScore",
    color="Gender",
    title="Income vs Spending Score"
)

st.plotly_chart(fig3, use_container_width=True)