import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("datasets/customers.csv")

st.title("🏠 Home Dashboard")

col1,col2,col3 = st.columns(3)

col1.metric("Customers", len(df))
col2.metric("Avg Income", f"₹{int(df['Income'].mean())}")
col3.metric("Avg Spending", round(df["SpendingScore"].mean(),2))

fig1 = px.pie(
    df,
    names="Gender",
    title="Gender Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(
    df.groupby("Gender")["Income"].mean().reset_index(),
    x="Gender",
    y="Income",
    title="Average Income by Gender"
)

st.plotly_chart(fig2, use_container_width=True)