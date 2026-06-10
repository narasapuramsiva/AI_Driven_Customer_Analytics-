import streamlit as st
import pandas as pd
import plotly.express as px

from ml.preprocess import clean_data, scale_features
from ml.segmentation import (
    find_clusters,
    add_cluster_labels,
    cluster_summary
)
from ml.recommendation import generate_recommendations

st.title("📊 Business Dashboard")

# Load Dataset
df = pd.read_csv("datasets/customers.csv")

# Clean Data
df = clean_data(df)

# Scale Features
scaled_data, scaler = scale_features(df)

# Apply K-Means
labels, model = find_clusters(
    scaled_data,
    n_clusters=4
)

df = add_cluster_labels(df, labels)

# ---------------------------
# KPI CARDS
# ---------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Average Income",
        f"₹{int(df['Income'].mean())}"
    )

with col3:
    st.metric(
        "Average Spending Score",
        round(df["SpendingScore"].mean(), 2)
    )

# ---------------------------
# CLUSTER DISTRIBUTION
# ---------------------------

st.subheader("Customer Distribution by Cluster")

cluster_count = (
    df["Cluster"]
    .value_counts()
    .reset_index()
)

cluster_count.columns = [
    "Cluster",
    "Customers"
]

fig1 = px.bar(
    cluster_count,
    x="Cluster",
    y="Customers",
    title="Customers per Cluster"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# ---------------------------
# PIE CHART
# ---------------------------

st.subheader("Cluster Percentage")

fig2 = px.pie(
    cluster_count,
    values="Customers",
    names="Cluster",
    title="Cluster Distribution"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ---------------------------
# SCATTER PLOT
# ---------------------------

st.subheader("Income vs Spending Score")

fig3 = px.scatter(
    df,
    x="Income",
    y="SpendingScore",
    color=df["Cluster"].astype(str),
    hover_data=["CustomerID"],
    title="Customer Segmentation"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ---------------------------
# CLUSTER SUMMARY
# ---------------------------

st.subheader("Cluster Summary")

summary = cluster_summary(df)

st.dataframe(
    summary,
    use_container_width=True
)

# ---------------------------
# RECOMMENDATIONS
# ---------------------------

st.subheader("Marketing Recommendations")

recommendations = generate_recommendations(df)

for cluster, details in recommendations.items():

    st.info(
        f"""
Cluster {cluster}

Customer Type: {details['Customer Type']}

Strategy: {details['Strategy']}
"""
    )

st.success("Dashboard Loaded Successfully")