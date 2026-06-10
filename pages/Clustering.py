import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

from ml.preprocess import clean_data, scale_features
from ml.segmentation import find_clusters, add_cluster_labels

st.title("🤖 Customer Segmentation")

# Load Dataset
df = pd.read_csv("datasets/customers.csv")

# Clean Data
df = clean_data(df)

# Scale Features
scaled_data, scaler = scale_features(df)

# Elbow Method
wcss = []

for i in range(1, 11):
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_data)
    wcss.append(model.inertia_)

st.subheader("Elbow Method")

fig1 = px.line(
    x=list(range(1, 11)),
    y=wcss,
    markers=True,
    labels={"x": "Number of Clusters", "y": "WCSS"},
    title="Elbow Method for Optimal K"
)

st.plotly_chart(fig1, use_container_width=True)

# Select K
k = st.slider(
    "Select Number of Clusters",
    2,
    10,
    4
)

# KMeans Clustering
labels, model = find_clusters(
    scaled_data,
    n_clusters=k
)

df = add_cluster_labels(df, labels)

st.subheader("Clustered Dataset")

st.dataframe(df, use_container_width=True)

# Scatter Plot
fig2 = px.scatter(
    df,
    x="Income",
    y="SpendingScore",
    color=df["Cluster"].astype(str),
    hover_data=["CustomerID"],
    title="Customer Segments"
)

st.plotly_chart(fig2, use_container_width=True)

# Cluster Count
cluster_count = (
    df["Cluster"]
    .value_counts()
    .reset_index()
)

cluster_count.columns = ["Cluster", "Customers"]

fig3 = px.bar(
    cluster_count,
    x="Cluster",
    y="Customers",
    title="Customers per Cluster"
)

st.plotly_chart(fig3, use_container_width=True)