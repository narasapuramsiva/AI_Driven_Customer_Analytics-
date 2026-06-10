import streamlit as st
import pandas as pd
import plotly.express as px

from ml.preprocess import load_data, clean_data, scale_features
from ml.segmentation import (
    find_clusters,
    add_cluster_labels,
    cluster_summary
)
from ml.recommendation import generate_recommendations


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Driven Customer Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Driven Customer Analytics Platform")
st.markdown("### Customer Segmentation using K-Means Clustering")

st.sidebar.header("Settings")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    df = load_data("datasets/customers.csv")

except FileNotFoundError:
    st.error("customers.csv file not found!")
    st.stop()

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

df = clean_data(df)

# --------------------------------------------------
# SHOW DATASET
# --------------------------------------------------

st.subheader("Customer Dataset")

st.dataframe(
    df,
    use_container_width=True
)

# --------------------------------------------------
# DATASET INFO
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Average Income",
        f"₹ {int(df['Income'].mean())}"
    )

with col3:
    st.metric(
        "Average Spending Score",
        round(df["SpendingScore"].mean(), 2)
    )
import streamlit as st
import pandas as pd
import plotly.express as px

from ml.preprocess import (
    load_data,
    clean_data,
    scale_features
)

from ml.segmentation import (
    find_clusters,
    add_cluster_labels,
    cluster_summary
)

from ml.recommendation import generate_recommendations


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Driven Customer Analytics",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 AI-Driven Customer Analytics Platform")
st.markdown("### Customer Segmentation using K-Means Clustering")

st.sidebar.header("Configuration")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    df = load_data("datasets/customers.csv")

except FileNotFoundError:
    st.error("customers.csv file not found inside datasets folder.")
    st.stop()

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

df = clean_data(df)

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

k = st.sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
    value=4
)

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader("📂 Customer Dataset")

st.dataframe(
    df,
    use_container_width=True
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

st.subheader("📈 Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Average Age",
        round(df["Age"].mean(), 1)
    )

with col3:
    st.metric(
        "Average Income",
        f"₹ {int(df['Income'].mean())}"
    )

with col4:
    st.metric(
        "Average Spending Score",
        round(df["SpendingScore"].mean(), 1)
    )

# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

scaled_data, scaler = scale_features(df)

# --------------------------------------------------
# K-MEANS CLUSTERING
# --------------------------------------------------

labels, model = find_clusters(
    scaled_data,
    n_clusters=k
)

df = add_cluster_labels(
    df,
    labels
)

# --------------------------------------------------
# CLUSTERED DATA
# --------------------------------------------------

st.subheader("🤖 Clustered Customer Data")

st.dataframe(
    df,
    use_container_width=True
)

# --------------------------------------------------
# CLUSTER COUNTS
# --------------------------------------------------

cluster_count = (
    df["Cluster"]
    .value_counts()
    .reset_index()
)

cluster_count.columns = [
    "Cluster",
    "Customer Count"
]

# --------------------------------------------------
# BAR CHART
# --------------------------------------------------

st.subheader("📊 Customer Distribution by Cluster")

fig_bar = px.bar(
    cluster_count,
    x="Cluster",
    y="Customer Count",
    text="Customer Count",
    title="Customers in Each Cluster"
)

st.plotly_chart(
    fig_bar,
    use_container_width=True,
    key="customer_distribution_bar"
)

# --------------------------------------------------
# PIE CHART
# --------------------------------------------------

st.subheader("🥧 Cluster Percentage")

fig_pie = px.pie(
    cluster_count,
    values="Customer Count",
    names="Cluster",
    title="Cluster Distribution"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True,
    key="cluster_distribution_pie"
)

# --------------------------------------------------
# SCATTER PLOT
# --------------------------------------------------

st.subheader("💰 Income vs Spending Score")

fig_scatter = px.scatter(
    df,
    x="Income",
    y="SpendingScore",
    color=df["Cluster"].astype(str),
    hover_data=["CustomerID"],
    title="Customer Segmentation"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True,
    key="income_spending_scatter"
)

# --------------------------------------------------
# CLUSTER SUMMARY
# --------------------------------------------------

st.subheader("📋 Cluster Summary Statistics")

summary = cluster_summary(df)

st.dataframe(
    summary,
    use_container_width=True
)

# --------------------------------------------------
# BUSINESS RECOMMENDATIONS
# --------------------------------------------------

st.subheader("🎯 Marketing Recommendations")

recommendations = generate_recommendations(df)

for cluster, details in recommendations.items():

    st.info(
        f"""
Cluster {cluster}

Customer Type: {details['Customer Type']}

Strategy: {details['Strategy']}
"""
    )

# --------------------------------------------------
# DOWNLOAD SECTION
# --------------------------------------------------

st.subheader("⬇️ Download Results")

csv = df.to_csv(index=False)

st.download_button(
    label="Download Clustered Customer Data",
    data=csv,
    file_name="customer_segmentation_results.csv",
    mime="text/csv"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.success(
    "Customer Segmentation Completed Successfully!"
)

st.caption(
    "Developed using Streamlit, Pandas, Scikit-Learn and Plotly."
)



# --------------------------------------------------
# K VALUE SELECTION
# --------------------------------------------------

k = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=4
)

# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

scaled_data, scaler = scale_features(df)

# --------------------------------------------------
# KMEANS CLUSTERING
# --------------------------------------------------

labels, model = find_clusters(
    scaled_data,
    n_clusters=k
)

df = add_cluster_labels(
    df,
    labels
)

# --------------------------------------------------
# CLUSTERED DATA
# --------------------------------------------------

st.subheader("Clustered Customer Data")

st.dataframe(
    df,
    use_container_width=True
)

# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------

st.subheader("Customer Distribution by Cluster")

cluster_count = (
    df["Cluster"]
    .value_counts()
    .reset_index()
)

cluster_count.columns = [
    "Cluster",
    "Customer Count"
]

fig1 = px.bar(
    cluster_count,
    x="Cluster",
    y="Customer Count",
    title="Customers in Each Cluster",
    text="Customer Count"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# --------------------------------------------------
# SCATTER PLOT
# --------------------------------------------------

st.subheader("Income vs Spending Score")

fig2 = px.scatter(
    df,
    x="Income",
    y="SpendingScore",
    color=df["Cluster"].astype(str),
    hover_data=["CustomerID"],
    title="Customer Segmentation"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# --------------------------------------------------
# PIE CHART
# --------------------------------------------------

st.subheader("Cluster Percentage")

fig3 = px.pie(
    cluster_count,
    values="Customer Count",
    names="Cluster",
    title="Cluster Distribution"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# --------------------------------------------------
# CLUSTER SUMMARY
# --------------------------------------------------

st.subheader("Cluster Summary Statistics")

summary = cluster_summary(df)

st.dataframe(
    summary,
    use_container_width=True
)

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

st.subheader("Business Recommendations")

recommendations = generate_recommendations(df)

for cluster, details in recommendations.items():

    st.info(
        f"""
Cluster {cluster}

Customer Type: {details['Customer Type']}

Strategy: {details['Strategy']}
"""
    )

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.subheader("Download Segmentation Results")

csv = df.to_csv(index=False)

st.download_button(
    label="Download Clustered Data",
    data=csv,
    file_name="customer_segmentation_results.csv",
    mime="text/csv"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.success("Customer Segmentation Completed Successfully!")

st.markdown("---")
st.markdown(
    "Developed using Streamlit, Scikit-Learn, Pandas and Plotly"
)