from sklearn.cluster import KMeans
import pandas as pd


def find_clusters(data, n_clusters=4):
    """
    Apply K-Means clustering
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(data)

    return labels, model


def add_cluster_labels(df, labels):
    """
    Add cluster column to dataframe
    """

    df["Cluster"] = labels

    return df


def cluster_summary(df):
    """
    Generate cluster statistics
    """

    summary = df.groupby("Cluster").agg(
        {
            "Age": "mean",
            "Income": "mean",
            "SpendingScore": "mean",
            "PurchaseFrequency": "mean"
        }
    )

    return summary