import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load customer dataset
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """
    Remove missing values and duplicates
    """
    df = df.drop_duplicates()
    df = df.dropna()

    return df


def scale_features(df):
    """
    Scale numerical features for clustering
    """

    features = [
        "Age",
        "Income",
        "SpendingScore",
        "PurchaseFrequency"
    ]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(df[features])

    return scaled_data, scaler