import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load the logistics dataset.
    """
    return pd.read_csv(file_path)


def add_time_features(df):
    """
    Create cyclical time features from time_of_day.
    """

    df = df.copy()

    if "time_of_day" in df.columns:

        df["time_sin"] = np.sin(
            2 * np.pi * df["time_of_day"] / 24
        )

        df["time_cos"] = np.cos(
            2 * np.pi * df["time_of_day"] / 24
        )

    return df


def prepare_clustering_data(df, clustering_features):
    """
    Prepare the dataset using the features required
    by the K-Means clustering model.
    """

    df = add_time_features(df)

    missing_features = [
        feature
        for feature in clustering_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = df[clustering_features].copy()

    return df, X