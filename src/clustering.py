from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def scale_features(X):
    """
    Standardize clustering features using StandardScaler.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def find_optimal_k(X_scaled, min_k=2, max_k=10):
    """
    Evaluate K-Means models using:
    - Inertia
    - Silhouette Score
    """

    inertia_values = []
    silhouette_values = []
    k_values = range(min_k, max_k + 1)

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        inertia_values.append(
            model.inertia_
        )

        silhouette_values.append(
            silhouette_score(
                X_scaled,
                labels
            )
        )

    return (
        list(k_values),
        inertia_values,
        silhouette_values
    )


def train_kmeans(X_scaled, n_clusters=2):
    """
    Train the final K-Means clustering model.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    return model, labels


def calculate_silhouette_score(
    X_scaled,
    labels
):
    """
    Calculate the silhouette score
    for the final clustering model.
    """

    return silhouette_score(
        X_scaled,
        labels
    )