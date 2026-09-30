import matplotlib.pyplot as plt
import plotly.express as px


def plot_elbow_curve(k_values, inertia_values):
    """
    Create the Elbow Method plot.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        k_values,
        inertia_values,
        marker="o"
    )

    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal K")

    ax.set_xticks(k_values)

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig


def plot_silhouette_curve(
    k_values,
    silhouette_values
):
    """
    Create the Silhouette Score plot.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        k_values,
        silhouette_values,
        marker="o"
    )

    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by Number of Clusters")

    ax.set_xticks(k_values)

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig


def plot_cluster_distribution(
    cluster_data
):
    """
    Create an interactive cluster distribution chart.
    """

    fig = px.bar(
        cluster_data,
        x="Cluster",
        y="Orders",
        text="Orders",
        title="Deliveries per Cluster"
    )

    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of Orders"
    )

    return fig


def plot_pca_clusters(pca_df):
    """
    Create an interactive PCA cluster visualization.
    """

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="Delivery Clusters in PCA Space",
        hover_data=["Cluster"]
    )

    fig.update_layout(
        height=650
    )

    return fig


def plot_feature_distribution(
    df,
    feature
):
    """
    Create an interactive feature distribution chart.
    """

    fig = px.histogram(
        df,
        x=feature,
        nbins=30,
        title=f"Distribution of {feature}"
    )

    return fig


def plot_feature_boxplot(
    df,
    feature
):
    """
    Create an interactive box plot.
    """

    fig = px.box(
        df,
        y=feature,
        title=f"Box Plot — {feature}"
    )

    return fig


def plot_correlation_heatmap(
    correlation_matrix
):
    """
    Create an interactive correlation heatmap.
    """

    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Feature Correlation Matrix"
    )

    return fig


def plot_cluster_comparison(
    data,
    metric
):
    """
    Compare a logistics metric across clusters.
    """

    metric_data = (
        data
        .groupby("cluster")[metric]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        metric_data,
        x="cluster",
        y=metric,
        text_auto=".2f",
        title=f"Average {metric} by Cluster"
    )

    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title=metric
    )

    return fig