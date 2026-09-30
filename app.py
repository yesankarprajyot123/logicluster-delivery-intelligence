# ============================================================
# LOGICLUSTER
# Delivery Behavior & Route Pattern Intelligence
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.decomposition import PCA

# ============================================================
# PROJECT MODULES
# ============================================================

from src.data_processing import (
    load_data as load_logistics_data,
    prepare_clustering_data
)

from src.clustering import calculate_silhouette_score

from src.visualization import (
    plot_cluster_distribution,
    plot_pca_clusters,
    plot_feature_distribution,
    plot_feature_boxplot,
    plot_correlation_heatmap,
    plot_cluster_comparison
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LogiCluster | Prajyot Yesankar",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ecommerce_logistics_route_planning_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "kmeans_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "clustering_features.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_project_data():

    if not os.path.exists(DATA_PATH):

        st.error(
            "Dataset file not found.\n\n"
            "Expected file:\n"
            "data/ecommerce_logistics_route_planning_dataset.csv"
        )

        st.stop()

    return load_logistics_data(DATA_PATH)


# ============================================================
# LOAD MACHINE LEARNING OBJECTS
# ============================================================

@st.cache_resource
def load_ml_objects():

    if not os.path.exists(MODEL_PATH):

        st.error(
            "K-Means model not found.\n\n"
            "Expected:\n"
            "models/kmeans_model.pkl"
        )

        st.stop()

    if not os.path.exists(SCALER_PATH):

        st.error(
            "Scaler not found.\n\n"
            "Expected:\n"
            "models/scaler.pkl"
        )

        st.stop()

    if not os.path.exists(FEATURES_PATH):

        st.error(
            "Feature list not found.\n\n"
            "Expected:\n"
            "models/clustering_features.pkl"
        )

        st.stop()

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    features = joblib.load(FEATURES_PATH)

    return model, scaler, features


# ============================================================
# LOAD PROJECT
# ============================================================

df = load_project_data()

kmeans_model, scaler, clustering_features = load_ml_objects()


# ============================================================
# PREPARE CLUSTERING DATA
# ============================================================

try:

    analysis_df, X = prepare_clustering_data(
        df,
        clustering_features
    )

except ValueError as error:

    st.error(str(error))

    st.stop()


# ============================================================
# SCALE DATA
# ============================================================

X_scaled = scaler.transform(X)


# ============================================================
# GENERATE CLUSTER LABELS
# ============================================================

cluster_labels = kmeans_model.predict(
    X_scaled
)

analysis_df["cluster"] = cluster_labels


# ============================================================
# CLUSTER PROFILE
# ============================================================

cluster_profile = (
    analysis_df
    .groupby("cluster")[clustering_features]
    .mean()
    .round(2)
)


# ============================================================
# CLUSTER SIZE
# ============================================================

cluster_sizes = (
    analysis_df["cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CLUSTER KPI TABLE
# ============================================================

cluster_kpis = (
    analysis_df
    .groupby("cluster")
    .agg(
        Orders=("cluster", "size"),

        Avg_Distance_KM=(
            "distance_km",
            "mean"
        ),

        Avg_Route_Time_Min=(
            "optimized_route_time_min",
            "mean"
        ),

        Avg_Route_Cost=(
            "optimized_route_cost",
            "mean"
        ),

        Avg_Efficiency=(
            "delivery_efficiency_score",
            "mean"
        ),

        Avg_Reliability=(
            "route_reliability_index",
            "mean"
        ),

        Avg_Speed=(
            "average_speed_kmph",
            "mean"
        )
    )
    .round(2)
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚚 LogiCluster")

st.sidebar.write(
    "Delivery Behavior & Route Pattern Intelligence"
)

st.sidebar.caption(
    "Discover hidden logistics patterns using K-Means Clustering."
)

st.sidebar.divider()


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Explorer",
        "📈 EDA",
        "🔬 Clustering Lab",
        "🧠 Cluster Intelligence",
        "🚚 Logistics Insights",
        "📍 Assign New Delivery",
        "⬇️ Download",
        "👤 About"
    ]
)


# ============================================================
# SIDEBAR BRANDING
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "👨‍💻 Prajyot Yesankar"
)

st.sidebar.write(
    "Data Analyst Aspirant"
)

st.sidebar.markdown(
    "[🔗 LinkedIn](https://www.linkedin.com/in/prajyot-yesankar-79215b258/)"
)

st.sidebar.markdown(
    "[💻 GitHub](https://github.com/yesankarprajyot123)"
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("🚚 LogiCluster")

    st.subheader(
        "Delivery Behavior & Route Pattern Intelligence"
    )

    st.write(
        """
        LogiCluster is an unsupervised machine learning
        application designed to discover hidden patterns
        in logistics and delivery operations.
        """
    )

    st.write(
        """
        The project uses K-Means Clustering to group
        delivery records according to operational
        characteristics such as distance, route time,
        cost, traffic, speed, efficiency, reliability,
        and vehicle utilization.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.subheader("📊 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Orders",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "ML Features",
            len(clustering_features)
        )

    with col3:

        st.metric(
            "Clusters",
            kmeans_model.n_clusters
        )

    with col4:

        st.metric(
            "Algorithm",
            "K-Means"
        )

    st.divider()

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write(
        """
        The objective of LogiCluster is to identify
        meaningful delivery behavior patterns from
        logistics data and translate those patterns
        into operational insights.

        Unlike supervised learning, there is no
        predefined target variable. K-Means discovers
        groups of deliveries with similar operational
        characteristics.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.subheader("🔄 Machine Learning Workflow")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.info(
            "1. Data\n\n"
            "Logistics delivery records"
        )

    with c2:

        st.info(
            "2. Preparation\n\n"
            "Feature engineering and scaling"
        )

    with c3:

        st.info(
            "3. Clustering\n\n"
            "K-Means segmentation"
        )

    with c4:

        st.info(
            "4. Intelligence\n\n"
            "Business and logistics insights"
        )

    st.divider()

    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.subheader("🛠️ Technology Stack")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:

        st.write("🐍 Python")
        st.write("🐼 Pandas")
        st.write("🔢 NumPy")

    with tech2:

        st.write("🤖 Scikit-learn")
        st.write("📊 Plotly")
        st.write("📈 Matplotlib")

    with tech3:

        st.write("🎨 Seaborn")
        st.write("⚡ Streamlit")
        st.write("💾 Joblib")

    with tech4:

        st.write("📦 K-Means")
        st.write("📐 PCA")
        st.write("📏 StandardScaler")


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "📊 Data Explorer":

    st.title("📊 Data Explorer")

    st.write(
        "Explore the logistics dataset used by LogiCluster."
    )

    st.subheader("📌 Dataset KPIs")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:

        st.metric(
            "Duplicates",
            int(df.duplicated().sum())
        )

    st.divider()

    st.subheader("Dataset Preview")

    rows_to_show = st.slider(
        "Rows to display",
        min_value=10,
        max_value=min(500, len(df)),
        value=100,
        step=10
    )

    st.dataframe(
        df.head(rows_to_show),
        use_container_width=True,
        height=450
    )

    st.divider()

    st.subheader("Dataset Information")

    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        }
    )

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )


# ============================================================
# EDA
# ============================================================

elif page == "📈 EDA":

    st.title("📈 Exploratory Data Analysis")

    st.write(
        """
        Explore feature distributions, relationships,
        correlations, and potential outliers.
        """
    )

    numeric_columns = (
        df
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📊 Feature Distribution")

    selected_feature = st.selectbox(
        "Select a feature",
        numeric_columns
    )

    fig_hist = plot_feature_distribution(
        df,
        selected_feature
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader("🔗 Correlation Analysis")

    correlation_matrix = (
        df[numeric_columns]
        .corr()
    )

    fig_corr = plot_correlation_heatmap(
        correlation_matrix
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    st.subheader("📦 Outlier Analysis")

    selected_outlier_feature = st.selectbox(
        "Select feature",
        numeric_columns,
        key="outlier_feature"
    )

    fig_box = plot_feature_boxplot(
        df,
        selected_outlier_feature
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )


# ============================================================
# CLUSTERING LAB
# ============================================================

elif page == "🔬 Clustering Lab":

    st.title("🔬 Clustering Lab")

    st.write(
        """
        Explore the K-Means configuration used by
        LogiCluster.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Algorithm",
            "K-Means"
        )

    with col2:

        st.metric(
            "Selected K",
            kmeans_model.n_clusters
        )

    with col3:

        st.metric(
            "Features",
            len(clustering_features)
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader("🧩 Features Used for Clustering")

    feature_df = pd.DataFrame(
        {
            "Feature": clustering_features
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # SILHOUETTE SCORE
    # --------------------------------------------------------

    st.subheader("📏 Silhouette Score")

    try:

        silhouette = calculate_silhouette_score(
            X_scaled,
            cluster_labels
        )

        st.metric(
            "Final Silhouette Score",
            f"{silhouette:.4f}"
        )

    except Exception:

        st.warning(
            "Silhouette score could not be calculated."
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader("⚙️ Model Configuration")

    st.write(
        f"Number of clusters: {kmeans_model.n_clusters}"
    )

    st.write(
        "Random state: 42"
    )

    st.write(
        "Initialization: K-Means with n_init=10"
    )

    st.write(
        "Scaling method: StandardScaler"
    )


# ============================================================
# CLUSTER INTELLIGENCE
# ============================================================

elif page == "🧠 Cluster Intelligence":

    st.title("🧠 Cluster Intelligence")

    st.write(
        """
        Explore the operational characteristics of the
        delivery clusters discovered by K-Means.
        """
    )

    # --------------------------------------------------------
    # CLUSTER DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📊 Cluster Distribution")

    distribution_df = (
        cluster_sizes
        .reset_index()
    )

    distribution_df.columns = [
        "Cluster",
        "Orders"
    ]

    fig_cluster = plot_cluster_distribution(
        distribution_df
    )

    st.plotly_chart(
        fig_cluster,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # CLUSTER PROFILE
    # --------------------------------------------------------

    st.subheader("📋 Cluster Profile")

    st.dataframe(
        cluster_profile,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    st.subheader("📐 PCA Visualization")

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": cluster_labels.astype(str)
        }
    )

    fig_pca = plot_pca_clusters(
        pca_df
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True
    )

    st.caption(
        "PCA is used only for visualization. "
        "K-Means operates on the original scaled features."
    )

    st.divider()

    # --------------------------------------------------------
    # EXPLAINED VARIANCE
    # --------------------------------------------------------

    st.subheader("📈 PCA Explained Variance")

    explained_variance_df = pd.DataFrame(
        {
            "Component": [
                "PC1",
                "PC2"
            ],

            "Explained Variance (%)": [
                round(
                    pca.explained_variance_ratio_[0] * 100,
                    2
                ),

                round(
                    pca.explained_variance_ratio_[1] * 100,
                    2
                )
            ]
        }
    )

    st.dataframe(
        explained_variance_df,
        use_container_width=True
    )


# ============================================================
# LOGISTICS INSIGHTS
# ============================================================

elif page == "🚚 Logistics Insights":

    st.title("🚚 Logistics Insights")

    st.write(
        """
        Operational KPIs and cluster-level logistics insights.
        """
    )

    # --------------------------------------------------------
    # OVERALL KPIs
    # --------------------------------------------------------

    avg_distance = (
        analysis_df["distance_km"].mean()
    )

    avg_route_time = (
        analysis_df[
            "optimized_route_time_min"
        ].mean()
    )

    avg_route_cost = (
        analysis_df[
            "optimized_route_cost"
        ].mean()
    )

    avg_efficiency = (
        analysis_df[
            "delivery_efficiency_score"
        ].mean()
    )

    avg_reliability = (
        analysis_df[
            "route_reliability_index"
        ].mean()
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Avg Distance",
            f"{avg_distance:.2f} km"
        )

    with col2:

        st.metric(
            "Avg Route Time",
            f"{avg_route_time:.2f} min"
        )

    with col3:

        st.metric(
            "Avg Route Cost",
            f"{avg_route_cost:.2f}"
        )

    with col4:

        st.metric(
            "Avg Efficiency",
            f"{avg_efficiency:.2f}"
        )

    with col5:

        st.metric(
            "Avg Reliability",
            f"{avg_reliability:.2f}"
        )

    st.divider()

    # --------------------------------------------------------
    # CLUSTER KPIs
    # --------------------------------------------------------

    st.subheader("📊 Cluster-Level KPIs")

    st.dataframe(
        cluster_kpis,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # CLUSTER COMPARISON
    # --------------------------------------------------------

    st.subheader("🔎 Cluster Comparison")

    metric_options = [
        "distance_km",
        "optimized_route_time_min",
        "optimized_route_cost",
        "average_speed_kmph",
        "order_weight_kg",
        "delivery_efficiency_score",
        "route_reliability_index",
        "traffic_density_index"
    ]

    selected_metric = st.selectbox(
        "Select logistics metric",
        metric_options
    )

    fig_metric = plot_cluster_comparison(
        analysis_df,
        selected_metric
    )

    st.plotly_chart(
        fig_metric,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # BUSINESS INTERPRETATION
    # --------------------------------------------------------

    st.subheader("💡 Operational Interpretation")

    if (
        0 in cluster_profile.index
        and 1 in cluster_profile.index
    ):

        distance_difference = (
            cluster_profile.loc[
                1,
                "distance_km"
            ]
            -
            cluster_profile.loc[
                0,
                "distance_km"
            ]
        )

        time_difference = (
            cluster_profile.loc[
                1,
                "optimized_route_time_min"
            ]
            -
            cluster_profile.loc[
                0,
                "optimized_route_time_min"
            ]
        )

        cost_difference = (
            cluster_profile.loc[
                1,
                "optimized_route_cost"
            ]
            -
            cluster_profile.loc[
                0,
                "optimized_route_cost"
            ]
        )

        speed_difference = (
            cluster_profile.loc[
                1,
                "average_speed_kmph"
            ]
            -
            cluster_profile.loc[
                0,
                "average_speed_kmph"
            ]
        )

        if (
            distance_difference < 0
            and time_difference < 0
            and cost_difference < 0
            and speed_difference > 0
        ):

            st.success(
                """
                Cluster 1 shows a shorter, faster and
                lower-cost delivery pattern relative to
                Cluster 0 based on the observed cluster
                averages.

                Cluster 0 shows a comparatively longer
                and more resource-intensive route pattern.
                """
            )

        else:

            st.info(
                """
                The clusters exhibit different operational
                characteristics. Use the cluster profile and
                comparison charts above to examine the
                specific differences.
                """
            )


# ============================================================
# ASSIGN NEW DELIVERY
# ============================================================

elif page == "📍 Assign New Delivery":

    st.title("📍 Assign New Delivery")

    st.write(
        """
        Enter the characteristics of a new delivery.
        LogiCluster will assign it to one of the learned
        operational clusters.
        """
    )

    st.info(
        """
        This is a cluster assignment, not a supervised
        prediction. The new delivery is assigned to the
        nearest learned K-Means cluster.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        distance_km = st.number_input(
            "Distance (km)",
            min_value=0.0,
            value=25.0,
            step=1.0
        )

        delivery_time_window_hrs = st.number_input(
            "Delivery Time Window (hrs)",
            min_value=0.0,
            value=4.0,
            step=0.5
        )

        order_priority = st.number_input(
            "Order Priority",
            min_value=1,
            max_value=4,
            value=2,
            step=1
        )

        vehicle_capacity_kg = st.number_input(
            "Vehicle Capacity (kg)",
            min_value=0.0,
            value=100.0,
            step=5.0
        )

        order_weight_kg = st.number_input(
            "Order Weight (kg)",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

        vehicle_utilization_ratio = st.number_input(
            "Vehicle Utilization Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05
        )

        traffic_density_index = st.number_input(
            "Traffic Density Index",
            min_value=0.0,
            value=0.40,
            step=0.05
        )

    with col2:

        average_speed_kmph = st.number_input(
            "Average Speed (km/h)",
            min_value=0.0,
            value=35.0,
            step=1.0
        )

        weather_impact_index = st.number_input(
            "Weather Impact Index",
            min_value=0.0,
            value=0.20,
            step=0.05
        )

        fuel_cost_per_km = st.number_input(
            "Fuel Cost per km",
            min_value=0.0,
            value=8.0,
            step=0.5
        )

        driver_cost_per_hour = st.number_input(
            "Driver Cost per Hour",
            min_value=0.0,
            value=150.0,
            step=10.0
        )

        optimized_route_time_min = st.number_input(
            "Optimized Route Time (min)",
            min_value=0.0,
            value=60.0,
            step=5.0
        )

        optimized_route_cost = st.number_input(
            "Optimized Route Cost",
            min_value=0.0,
            value=500.0,
            step=25.0
        )

        delivery_efficiency_score = st.number_input(
            "Delivery Efficiency Score",
            min_value=0.0,
            value=0.75,
            step=0.05
        )

        route_reliability_index = st.number_input(
            "Route Reliability Index",
            min_value=0.0,
            value=0.80,
            step=0.05
        )

        time_of_day = st.number_input(
            "Time of Day (0–23)",
            min_value=0,
            max_value=23,
            value=14,
            step=1
        )

    st.divider()

    # --------------------------------------------------------
    # ASSIGN BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚚 Assign Delivery to Cluster",
        use_container_width=True,
        type="primary"
    ):

        new_delivery = {

            "distance_km":
                distance_km,

            "delivery_time_window_hrs":
                delivery_time_window_hrs,

            "order_priority":
                order_priority,

            "vehicle_capacity_kg":
                vehicle_capacity_kg,

            "order_weight_kg":
                order_weight_kg,

            "vehicle_utilization_ratio":
                vehicle_utilization_ratio,

            "traffic_density_index":
                traffic_density_index,

            "average_speed_kmph":
                average_speed_kmph,

            "weather_impact_index":
                weather_impact_index,

            "fuel_cost_per_km":
                fuel_cost_per_km,

            "driver_cost_per_hour":
                driver_cost_per_hour,

            "optimized_route_time_min":
                optimized_route_time_min,

            "optimized_route_cost":
                optimized_route_cost,

            "delivery_efficiency_score":
                delivery_efficiency_score,

            "route_reliability_index":
                route_reliability_index,

            "time_of_day":
                time_of_day
        }

        # ----------------------------------------------------
        # TIME CYCLICAL FEATURES
        # ----------------------------------------------------

        hour = new_delivery["time_of_day"]

        new_delivery["time_sin"] = np.sin(
            2 * np.pi * hour / 24
        )

        new_delivery["time_cos"] = np.cos(
            2 * np.pi * hour / 24
        )

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        new_delivery_df = pd.DataFrame(
            [new_delivery]
        )

        # ----------------------------------------------------
        # SAME FEATURE ORDER AS TRAINING
        # ----------------------------------------------------

        try:

            new_delivery_features = (
                new_delivery_df[
                    clustering_features
                ]
            )

        except KeyError as error:

            st.error(
                f"Required feature missing: {error}"
            )

            st.stop()

        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        new_delivery_scaled = scaler.transform(
            new_delivery_features
        )

        # ----------------------------------------------------
        # ASSIGN CLUSTER
        # ----------------------------------------------------

        new_cluster = int(
            kmeans_model.predict(
                new_delivery_scaled
            )[0]
        )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        if new_cluster == 0:

            interpretation = (
                "Longer & More Resource-Intensive Route"
            )

        else:

            interpretation = (
                "Shorter & Faster Delivery Route"
            )

        st.divider()

        st.subheader("🎯 Assignment Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Assigned Cluster",
                str(new_cluster)
            )

        with result_col2:

            st.metric(
                "Operational Pattern",
                interpretation
            )

        st.success(
            f"New delivery assigned to Cluster {new_cluster}."
        )


# ============================================================
# DOWNLOAD
# ============================================================

elif page == "⬇️ Download":

    st.title("⬇️ Download")

    st.write(
        "Download the dataset and clustering results."
    )

    # --------------------------------------------------------
    # ORIGINAL DATA
    # --------------------------------------------------------

    st.subheader("📥 Original Dataset")

    original_csv = (
        df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="Download Original Dataset",
        data=original_csv,
        file_name="logicluster_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # CLUSTERED DATA
    # --------------------------------------------------------

    st.subheader("📥 Clustered Dataset")

    clustered_csv = (
        analysis_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="Download Clustered Dataset",
        data=clustered_csv,
        file_name="logicluster_clustered_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # CLUSTER PROFILE
    # --------------------------------------------------------

    st.subheader("📋 Cluster Profile")

    profile_csv = (
        cluster_profile
        .to_csv()
        .encode("utf-8")
    )

    st.download_button(
        label="Download Cluster Profile",
        data=profile_csv,
        file_name="logicluster_cluster_profile.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "👤 About":

    st.title("👤 About LogiCluster")

    st.subheader(
        "🚚 Delivery Behavior & Route Pattern Intelligence"
    )

    st.write(
        """
        LogiCluster is an unsupervised machine learning
        project focused on discovering hidden patterns
        in logistics and delivery operations.
        """
    )

    st.divider()

    st.subheader("🤖 Machine Learning")

    st.write(
        """
        • K-Means Clustering

        • StandardScaler

        • Elbow Method

        • Silhouette Score

        • Principal Component Analysis (PCA)
        """
    )

    st.divider()

    st.subheader("🛠️ Technology Stack")

    st.write(
        """
        Python • Pandas • NumPy • Scikit-learn • Plotly
        • Streamlit • Matplotlib • Seaborn • Joblib
        """
    )

    st.divider()

    st.subheader("🔄 Project Workflow")

    st.write(
        """
        Dataset → Data Quality → EDA → Feature Engineering
        → Scaling → K Selection → K-Means → Cluster Profiling
        → PCA → Logistics Insights → New Delivery Assignment
        """
    )

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.write(
        "### Prajyot Yesankar"
    )

    st.write(
        "Data Analyst Aspirant"
    )

    st.write(
        "SQL • Python • Excel • Power BI • Machine Learning"
    )

    st.markdown(
        "[🔗 LinkedIn](https://www.linkedin.com/in/prajyot-yesankar-79215b258/)"
    )

    st.markdown(
        "[💻 GitHub](https://github.com/yesankarprajyot123)"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚚 LogiCluster"
)

st.caption(
    "Delivery Behavior & Route Pattern Intelligence"
)

st.caption(
    "Built with Python & Streamlit by Prajyot Yesankar"
)

st.markdown(
    "[🔗 LinkedIn](https://www.linkedin.com/in/prajyot-yesankar-79215b258/) "
    " • "
    "[💻 GitHub](https://github.com/yesankarprajyot123)"
)

st.caption(
    "© 2026 Prajyot Yesankar · LogiCluster"
)