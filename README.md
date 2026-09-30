# 🚚 LogiCluster
## Delivery Behavior & Route Pattern Intelligence

> An unsupervised machine learning project that uses **K-Means Clustering** to discover hidden delivery behavior and route patterns in logistics operations.

---

## 📌 Project Overview

**LogiCluster** is a logistics analytics and machine learning project designed to identify groups of delivery records with similar operational characteristics.

The project applies **K-Means Clustering** to logistics data containing information about:

- Delivery distance
- Delivery time windows
- Order priority
- Vehicle capacity
- Order weight
- Vehicle utilization
- Traffic conditions
- Average speed
- Weather impact
- Fuel cost
- Driver cost
- Route time
- Route cost
- Delivery efficiency
- Route reliability
- Time of day

The objective is to transform raw logistics data into meaningful **delivery behavior segments** that can support operational analysis.

---

# 🎯 Project Objective

The main objective of LogiCluster is to:

1. Explore logistics delivery data.
2. Identify important operational patterns.
3. Engineer meaningful machine learning features.
4. Standardize numerical features.
5. Determine an appropriate number of clusters.
6. Apply K-Means clustering.
7. Profile the resulting delivery clusters.
8. Visualize the clusters using PCA.
9. Translate cluster differences into logistics insights.
10. Assign new deliveries to learned operational clusters.

---

# 💼 Business Problem

Logistics companies generate large amounts of delivery data.

However, raw delivery records do not immediately reveal whether different deliveries follow similar operational patterns.

For example, deliveries may differ in:

- Distance
- Route time
- Route cost
- Traffic
- Vehicle utilization
- Delivery efficiency
- Reliability

Manually identifying these patterns across hundreds or thousands of records can be difficult.

### LogiCluster addresses this problem by using unsupervised machine learning to automatically group similar delivery records.

---

# 💡 Solution

The project uses **K-Means Clustering** to divide delivery records into groups based on their operational characteristics.

The resulting clusters can then be analyzed to understand differences between delivery patterns.

### Machine Learning Pipeline

```text
Raw Logistics Data
        ↓
Data Quality Analysis
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Cyclical Time Encoding
        ↓
Feature Selection
        ↓
StandardScaler
        ↓
K Selection
        ↓
K-Means Clustering
        ↓
Cluster Profiling
        ↓
PCA Visualization
        ↓
Logistics Intelligence
        ↓
New Delivery Assignment
```

---

# 🤖 Machine Learning Approach

## Algorithm

### K-Means Clustering

K-Means is an **unsupervised machine learning algorithm** that groups observations into clusters based on similarity.

The algorithm attempts to minimize the distance between observations and their assigned cluster centroid.

In this project, the final model uses:

```text
Algorithm: K-Means
Number of Clusters: 2
Random State: 42
n_init: 10
```

---

# 🔢 Feature Engineering

The dataset contains a `time_of_day` variable representing the hour of the day.

Instead of treating time as a simple linear variable, LogiCluster converts it into two cyclical features:

```text
time_sin
time_cos
```

The transformation is:

```python
time_sin = sin(2π × time_of_day / 24)

time_cos = cos(2π × time_of_day / 24)
```

This allows the model to represent the cyclical relationship between hours.

For example:

```text
23:00 → close to 00:00
```

rather than treating them as far apart numerically.

---

# 📊 Clustering Features

The model uses the following features:

```text
distance_km
delivery_time_window_hrs
order_priority
vehicle_capacity_kg
order_weight_kg
vehicle_utilization_ratio
traffic_density_index
average_speed_kmph
weather_impact_index
fuel_cost_per_km
driver_cost_per_hour
optimized_route_time_min
optimized_route_cost
delivery_efficiency_score
route_reliability_index
time_sin
time_cos
```

`order_latitude` and `order_longitude` are not included in the initial clustering feature set.

---

# 📏 Feature Scaling

K-Means is distance-based, so features with larger numerical scales could disproportionately affect the clustering process.

Therefore, the project uses:

```text
StandardScaler
```

from Scikit-learn.

The transformation standardizes the clustering features before applying K-Means.

---

# 🔢 Selecting the Number of Clusters

The project evaluates different values of K using:

## 1. Elbow Method

The Elbow Method evaluates the relationship between:

```text
Number of Clusters
        ↓
Inertia
```

The objective is to identify a point where increasing the number of clusters provides diminishing improvement.

## 2. Silhouette Score

The Silhouette Score measures how well observations fit within their assigned clusters compared with other clusters.

The project evaluated:

```text
K = 2 through K = 10
```

The highest observed silhouette score occurred at:

```text
K = 2
```

Therefore, the final clustering model uses:

```text
K = 2
```

---

# 🧠 Cluster Intelligence

After training the K-Means model, the project calculates cluster-level averages for important logistics variables.

Examples include:

- Average distance
- Average route time
- Average route cost
- Average delivery efficiency
- Average route reliability
- Average speed
- Order weight
- Traffic density

This allows the clusters to be interpreted from an operational perspective.

---

# 🚚 Operational Pattern Analysis

The current model identifies two operational groups.

Based on the observed cluster averages:

### Cluster 1

Shows a comparatively:

- Shorter route distance
- Lower optimized route time
- Lower optimized route cost
- Higher average speed

This pattern is represented in the application as:

> **Shorter & Faster Delivery Route**

### Cluster 0

Shows a comparatively:

> **Longer & More Resource-Intensive Route**

These descriptions are based on the observed cluster averages rather than predefined labels.

---

# 📐 PCA Visualization

Principal Component Analysis (**PCA**) is used to reduce the high-dimensional feature space to two dimensions for visualization.

The clustering model itself operates on the original scaled features.

PCA is used only to visualize the resulting cluster structure.

```text
17-dimensional feature space
            ↓
           PCA
            ↓
       PC1 + PC2
            ↓
  2D Cluster Visualization
```

---

# 📊 Exploratory Data Analysis

The Streamlit application provides an EDA section containing:

### Feature Distribution

Interactive histograms allow users to inspect numerical feature distributions.

### Correlation Analysis

A correlation heatmap helps examine relationships between numerical variables.

### Outlier Analysis

Interactive box plots allow individual numerical features to be inspected for potential outliers.

---

# 🖥️ Streamlit Application

LogiCluster includes an interactive Streamlit application.

## Application Sections

### 🏠 Home

Provides:

- Project overview
- Project objective
- Machine learning workflow
- Technology stack

### 📊 Data Explorer

Provides:

- Dataset preview
- Number of records
- Number of columns
- Missing value count
- Duplicate count
- Data types
- Unique values
- Statistical summary

### 📈 EDA

Provides:

- Feature distributions
- Correlation analysis
- Outlier visualization

### 🔬 Clustering Lab

Provides:

- Algorithm information
- Selected K
- Number of features
- Clustering feature list
- Silhouette Score
- Model configuration

### 🧠 Cluster Intelligence

Provides:

- Cluster distribution
- Cluster profile
- PCA visualization
- PCA explained variance

### 🚚 Logistics Insights

Provides:

- Average distance
- Average route time
- Average route cost
- Average efficiency
- Average reliability
- Cluster-level KPIs
- Cluster comparison

### 📍 Assign New Delivery

Users can enter characteristics of a new delivery.

The application then:

```text
User Input
    ↓
Feature Engineering
    ↓
Feature Ordering
    ↓
StandardScaler
    ↓
Trained K-Means Model
    ↓
Cluster Assignment
```

The delivery is assigned to the nearest learned K-Means cluster.

### ⬇️ Download

Users can download:

- Original dataset
- Clustered dataset
- Cluster profile

### 👤 About

Contains:

- Project information
- Machine learning approach
- Technology stack
- Project workflow
- Developer information

---

# 🏗️ Project Architecture

```text
LogiCluster/
│
├── .venv/
│
├── assets/
│
├── data/
│   └── ecommerce_logistics_route_planning_dataset.csv
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── clustering_features.pkl
│
├── notebooks/
│   └── 01_data_inspection.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── clustering.py
│   └── visualization.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📂 Source Code Modules

## `src/data_processing.py`

Responsible for:

- Loading the dataset
- Creating cyclical time features
- Preparing clustering data
- Validating required features

## `src/clustering.py`

Responsible for:

- Feature scaling
- K evaluation
- K-Means training
- Silhouette Score calculation

## `src/visualization.py`

Responsible for reusable visualizations including:

- Elbow curves
- Silhouette curves
- Cluster distribution
- PCA visualization
- Feature distributions
- Box plots
- Correlation heatmaps
- Cluster comparisons

---

# 📓 Jupyter Notebook

The project includes:

```text
notebooks/01_data_inspection.ipynb
```

The notebook is used for:

- Data inspection
- Data quality checks
- Exploratory analysis
- Feature engineering
- Outlier analysis
- Feature scaling
- K evaluation
- Clustering experimentation

---

# 📦 Dataset

The project uses the:

**E-Commerce Logistics Route Planning Dataset**

Dataset source:

https://www.kaggle.com/datasets/zara2099/e-commerce-logistics-route-planning-dataset/data

The dataset contains logistics-related delivery and route information.

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| Matplotlib | Visualization |
| Seaborn | Exploratory visualization |
| Plotly | Interactive visualization |
| Streamlit | Web application |
| Joblib | Model serialization |
| Jupyter Notebook | Data analysis |

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Then:

```bash
cd LogiCluster
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# 🔍 Model Files

The application uses the following trained machine learning artifacts:

```text
models/
├── kmeans_model.pkl
├── scaler.pkl
└── clustering_features.pkl
```

### `kmeans_model.pkl`

Contains the trained K-Means clustering model.

### `scaler.pkl`

Contains the fitted StandardScaler used during model training.

### `clustering_features.pkl`

Contains the feature list and ordering used by the clustering model.

Keeping the feature order consistent is important when assigning new deliveries.

---

# 📈 Key Project Outcomes

The project demonstrates an end-to-end unsupervised machine learning workflow:

```text
Data
 ↓
EDA
 ↓
Feature Engineering
 ↓
Scaling
 ↓
K Selection
 ↓
K-Means
 ↓
Cluster Profiling
 ↓
PCA
 ↓
Business Interpretation
 ↓
New Delivery Assignment
```

This project demonstrates practical skills in:

- Data Analysis
- Python
- Machine Learning
- Unsupervised Learning
- Feature Engineering
- Data Visualization
- Business Intelligence
- Logistics Analytics
- Streamlit Development

---

# 🔮 Future Improvements

Potential future improvements include:

- Add DBSCAN clustering
- Compare multiple clustering algorithms
- Add automated cluster naming
- Add geographic route visualization
- Add interactive logistics maps
- Add delivery-risk analysis
- Add time-based delivery analysis
- Add route optimization experiments
- Add model monitoring
- Add automated model retraining
- Add database integration
- Add cloud deployment
- Add authentication
- Add advanced logistics KPIs

---

# 🚀 Future Project Vision

The long-term goal of LogiCluster is to evolve from a clustering project into a more complete logistics intelligence platform.

```text
Data Sources
     ↓
Data Pipeline
     ↓
Data Warehouse
     ↓
Analytics Layer
     ↓
Machine Learning
     ↓
Logistics Intelligence
     ↓
Interactive Dashboard
     ↓
Operational Decision Support
```

---

# 👨‍💻 Developer

## Prajyot Yesankar

**Data Analyst Aspirant**

### Skills

```text
SQL
Python
Excel
Power BI
Machine Learning
Data Analytics
```

### Connect With Me

**LinkedIn**

https://www.linkedin.com/in/prajyot-yesankar-79215b258/

**GitHub**

https://github.com/yesankarprajyot123

---

# 📜 License

This project is created for educational, portfolio, and demonstration purposes.

---

# ⭐ Project

If you find this project useful, feel free to explore the repository and experiment with the clustering approach.

---

## 🚚 LogiCluster

**Delivery Behavior & Route Pattern Intelligence**

> Turning logistics data into operational patterns with unsupervised machine learning.