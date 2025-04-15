import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title and setup
st.set_page_config(page_title="Sleep Efficiency ML Dashboard", layout="wide")
st.title("Sleep Efficiency ML Dashboard")

# Load dataset
df = pd.read_csv("C:/Users/rhyth/Downloads/Sleep_Efficiency (2).csv")
st.sidebar.header("Data Exploration")

# Show raw data
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.write(df.head())

# Dataset Info
if st.sidebar.checkbox("Show Data Summary"):
    st.subheader("Data Summary")
    st.write(df.describe())
    st.write("Shape:", df.shape)

# Handle categorical variables
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Correlation heatmap
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ----- Linear Regression -----
st.sidebar.header("Linear Regression")
use_lr = st.sidebar.checkbox("Enable Linear Regression")

if use_lr:
    target = st.sidebar.selectbox("Select Target Variable", options=[col for col in df.columns if df[col].dtype in [np.float64, np.int64]])
    features = st.sidebar.multiselect("Select Feature Variables", options=[col for col in df.columns if col != target])

    if features:
        X = df[features].dropna()
        y = df.loc[X.index, target]  

        test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2)
        random_state = st.sidebar.number_input("Random State", value=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if st.sidebar.button("Train Linear Regression Model"):
            st.subheader("Linear Regression Performance")
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("**R² Score:**", r2_score(y_test, y_pred))
            st.write("**Mean Absolute Error (MAE):**", mean_absolute_error(y_test, y_pred))
            st.write("**Root Mean Squared Error (RMSE):**", np.sqrt(mean_squared_error(y_test, y_pred)))

            # Residual Plot
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax2)
            ax2.set_xlabel("Actual")
            ax2.set_ylabel("Predicted")
            ax2.set_title("Actual vs Predicted")
            st.pyplot(fig2)

            # Line Fit Plot (only for one feature)
            if len(features) == 1:
                st.subheader("Linear Regression Line Fit")
                fig_line, ax_line = plt.subplots()
                ax_line.scatter(X_test, y_test, color='blue', label='Actual')
                ax_line.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
                ax_line.set_xlabel(features[0])
                ax_line.set_ylabel(target)
                ax_line.set_title("Best Fit Line")
                ax_line.legend()
                st.pyplot(fig_line)

            # Try Prediction
            st.subheader("Try Your Own Prediction")
            input_data = []
            for feature in features:
                val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
                input_data.append(val)

            input_array = np.array(input_data).reshape(1, -1)
            pred = model.predict(input_array)[0]
            st.success(f"Predicted {target}: {pred:.2f}")
    else:
        st.warning("Please select at least one feature for Linear Regression.")

# ----- KMeans Clustering -----
st.sidebar.header("KMeans Clustering")
use_kmeans = st.sidebar.checkbox("Enable KMeans Clustering")

if use_kmeans:
    cluster_features = st.sidebar.multiselect("Select Features for Clustering", options=df.columns.tolist())

    if len(cluster_features) < 2:
        st.warning("Select at least 2 features for clustering.")
    else:
        num_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)

        # Handle missing values
        X_cluster = df[cluster_features].copy()
        X_cluster = X_cluster.fillna(X_cluster.mean())  # You can switch to .dropna() if preferred

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df_clustered = X_cluster.copy()
        df_clustered['Cluster'] = clusters

        st.subheader("Cluster Summary")
        st.dataframe(df_clustered['Cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'Cluster': 'Count'}))

        # Cluster Centers Table
        centers_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cluster_features)
        centers_df['Cluster'] = range(num_clusters)
        st.subheader("Cluster Centers")
        st.dataframe(centers_df)

        # 2D Scatterplot using first two features
        st.subheader("Cluster Visualization (2D)")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(
            x=X_cluster.iloc[:, 0], y=X_cluster.iloc[:, 1],
            hue=clusters, palette='viridis', ax=ax4
        )
        ax4.set_xlabel(cluster_features[0])
        ax4.set_ylabel(cluster_features[1])
        ax4.set_title("KMeans Clusters (First 2 Features)")
        st.pyplot(fig4)

        # Download clustered data
        st.subheader("Download Clustered Data")
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "clustered_sleep_efficiency.csv", "text/csv")
