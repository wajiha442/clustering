import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from io import BytesIO
import warnings

# Suppress version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Load models
try:
    kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))
    dbscan_model = pickle.load(open("dbscan_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    models_loaded = True
except:
    st.warning("⚠️ Model files not found. Using demo mode.")
    models_loaded = False
    scaler = None

# Title
st.title("🎯 Customer Segmentation Dashboard")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Number of clusters
    n_clusters = st.slider("Number of K-Means Clusters", 2, 8, 5)
    
    # Sample size
    sample_size = st.slider("Sample Data Size", 100, 1000, 300, step=50)
    
    st.markdown("---")
    st.header("📊 Your Customer Data")
    
    income = st.slider("Annual Income (k$)", 10, 150, 50)
    score = st.slider("Spending Score (1-100)", 1, 100, 50)
    
    st.markdown("---")
    
    # Show input values
    st.metric("Income", f"${income}k")
    st.metric("Spending Score", score)

# Generate sample data
np.random.seed(42)

# Create realistic customer segments
centers = [
    [20, 20],   # Low income, low spending
    [120, 80],  # High income, high spending
    [120, 20],  # High income, low spending
    [30, 80],   # Low income, high spending
    [70, 50]    # Average
]

sample_data = []
for center in centers[:n_clusters]:
    n = sample_size // n_clusters
    data = np.random.randn(n, 2) * 15 + center
    data[:, 0] = np.clip(data[:, 0], 10, 150)
    data[:, 1] = np.clip(data[:, 1], 1, 100)
    sample_data.append(data)

sample_data = np.vstack(sample_data)
df_sample = pd.DataFrame(sample_data, columns=['Income', 'SpendingScore'])

# Train K-Means on sample data
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_sample['KMeans_Cluster'] = kmeans.fit_predict(sample_data)

# Predict for user input
user_data = np.array([[income, score]])
user_kmeans = kmeans.predict(user_data)[0]

if models_loaded and scaler is not None:
    user_scaled = scaler.transform(user_data)
    user_dbscan = dbscan_model.fit_predict(user_scaled)[0]
else:
    user_dbscan = -1

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>K-Means Cluster</h3>
        <h1 style='color: white; margin: 10px 0;'>Cluster {}</h1>
        <p style='color: white; margin: 0;'>Based on {} clusters</p>
    </div>
    """.format(user_kmeans, n_clusters), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>DBSCAN Cluster</h3>
        <h1 style='color: white; margin: 10px 0;'>{}</h1>
        <p style='color: white; margin: 0;'>{}</p>
    </div>
    """.format(
        "Outlier" if user_dbscan == -1 else f"Cluster {user_dbscan}",
        "Anomaly detected" if user_dbscan == -1 else "Normal pattern"
    ), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 20px; border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>Your Position</h3>
        <h1 style='color: white; margin: 10px 0;'>${}k, {}</h1>
        <p style='color: white; margin: 0;'>Income & Score</p>
    </div>
    """.format(income, score), unsafe_allow_html=True)

st.markdown("---")

# Visualization tabs
tab1, tab2, tab3 = st.tabs(["📈 K-Means Clustering", "📊 Data Preview", "📉 Cluster Analysis"])

with tab1:
    st.subheader(f"K-Means Clustering with {n_clusters} Clusters")
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each cluster with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_data = df_sample[df_sample['KMeans_Cluster'] == i]
        ax.scatter(cluster_data['Income'], cluster_data['SpendingScore'], 
                  c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
    
    # Plot user point
    ax.scatter(income, score, c='red', marker='*', s=500, 
              edgecolors='black', linewidths=2, label='Your Position', zorder=5)
    
    # Plot cluster centers
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', 
              s=200, edgecolors='white', linewidths=2, label='Centroids')
    
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Spending Score', fontsize=12)
    ax.set_title(f'Customer Segmentation - {n_clusters} Clusters', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    # Show cluster centers
    st.subheader("📍 Cluster Centers")
    centers_df = pd.DataFrame(centers, columns=['Income (k$)', 'Spending Score'])
    centers_df.index = [f'Cluster {i}' for i in range(n_clusters)]
    centers_df = centers_df.round(2)
    st.dataframe(centers_df, use_container_width=True)

with tab2:
    st.subheader("📋 Sample Data Preview")
    
    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df_sample))
    with col2:
        st.metric("Avg Income", f"${df_sample['Income'].mean():.1f}k")
    with col3:
        st.metric("Avg Score", f"{df_sample['SpendingScore'].mean():.1f}")
    with col4:
        st.metric("Your Cluster", f"Cluster {user_kmeans}")
    
    st.markdown("---")
    
    # Show data table
    display_df = df_sample.copy()
    display_df['Income'] = display_df['Income'].round(1)
    display_df['SpendingScore'] = display_df['SpendingScore'].round(1)
    display_df = display_df.rename(columns={
        'Income': 'Income (k$)',
        'SpendingScore': 'Spending Score',
        'KMeans_Cluster': 'Cluster'
    })
    
    # Highlight user's cluster
    st.write(f"**Showing all {len(display_df)} customers (Your cluster: {user_kmeans})**")
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("📊 Cluster Distribution & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster size distribution
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        cluster_counts = df_sample['KMeans_Cluster'].value_counts().sort_index()
        colors_bar = [colors[i] for i in range(n_clusters)]
        ax1.bar(range(n_clusters), cluster_counts.values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Cluster', fontsize=12)
        ax1.set_ylabel('Number of Customers', fontsize=12)
        ax1.set_title('Customer Distribution Across Clusters', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(n_clusters))
        ax1.set_xticklabels([f'C{i}' for i in range(n_clusters)])
        ax1.grid(axis='y', alpha=0.3)
        
        # Highlight user's cluster
        ax1.bar(user_kmeans, cluster_counts.values[user_kmeans], 
               color='red', alpha=0.8, edgecolor='black', linewidth=2)
        
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        # Cluster statistics
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        cluster_stats = df_sample.groupby('KMeans_Cluster').agg({
            'Income': 'mean',
            'SpendingScore': 'mean'
        }).round(1)
        
        x = np.arange(n_clusters)
        width = 0.35
        
        ax2.bar(x - width/2, cluster_stats['Income'], width, 
               label='Avg Income (k$)', color='steelblue', alpha=0.7)
        ax2.bar(x + width/2, cluster_stats['SpendingScore'], width, 
               label='Avg Spending Score', color='coral', alpha=0.7)
        
        ax2.set_xlabel('Cluster', fontsize=12)
        ax2.set_ylabel('Average Value', fontsize=12)
        ax2.set_title('Average Income & Spending by Cluster', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'C{i}' for i in range(n_clusters)])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig2)
        plt.close()
    
    # Detailed cluster statistics table
    st.subheader("📈 Detailed Cluster Statistics")
    cluster_details = df_sample.groupby('KMeans_Cluster').agg({
        'Income': ['mean', 'min', 'max'],
        'SpendingScore': ['mean', 'min', 'max'],
        'KMeans_Cluster': 'count'
    }).round(2)
    
    cluster_details.columns = ['Avg Income', 'Min Income', 'Max Income', 
                               'Avg Score', 'Min Score', 'Max Score', 'Count']
    cluster_details.index = [f'Cluster {i}' for i in range(n_clusters)]
    
    st.dataframe(cluster_details, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>Tip:</strong> Adjust the number of clusters in the sidebar to see how segmentation changes!</p>
    <p>🎯 Your customer profile is highlighted in red on the scatter plot</p>
</div>
""", unsafe_allow_html=True)
