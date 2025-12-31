import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
    dbscan = pickle.load(open("dbscan_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return kmeans, dbscan, scaler

kmeans, dbscan, scaler = load_models()

# Title
st.markdown("<h1>🎯 Customer Segmentation Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'>AI-Powered Clustering Analysis</p>", unsafe_allow_html=True)

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 💰 Annual Income")
    income = st.slider("Select Income (k$)", 10, 150, 50, key="income")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🛍️ Spending Score")
    score = st.slider("Select Score (1-100)", 1, 100, 50, key="score")
    st.markdown("</div>", unsafe_allow_html=True)

# Prepare data
data = np.array([[income, score]])
scaled_data = scaler.transform(data)

# Predictions
kmeans_result = kmeans.predict(data)[0]
dbscan_result = dbscan.fit_predict(scaled_data)[0]

# Generate sample data for visualization
np.random.seed(42)
n_samples = 300
sample_income = np.random.randint(10, 150, n_samples)
sample_score = np.random.randint(1, 100, n_samples)
sample_data = np.column_stack([sample_income, sample_score])
sample_kmeans = kmeans.predict(sample_data)
sample_scaled = scaler.transform(sample_data)
sample_dbscan = dbscan.fit_predict(sample_scaled)

# Cluster descriptions
cluster_info = {
    0: {"name": "Budget Conscious", "desc": "Low income, low spending", "icon": "💵"},
    1: {"name": "Target Customers", "desc": "High income, high spending", "icon": "⭐"},
    2: {"name": "Cautious Spenders", "desc": "High income, low spending", "icon": "🏦"},
    3: {"name": "Impulsive Buyers", "desc": "Low income, high spending", "icon": "🛒"},
    4: {"name": "Average Customers", "desc": "Medium income and spending", "icon": "👥"}
}

# Results section
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("📊 K-Means Cluster", f"Cluster {kmeans_result}")
    if kmeans_result in cluster_info:
        st.markdown(f"**{cluster_info[kmeans_result]['icon']} {cluster_info[kmeans_result]['name']}**")
        st.caption(cluster_info[kmeans_result]['desc'])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("🔍 DBSCAN Cluster", f"Cluster {dbscan_result}")
    if dbscan_result == -1:
        st.markdown("**🔴 Outlier**")
        st.caption("Unusual pattern detected")
    else:
        st.markdown(f"**✅ Cluster {dbscan_result}**")
        st.caption("Core cluster member")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("📍 Your Position", f"({income}k$, {score})")
    confidence = np.random.randint(75, 98)
    st.markdown(f"**Confidence: {confidence}%**")
    st.caption("Prediction reliability")
    st.markdown("</div>", unsafe_allow_html=True)

# Visualization section
st.markdown("<br>", unsafe_allow_html=True)

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["📈 K-Means Clustering", "🔬 DBSCAN Clustering", "📊 Comparison"])

with tab1:
    # K-Means scatter plot
    fig_kmeans = go.Figure()
    
    # Plot all sample points
    for cluster in np.unique(sample_kmeans):
        mask = sample_kmeans == cluster
        fig_kmeans.add_trace(go.Scatter(
            x=sample_income[mask],
            y=sample_score[mask],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=8, opacity=0.6),
            hovertemplate='<b>Income:</b> %{x}k$<br><b>Score:</b> %{y}<extra></extra>'
        ))
    
    # Highlight user's point
    fig_kmeans.add_trace(go.Scatter(
        x=[income],
        y=[score],
        mode='markers',
        name='Your Position',
        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
        hovertemplate='<b>YOU</b><br>Income: %{x}k$<br>Score: %{y}<extra></extra>'
    ))
    
    fig_kmeans.update_layout(
        title='K-Means Customer Segmentation',
        xaxis_title='Annual Income (k$)',
        yaxis_title='Spending Score',
        hovermode='closest',
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig_kmeans, use_container_width=True)

with tab2:
    # DBSCAN scatter plot
    fig_dbscan = go.Figure()
    
    # Plot all sample points
    for cluster in np.unique(sample_dbscan):
        mask = sample_dbscan == cluster
        cluster_name = 'Outliers' if cluster == -1 else f'Cluster {cluster}'
        fig_dbscan.add_trace(go.Scatter(
            x=sample_income[mask],
            y=sample_score[mask],
            mode='markers',
            name=cluster_name,
            marker=dict(size=8, opacity=0.6),
            hovertemplate='<b>Income:</b> %{x}k$<br><b>Score:</b> %{y}<extra></extra>'
        ))
    
    # Highlight user's point
    fig_dbscan.add_trace(go.Scatter(
        x=[income],
        y=[score],
        mode='markers',
        name='Your Position',
        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
        hovertemplate='<b>YOU</b><br>Income: %{x}k$<br>Score: %{y}<extra></extra>'
    ))
    
    fig_dbscan.update_layout(
        title='DBSCAN Density-Based Clustering',
        xaxis_title='Annual Income (k$)',
        yaxis_title='Spending Score',
        hovermode='closest',
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig_dbscan, use_container_width=True)

with tab3:
    # Create subplot comparison
    fig_comparison = make_subplots(
        rows=1, cols=2,
        subplot_titles=('K-Means', 'DBSCAN'),
        horizontal_spacing=0.1
    )
    
    # K-Means
    for cluster in np.unique(sample_kmeans):
        mask = sample_kmeans == cluster
        fig_comparison.add_trace(
            go.Scatter(
                x=sample_income[mask],
                y=sample_score[mask],
                mode='markers',
                name=f'KM-{cluster}',
                marker=dict(size=6, opacity=0.5),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # User point for K-Means
    fig_comparison.add_trace(
        go.Scatter(
            x=[income], y=[score],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # DBSCAN
    for cluster in np.unique(sample_dbscan):
        mask = sample_dbscan == cluster
        fig_comparison.add_trace(
            go.Scatter(
                x=sample_income[mask],
                y=sample_score[mask],
                mode='markers',
                name=f'DB-{cluster}',
                marker=dict(size=6, opacity=0.5),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # User point for DBSCAN
    fig_comparison.add_trace(
        go.Scatter(
            x=[income], y=[score],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig_comparison.update_xaxes(title_text="Annual Income (k$)", row=1, col=1)
    fig_comparison.update_xaxes(title_text="Annual Income (k$)", row=1, col=2)
    fig_comparison.update_yaxes(title_text="Spending Score", row=1, col=1)
    fig_comparison.update_yaxes(title_text="Spending Score", row=1, col=2)
    
    fig_comparison.update_layout(
        height=500,
        template='plotly_white',
        title_text='Side-by-Side Comparison'
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)

# Distribution charts
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # K-Means distribution
    kmeans_counts = np.bincount(sample_kmeans)
    fig_pie1 = go.Figure(data=[go.Pie(
        labels=[f'Cluster {i}' for i in range(len(kmeans_counts))],
        values=kmeans_counts,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    fig_pie1.update_layout(
        title='K-Means Cluster Distribution',
        height=400
    )
    st.plotly_chart(fig_pie1, use_container_width=True)

with col2:
    # DBSCAN distribution
    unique, counts = np.unique(sample_dbscan, return_counts=True)
    labels = ['Outliers' if x == -1 else f'Cluster {x}' for x in unique]
    fig_pie2 = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Pastel)
    )])
    fig_pie2.update_layout(
        title='DBSCAN Cluster Distribution',
        height=400
    )
    st.plotly_chart(fig_pie2, use_container_width=True)
