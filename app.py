import streamlit as st
import numpy as np
import pickle

# Load models
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
dbscan = pickle.load(open("dbscan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Segmentation App")
st.write("K-Means & DBSCAN Clustering")

income = st.slider("Annual Income (k$)", 10, 150, 50)
score = st.slider("Spending Score", 1, 100, 50)

data = np.array([[income, score]])

# KMeans
kmeans_result = kmeans.predict(data)[0]

# DBSCAN
scaled_data = scaler.transform(data)
dbscan_result = dbscan.fit_predict(scaled_data)[0]

st.subheader("Results")
st.write(f"✅ K-Means Cluster: {kmeans_result}")
st.write(f"✅ DBSCAN Cluster: {dbscan_result}")
