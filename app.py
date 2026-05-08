import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Set page config
st.set_page_config(page_title="Movie SVD Explorer", layout="wide")

st.title("🎬 Movie Feature Analysis using SVD")
st.markdown("""
This app uses **Singular Value Decomposition (SVD)** to analyze the IMDB Top 1000 movies. 
It reduces complex movie features into 'latent factors' to find similar movies and visualize patterns.
""")

# --- Data Loading ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Look for the file locally or use uploader
data_file = "imdb_top_1000.csv"
if not os.path.exists(data_file):
    uploaded_file = st.sidebar.file_uploader("Upload 'imdb_top_1000.csv'", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        st.info("Please upload the 'imdb_top_1000.csv' file to begin.")
        st.stop()
else:
    df = load_data(data_file)

# --- Preprocessing ---
features_to_use = ['IMDB_Rating', 'Meta_score', 'Released_Year', 'No_of_Votes']
X = df[features_to_use].copy()

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Handle missing values
X = X.fillna(X.mean())

# Normalize
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# --- SVD Computation ---
@st.cache_resource
def perform_svd(data):
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    return U, S, Vt

U, S, Vt = perform_svd(X_normalized)
explained_variance_ratio = S**2 / np.sum(S**2)

# --- Sidebar: Movie Selection ---
st.sidebar.header("Find Similar Movies")
movie_list = df['Series_Title'].tolist()
selected_movie_name = st.sidebar.selectbox("Select a Movie", movie_list)
movie_idx = df[df['Series_Title'] == selected_movie_name].index[0]
num_recommendations = st.sidebar.slider("Number of recommendations", 1, 10, 5)

# --- Main Layout: Tabs ---
tab1, tab2, tab3 = st.tabs(["Recommendations", "SVD Visualization", "Variance Analysis"])

with tab1:
    st.subheader(f"Selected Movie: {selected_movie_name}")
    col1, col2, col3 = st.columns(3)
    movie_info = df.iloc[movie_idx]
    col1.metric("IMDB Rating", f"{movie_info['IMDB_Rating']}/10")
    col2.metric("Meta Score", f"{movie_info['Meta_score']}")
    col3.metric("Year", f"{movie_info['Released_Year']}")
    
    st.divider()
    
    # Recommendation Logic
    k = 2  # Using 2 components as in notebook
    X_reduced = U[:, :k]
    
    distances = np.linalg.norm(X_reduced - X_reduced[movie_idx], axis=1)
    similar_indices = np.argsort(distances)[1:num_recommendations+1]
    
    st.subheader("Top Similar Movies (based on SVD factors)")
    recs = df.iloc[similar_indices][['Series_Title', 'Released_Year', 'IMDB_Rating', 'Genre']]
    st.table(recs)

with tab2:
    st.subheader("Movies in 2D SVD Space")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = df['IMDB_Rating']
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colors, cmap='viridis', 
                         s=50, alpha=0.5, edgecolors='none')
    
    # Highlight selected movie
    ax.scatter(X_reduced[movie_idx, 0], X_reduced[movie_idx, 1], c='red', s=200, marker='*', label='Selected')
    
    ax.set_xlabel(f'Component 1 ({explained_variance_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'Component 2 ({explained_variance_ratio[1]*100:.1f}%)')
    plt.colorbar(scatter, label='IMDB Rating')
    st.pyplot(fig)
    st.write("Each dot represents a movie. Closer dots have more similar numerical features.")

with tab3:
    st.subheader("SVD Variance Analysis")
    col_a, col_b = st.columns(2)
    
    # Plot 1: Singular values
    fig1, ax1 = plt.subplots()
    ax1.bar(range(1, len(S)+1), S, color='steelblue')
    ax1.set_title("Importance of Components")
    col_a.pyplot(fig1)
    
    # Plot 2: Cumulative variance
    fig2, ax2 = plt.subplots()
    cumsum = np.cumsum(explained_variance_ratio) * 100
    ax2.plot(range(1, len(cumsum)+1), cumsum, 'o-', color='red')
    ax2.set_ylim(0, 105)
    ax2.set_title("Cumulative Explained Variance")
    col_b.pyplot(fig2)