import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Apply custom CSS with error handling
css_path = "static/style.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("CSS file not found at 'static/style.css'. Using default styling.")

# GNN Model Definition
class GNNRecommender(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNRecommender, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Load Entity Embeddings
@st.cache_data
def load_entity_embeddings(file_path):
    embeddings = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                entity_id = values[0]
                vector = [float(x) for x in values[1:]]
                embeddings[entity_id] = np.array(vector)
        return embeddings
    except Exception as e:
        st.warning(f"Error loading entity embeddings: {e}. Using TF-IDF features.")
        return None

# Load Data and Preprocess
@st.cache_data
def load_data(news_path, behaviors_path, entity_emb_path=None):
    try:
        news = pd.read_csv(news_path, sep='\t', 
                           names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 
                                  'title_entities', 'abstract_entities'])
        behaviors = pd.read_csv(behaviors_path, sep='\t', 
                               names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    except FileNotFoundError as e:
        st.error(f"Dataset file not found: {e}")
        return None, None, None, None, None

    if entity_emb_path and os.path.exists(entity_emb_path):
        entity_embeddings = load_entity_embeddings(entity_emb_path)
        if entity_embeddings:
            user_features = []
            for uid in behaviors['user_id'].unique():
                emb = entity_embeddings.get(uid, np.zeros(100))
                user_features.append(emb)
            news_features = []
            for nid in news['news_id']:
                emb = entity_embeddings.get(nid, np.zeros(100))
                news_features.append(emb)
            node_features = np.vstack([user_features, news_features])
        else:
            tfidf = TfidfVectorizer(max_features=100)
            news_features = tfidf.fit_transform(news['title'].fillna('')).toarray()
            user_counts = behaviors.groupby('user_id').size().values.reshape(-1, 1) / 100.0
            user_features = np.hstack([user_counts, np.zeros((len(user_counts), 99))])
            node_features = np.vstack([user_features, news_features])
    else:
        tfidf = TfidfVectorizer(max_features=100)
        news_features = tfidf.fit_transform(news['title'].fillna('')).toarray()
        user_counts = behaviors.groupby('user_id').size().values.reshape(-1, 1) / 100.0
        user_features = np.hstack([user_counts, np.zeros((len(user_counts), 99))])
        node_features = np.vstack([user_features, news_features])

    node_features = torch.tensor(node_features, dtype=torch.float)

    interactions = behaviors.explode('history')[['user_id', 'history']].dropna()
    user_ids_map = {uid: i for i, uid in enumerate(behaviors['user_id'].unique())}
    news_ids_map = {nid: i + len(user_ids_map) for i, nid in enumerate(news['news_id'].unique())}
    interactions = interactions[interactions['history'].isin(news_ids_map)]
    user_ids = interactions['user_id'].map(user_ids_map).values
    news_ids = interactions['history'].map(news_ids_map).values
    edge_index = torch.tensor(np.array([user_ids, news_ids]), dtype=torch.long)  # Optimized
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data = Data(x=node_features, edge_index=edge_index)

    return data, news, behaviors, user_ids_map, news_ids_map

# Load Model
@st.cache_resource
def load_model(model_path):
    try:
        model = GNNRecommender(in_channels=100, hidden_channels=16, out_channels=8)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Generate Recommendations
def get_recommendations(model, data, user_idx, news, user_ids_map, top_k=5):
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        user_embedding = out[user_idx]
        news_embeddings = out[len(user_ids_map):]
        scores = torch.matmul(news_embeddings, user_embedding)
        top_indices = torch.topk(scores, k=top_k).indices
        recommended_news = news.iloc[top_indices.numpy()]
    return recommended_news

# Streamlit App
def main():
    # Header
    st.markdown(
        """
        <div class="header">
            <h1>📰 News Recommendation System</h1>
            <p>Personalized news powered by Graph Neural Networks</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.title("User Settings")
    st.sidebar.markdown("Select a user to get personalized news recommendations.")

    # File paths
    news_path = 'data/news.tsv'
    behaviors_path = 'data/behaviors.tsv'
    model_path = 'models/news_recommender_model.pth'
    entity_emb_path = 'models/entity_embedding.vec'
    feedback_path = 'data/feedback.csv'

    # Load data and model
    data, news, behaviors, user_ids_map, news_ids_map = load_data(news_path, behaviors_path, entity_emb_path)
    if data is None:
        return
    model = load_model(model_path)
    if model is None:
        return

    # User selection
    user_id = st.sidebar.selectbox("Select User ID", list(user_ids_map.keys()))
    user_idx = user_ids_map[user_id]

    # Recommendation button
    if st.sidebar.button("Get Recommendations"):
        recommendations = get_recommendations(model, data, user_idx, news, user_ids_map)
        st.subheader("Recommended News Articles")
        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <div class="card-title">{row['title']}</div>
                    <div class="card-category">{row['category']}</div>
                    <div class="card-abstract">{row['abstract']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Feedback section
        with st.expander("Provide Feedback", expanded=True):
            feedback = st.radio("Did you like these recommendations?", ["Like", "Dislike"], horizontal=True)
            if st.button("Submit Feedback"):
                feedback_data = pd.DataFrame({
                    'user_id': [user_id],
                    'timestamp': [pd.Timestamp.now()],
                    'feedback': [feedback]
                })
                try:
                    if os.path.exists(feedback_path):
                        feedback_data.to_csv(feedback_path, mode='a', header=False, index=False)
                    else:
                        feedback_data.to_csv(feedback_path, index=False)
                    st.success("Feedback submitted!")
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")

    # App info
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **About**  
        This app uses a Graph Neural Network to recommend news articles based on user preferences.  
        Built with Streamlit and PyTorch Geometric.
        """
    )

if __name__ == "__main__":
    main()