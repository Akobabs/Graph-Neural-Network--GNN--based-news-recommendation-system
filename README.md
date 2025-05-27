---

# 📰 GNN-Based News Recommendation System

A personalized news recommendation system built using **Graph Neural Networks (GNNs)** on the **Microsoft News Dataset (MIND)**. This project leverages **Graph Convolutional Networks (GCNs)** via **PyTorch Geometric** to model user-news interactions and generate top-k personalized recommendations.

The frontend is built with **Streamlit**, featuring a clean, interactive, and aesthetic interface. The system is designed for local deployment and scalable architecture, with future readiness for **cloud-based deployment** using Flask and Docker.

---

## 🚀 Features

* **GNN Model**

  * Built with **PyTorch Geometric**
  * Models user-news relationships via GCN layers

* **Dataset**

  * Uses **MIND-small** dataset from Microsoft
  * Includes `news.tsv` and `behaviors.tsv`

* **Frontend (Streamlit)**

  * Interactive dropdown for user selection
  * Styled recommendation cards with title, category, abstract
  * Like/Dislike feedback saved to `feedback.csv`
  * Custom theme: blue accents, Roboto font, hover effects

* **Evaluation Metrics**

  * Precision\@K, Recall\@K, F1 Score\@K
  * ROC-AUC and Mean Squared Error (MSE)

* **Scalability**

  * Modular backend functions ready for Flask API
  * Docker-based containerization supported

---

## 📁 Project Structure

```
Graph-Neural-Network--GNN--based-news-recommendation-system/
│
├── app.py                         # Streamlit frontend script
├── NewsRecommendation.ipynb      # Jupyter notebook for training and evaluation
├── data/
│   ├── news.tsv                  # News metadata
│   ├── behaviors.tsv             # User interactions
│   ├── feedback.csv              # User feedback (Like/Dislike)
├── models/
│   ├── news_recommender_model.pth     # Trained GNN model
│   ├── entity_embedding.vec           # Pretrained entity embeddings
│   ├── relation_embedding.vec         # (Optional) Relation embeddings
├── static/
│   ├── style.css                 # Custom frontend styling
├── .streamlit/
│   ├── config.toml              # Custom Streamlit theme
├── .gitignore
└── README.md                    # Project documentation
```

---

## 🧰 Prerequisites

* Python 3.9 or higher
* Recommended: GPU for training, CPU sufficient for running the frontend

### Required Python Libraries

```bash
pip install streamlit torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric pandas numpy scikit-learn
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Akobabs/Graph-Neural-Network--GNN--based-news-recommendation-system.git
cd Graph-Neural-Network--GNN--based-news-recommendation-system
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

(See list above)

### 4. Download & Place Dataset

* Download [MIND-small](https://msnews.github.io/) from Microsoft.
* Place `news.tsv` and `behaviors.tsv` in the `data/` directory.

### 5. Prepare Model & Embeddings

Ensure these files exist in the `models/` directory:

* `news_recommender_model.pth`
* `entity_embedding.vec`

---

## 💻 Running the Application

```bash
.\venv\Scripts\activate  # Activate virtual environment
streamlit run app.py --server.fileWatcherType none
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## 🧪 Training & Evaluation

Use the Jupyter notebook `NewsRecommendation.ipynb` to:

* Perform EDA on the MIND dataset
* Train the GCN-based recommendation model
* Evaluate using metrics like Precision\@K, Recall\@K, ROC-AUC, F1 Score, and MSE

---

## ☁️ Scalability & Deployment

### Flask API Endpoint (example)

```python
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    user_idx = user_ids_map[user_id]
    recommendations = get_recommendations(model, data, user_idx, news, user_ids_map)
    return jsonify(recommendations[['title', 'category', 'abstract']].to_dict('records'))
```

### Dockerfile Example

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.fileWatcherType", "none"]
```

### Cloud Deployment

* Use **AWS ECS**, **Heroku**, or **Azure App Service**
* Transition CSV/SQLite to PostgreSQL
* Add graph sampling for large datasets like MIND-large

---

## 🧩 Troubleshooting

| Issue                       | Solution                                                                    |
| --------------------------- | --------------------------------------------------------------------------- |
| `FileNotFoundError`         | Ensure required files exist in `data/`, `models/`, `static/`, `.streamlit/` |
| `Tensor Creation Warning`   | Use `np.array` instead of list in `edge_index`                              |
| `RuntimeError with PyTorch` | Use `--server.fileWatcherType none`                                         |
| `Slow Graph Load`           | Subsample MIND dataset or use `NeighborSampler`                             |

---

## 🎨 UI Aesthetic

* **Header:** Blue background with 📰 emoji and white text
* **Sidebar:** Dropdown for user selection, app info
* **Cards:** White cards with category, title, abstract, and shadows
* **Feedback:** Radio buttons (Like/Dislike) + styled submit button
* **Font & Theme:** Roboto, #4A90E2 blue, soft grays, hover effects

---

## 🌱 Future Improvements

* Use `relation_embedding.vec` as edge attributes
* Add filters (e.g., by news category) in the UI
* Show evaluation metrics (Precision\@K, etc.) live in the frontend
* Fuse article images for multimodal GNN training
* Full cloud deployment with Flask API and PostgreSQL

---

## 🤝 Contributing

```bash
# Fork and clone this repository
git checkout -b feature/your-feature
# Make your changes
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a Pull Request on GitHub
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgments

* [Microsoft MIND Dataset](https://msnews.github.io/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
* [Streamlit](https://streamlit.io/) for the frontend framework

---
