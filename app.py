import streamlit as st
import json
import joblib
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="MovieLens ML Dashboard", layout="wide")

st.title("🎬 MovieLens ML Dashboard")

MODELS_DIR = Path("models")

# ---------- Load summary ----------
summary_path = MODELS_DIR / "ml_tasks_summary.json"

if not summary_path.exists():
    st.error("Model summary not found. Run training first.")
    st.stop()

with open(summary_path) as f:
    summary = json.load(f)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Classification",
    "👥 Clustering",
    "📉 PCA",
    "🎯 Recommendation"
])

# ---------- Classification ----------
with tab1:
    st.header("Rating Classification")

    clf = summary["classification"]
    st.metric("Accuracy", f"{clf['accuracy']:.4f}")

    st.subheader("Top Features")
    st.table(pd.DataFrame(clf["top_features"], columns=["Feature", "Importance"]))

# ---------- Clustering ----------
with tab2:
    st.header("User Clustering")

    cl = summary["clustering"]
    st.metric("Silhouette Score", f"{cl['silhouette_score']:.4f}")

    st.subheader("Cluster Sizes")
    st.bar_chart(cl["cluster_sizes"])

# ---------- PCA ----------
with tab3:
    st.header("Dimensionality Reduction (PCA)")

    pca = summary["pca"]
    st.metric("Explained Variance", f"{pca['cumulative_variance']:.4f}")

    st.subheader("Top PC1 Features")
    st.json(pca["top_pc1_features"])

# ---------- Recommendation ----------
with tab4:
    st.header("Recommendation System Summary")

    rec = summary["recommendation"]

    st.write("**Users:**", rec["n_users"])
    st.write("**Items:**", rec["n_items"])
    st.write("**Ratings:**", rec["n_ratings"])
    st.write("**Sparsity:**", f"{rec['sparsity']:.4f}")
