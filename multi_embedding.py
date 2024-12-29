import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

models_to_test = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]

# Generate synthetic dataset
review_lengths = np.random.randint(5, 100, size=100)  # Random lengths between 5 and 100 words
treatment_features = np.random.rand(100, 10)  # 10-dimensional synthetic treatment variable

texts = [
    f"This is a review of length {length} words with {'great' if treatment_features[i, 0] > 0.5 else 'poor'} content."
    for i, length in enumerate(review_lengths)
]

# Extract embeddings
def extract_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy())
    return np.array(embeddings)

# Predict confounders and calculate R2
def predict_confounders(embeddings, confounders):
    model = LinearRegression()
    model.fit(embeddings, confounders)
    predictions = model.predict(embeddings)
    return r2_score(confounders, predictions)

# Calculate mutual information
def calculate_mutual_information(embeddings, confounders):
    mutual_info = []
    for i in range(confounders.shape[1]):
        mi = mutual_info_regression(embeddings, confounders[:, i])
        mutual_info.append(np.mean(mi))
    return np.mean(mutual_info)

# Variance comparison
def plot_r2_scores(r2_original, r2_confounder, model_name):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=["Original", "Confounder"], y=[r2_original, r2_confounder], alpha=0.7)
    plt.title(f"{model_name} R² Scores for Confounder Prediction")
    plt.ylabel("R² Score")
    plt.xlabel("Embedding Type")
    plt.tight_layout()
    plt.show()

def plot_mutual_information(mi_original, mi_confounder, model_name):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=["Original", "Confounder"], y=[mi_original, mi_confounder], alpha=0.7)
    plt.title(f"{model_name} Mutual Information Comparison")
    plt.ylabel("Mutual Information")
    plt.xlabel("Embedding Type")
    plt.tight_layout()
    plt.show()

# Process models
for model_name in models_to_test:
    print(f"Processing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Extract original embeddings
    original_embeddings = extract_embeddings(texts, tokenizer, model)

    # Fit RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(treatment_features, original_embeddings)

    # Compute residual embeddings
    predicted_treatment_components = regressor.predict(treatment_features)
    confounder_embeddings = original_embeddings - predicted_treatment_components

    # R² scores
    r2_original = predict_confounders(original_embeddings, treatment_features)
    r2_confounder = predict_confounders(confounder_embeddings, treatment_features)
    print(f"R² (Original): {r2_original}, R² (Confounder): {r2_confounder}")
    plot_r2_scores(r2_original, r2_confounder, model_name)

    # Mutual information
    mi_original = calculate_mutual_information(original_embeddings, treatment_features)
    mi_confounder = calculate_mutual_information(confounder_embeddings, treatment_features)
    print(f"Mutual Information (Original): {mi_original}, Mutual Information (Confounder): {mi_confounder}")
    plot_mutual_information(mi_original, mi_confounder, model_name)
