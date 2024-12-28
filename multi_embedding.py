import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

models_to_test = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]

review_lengths = np.random.randint(5, 100, size=100)  # Random lengths between 5 and 100 words
treatment_features = np.random.rand(100, 10)  # 10-dimensional synthetic treatment variable

texts = [
    f"This is a review of length {length} words with {'great' if treatment_features[i, 0] > 0.5 else 'poor'} content."
    for i, length in enumerate(review_lengths)
]

def extract_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy())
    return np.array(embeddings)

def prepare_plot_data(embeddings, treatment, label):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    return pd.DataFrame({
        "Dim1": reduced[:, 0],
        "Dim2": reduced[:, 1],
        "Treatment": np.argmax(treatment, axis=1) if treatment.ndim > 1 else treatment,
        "Type": label
    })

def calculate_correlation(embeddings, treatment):
    correlations = []
    for dim in range(embeddings.shape[1]):  # Iterate over embedding dimensions
        for t_dim in range(treatment.shape[1]):  # Iterate over treatment dimensions
            corr = np.corrcoef(embeddings[:, dim], treatment[:, t_dim])[0, 1]
            correlations.append(abs(corr))
    return np.mean(correlations)

results = []

for model_name in models_to_test:
    print(f"Processing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    model_dir = os.path.join("results", model_name)
    os.makedirs(model_dir, exist_ok=True)

    original_embeddings = extract_embeddings(texts, tokenizer, model)

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(treatment_features, original_embeddings)

    predicted_treatment_components = regressor.predict(treatment_features)

    confounder_embeddings = original_embeddings - predicted_treatment_components

    correlation_original = calculate_correlation(original_embeddings, treatment_features)
    correlation_confounder = calculate_correlation(confounder_embeddings, treatment_features)

    results.append({
        "Model": model_name,
        "Correlation_Original": correlation_original,
        "Correlation_Confounder": correlation_confounder
    })

    original_df = prepare_plot_data(original_embeddings, treatment_features, f"{model_name} Original")
    confounder_df = prepare_plot_data(confounder_embeddings, treatment_features, f"{model_name} Confounder")

    original_df.to_csv(os.path.join(model_dir, f"{model_name}_original_embeddings.csv"), index=False)
    confounder_df.to_csv(os.path.join(model_dir, f"{model_name}_confounder_embeddings.csv"), index=False)

    sns.set_theme(style="whitegrid", palette="coolwarm", font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=original_df,
        x="Dim1", y="Dim2", hue="Treatment",
        palette="coolwarm", edgecolor="black", alpha=0.8
    )
    plt.title(f"{model_name} Original Embeddings (Showing Treatment Leakage)", fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{model_name}_original_embeddings_visualization.png"))
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=confounder_df,
        x="Dim1", y="Dim2", hue="Treatment",
        palette="coolwarm", edgecolor="black", alpha=0.8
    )
    plt.title(f"{model_name} Confounder Embeddings (Treatment-Agnostic)", fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{model_name}_confounder_embeddings_visualization.png"))
    plt.show()

results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv("model_performance_results.csv", index=False)
