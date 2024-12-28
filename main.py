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

# Initialize BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# generate synthetic dataset with high-dimensional treatment
review_lengths = np.random.randint(5, 100, size=100)  # random lengths between 5 and 100 words
treatment_features = np.random.rand(100, 10)  # 10-dimensional synthetic treatment variable
texts = [
    f"This is a review of length {length} words with {'great' if i % 3 == 0 else 'poor'} content."
    for i, length in enumerate(review_lengths)
]

# extract BERT embeddings
def extract_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy())
    return np.array(embeddings)

original_embeddings = extract_embeddings(texts, tokenizer, model)

# t-SNE visualization
def prepare_plot_data(embeddings, treatment, label):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    return pd.DataFrame({
        "Dim1": reduced[:, 0],
        "Dim2": reduced[:, 1],
        "Treatment": np.argmax(treatment, axis=1) if treatment.ndim > 1 else treatment,
        "Type": label
    })

# DataFrame for original embeddings
original_df = prepare_plot_data(original_embeddings, treatment_features, "Original")

# embedding decomposition using Random Forest
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(treatment_features, original_embeddings)

# predict treatment-related components
predicted_treatment_components = regressor.predict(treatment_features)

# compute residuals to obtain treatment-agnostic embeddings
confounder_embeddings = original_embeddings - predicted_treatment_components

# validate that residualization worked by checking correlation
def calculate_correlation(embeddings, treatment):
    correlations = []
    for dim in range(embeddings.shape[1]):  # iterate over embedding dimensions
        for t_dim in range(treatment.shape[1]):  # iterate over treatment dimensions
            corr = np.corrcoef(embeddings[:, dim], treatment[:, t_dim])[0, 1]
            correlations.append(abs(corr))
    return np.mean(correlations)

correlation_original = calculate_correlation(original_embeddings, treatment_features)
correlation_confounder = calculate_correlation(confounder_embeddings, treatment_features)

print("Correlation with treatment (original embeddings):", correlation_original)
print("Correlation with treatment (confounder embeddings):", correlation_confounder)


confounder_df = prepare_plot_data(confounder_embeddings, treatment_features, "Confounder")


original_df.to_csv("original_embeddings.csv", index=False)
confounder_df.to_csv("confounder_embeddings.csv", index=False)


sns.set_theme(style="whitegrid", palette="coolwarm", font_scale=1.2)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

sns.scatterplot(
    data=original_df,
    x="Dim1", y="Dim2", hue="Treatment",
    palette="coolwarm", edgecolor="black", alpha=0.8,
    ax=axes[0]
)
axes[0].set_title("Confounder Embeddings (No Treatment Leakage)", fontsize=14)
axes[0].set_xlabel("Dimension 1", fontsize=12)
axes[0].set_ylabel("Dimension 2", fontsize=12)

sns.scatterplot(
    data=confounder_df,
    x="Dim1", y="Dim2", hue="Treatment",
    palette="coolwarm", edgecolor="black", alpha=0.8,
    ax=axes[1]
)
axes[1].set_title("Original Embeddings (Showing Treatment Leakage)", fontsize=14)
axes[1].set_xlabel("Dimension 1", fontsize=12)

plt.tight_layout()
plt.show()