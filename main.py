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

# Generate synthetic dataset with high-dimensional treatment
review_lengths = np.random.randint(5, 100, size=100)  # Random lengths between 5 and 100 words
treatment_features = np.random.rand(100, 10)  # 10-dimensional synthetic treatment variable
texts = [
    f"This is a review of length {length} words with {'great' if i % 3 == 0 else 'poor'} content."
    for i, length in enumerate(review_lengths)
]

# Function to extract BERT embeddings
def extract_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy())
    return np.array(embeddings)

original_embeddings = extract_embeddings(texts, tokenizer, model)

# Embedding decomposition using Random Forest
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(treatment_features, original_embeddings)

# Predict treatment-related components
predicted_treatment_components = regressor.predict(treatment_features)

# Compute residuals to obtain treatment-agnostic embeddings
confounder_embeddings = original_embeddings - predicted_treatment_components

# Function to prepare t-SNE visualization
def prepare_tsne_plot(embeddings, treatment):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    return pd.DataFrame({
        "Dim1": reduced[:, 0],
        "Dim2": reduced[:, 1],
        "Treatment": np.argmax(treatment, axis=1) if treatment.ndim > 1 else treatment
    })

# Prepare t-SNE DataFrame for confounder embeddings
confounder_tsne_df = prepare_tsne_plot(confounder_embeddings, treatment_features)

# Plot t-SNE visualization
sns.set_theme(style="whitegrid", palette="coolwarm", font_scale=1.2)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=confounder_tsne_df,
    x="Dim1", y="Dim2", hue="Treatment",
    palette="coolwarm", edgecolor="black", alpha=0.8
)
plt.title("Confounder Embeddings (t-SNE)", fontsize=16)
plt.xlabel("Dim1", fontsize=14)
plt.ylabel("Dim2", fontsize=14)
plt.legend(title="Treatment", fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.show()
