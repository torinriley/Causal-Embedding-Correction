import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

models_to_test = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]

np.random.seed(42)  
review_lengths = np.random.randint(5, 100, size=100)  # Random lengths between 5 and 100 words
treatment_features = np.random.rand(100, 10)  # 10-dimensional synthetic treatment variable
outcome = (treatment_features[:, 0] * 2 + np.random.normal(0, 0.1, 100) > 1).astype(int)  # Binary outcome

texts = [
    f"This is a review of length {length} words with {'great' if treatment_features[i, 0] > 0.5 else 'poor'} content."
    for i, length in enumerate(review_lengths)
]

# Function to extract embeddings from text
def extract_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy())
    return np.array(embeddings)

# Function to apply partial residualization
def partial_residualization(original_embeddings, predicted_treatment_components, alpha=0.5):
    """
    Perform partial residualization to retain some treatment information while reducing bias.

    Parameters:
    - original_embeddings: The original high-dimensional embeddings.
    - predicted_treatment_components: The embeddings predicted from treatment.
    - alpha: Proportion of treatment-related components to remove (0 to 1).

    Returns:
    - Adjusted embeddings with reduced treatment bias.
    """
    return original_embeddings - alpha * predicted_treatment_components

# Function to calculate propensity scores and weights
def calculate_propensity_scores(treatment_features, outcome):
    """Calculate propensity scores using logistic regression and compute weights."""
    model = LogisticRegression()
    model.fit(treatment_features, outcome)
    propensity_scores = model.predict_proba(treatment_features)[:, 1]
    weights = outcome / propensity_scores + (1 - outcome) / (1 - propensity_scores)
    return propensity_scores, weights

# Correlation calculation
def calculate_correlation(embeddings, treatment):
    """Calculate mean absolute correlation between embeddings and treatment features."""
    correlations = []
    for dim in range(embeddings.shape[1]):
        for t_dim in range(treatment.shape[1]):
            corr = np.corrcoef(embeddings[:, dim], treatment[:, t_dim])[0, 1]
            correlations.append(abs(corr))
    return np.mean(correlations)


# Variance comparison
def plot_variance_comparison(original_var, adjusted_var, model_name):
    """Plot a bar chart comparing total variance before and after adjustment."""
    plt.figure(figsize=(8, 6))
    sns.barplot(x=["Original", "Adjusted"], y=[original_var.sum(), adjusted_var.sum()], alpha=0.7)
    plt.title(f"{model_name} Total Variance Comparison")
    plt.ylabel("Total Variance")
    plt.xlabel("Type")
    plt.tight_layout()
    plt.savefig(f"{model_name}_variance_comparison.png")
    plt.show()


def plot_correlation_comparison(mean_corr_original, mean_corr_adjusted, model_name):
    """Plot a bar chart comparing mean correlations before and after adjustment."""
    plt.figure(figsize=(8, 6))
    sns.barplot(x=["Original", "Adjusted"], y=[mean_corr_original, mean_corr_adjusted], alpha=0.7, palette="muted")
    plt.title(f"{model_name} Mean Correlation Comparison")
    plt.ylabel("Mean Correlation")
    plt.xlabel("Type")
    plt.tight_layout()
    plt.savefig(f"{model_name}_correlation_comparison.png")
    plt.show()

def plot_embeddings_scatter(embeddings, treatment, model_name, embedding_type):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    df = pd.DataFrame({
        "Dim1": reduced[:, 0],
        "Dim2": reduced[:, 1],
        "Treatment": np.argmax(treatment, axis=1) if treatment.ndim > 1 else treatment
    })
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Treatment", palette="coolwarm", alpha=0.8, edgecolor="black")
    plt.title(f"{model_name} {embedding_type} Embeddings Scatter Plot")
    plt.tight_layout()
    plt.savefig(f"{model_name}_{embedding_type}_embeddings_scatter.png")
    plt.show()

def plot_residuals_distribution(residuals, model_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color="blue")
    plt.title(f"{model_name} Confounder Embedding Residuals Distribution")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{model_name}_residuals_distribution.png")
    plt.show()

results = []

for model_name in models_to_test:
    print(f"Processing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    original_embeddings = extract_embeddings(texts, tokenizer, model)

    # Fit RandomForestRegressor to predict treatment components
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(treatment_features, original_embeddings)

    predicted_treatment_components = regressor.predict(treatment_features)

    alpha = 0.5  # Fraction of treatment components to remove
    adjusted_embeddings = partial_residualization(original_embeddings, predicted_treatment_components, alpha=alpha)

    propensity_scores, weights = calculate_propensity_scores(treatment_features, outcome)

    original_var = original_embeddings.var(axis=0)
    adjusted_var = adjusted_embeddings.var(axis=0)
    mean_corr_original = calculate_correlation(original_embeddings, treatment_features)
    mean_corr_adjusted = calculate_correlation(adjusted_embeddings, treatment_features)

    print(f"Model: {model_name}")
    print(f"Mean Correlation (Original): {mean_corr_original:.4f}")
    print(f"Mean Correlation (Adjusted): {mean_corr_adjusted:.4f}")
    print(f"Total Variance (Original): {original_var.sum():.4f}")
    print(f"Total Variance (Adjusted): {adjusted_var.sum():.4f}")
    print(f"Propensity Scores AUC: {roc_auc_score(outcome, propensity_scores):.4f}")

    plot_variance_comparison(original_var, adjusted_var, model_name)

    plot_correlation_comparison(mean_corr_original, mean_corr_adjusted, model_name)

    
    plot_embeddings_scatter(original_embeddings, treatment_features, model_name, "Original")
    plot_embeddings_scatter(adjusted_embeddings, treatment_features, model_name, "Adjusted")

    
    residuals = (original_embeddings - adjusted_embeddings).flatten()
    plot_residuals_distribution(residuals, model_name)


    results.append({
        "Model": model_name,
        "Mean_Correlation_Original": mean_corr_original,
        "Mean_Correlation_Adjusted": mean_corr_adjusted,
        "Total_Variance_Original": original_var.sum(),
        "Total_Variance_Adjusted": adjusted_var.sum(),
        "Propensity_Scores_AUC": roc_auc_score(outcome, propensity_scores)
    })

results_df = pd.DataFrame(results)
print(results_df)


results_df.to_csv("enhanced_partial_residualization_results.csv", index=False)

