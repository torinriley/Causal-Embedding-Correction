import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_selection import mutual_info_regression

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "bert-base-uncased"
np.random.seed(42)

review_lengths = np.random.randint(5, 100, size=100)
treatment_features = np.random.rand(100, 10)
outcome = (treatment_features[:, 0] * 2 + np.random.normal(0, 0.1, 100) > 1).astype(int)

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

def partial_residualization(original_embeddings, predicted_treatment_components, alpha=0.5):
    return original_embeddings - alpha * predicted_treatment_components

def calculate_propensity_scores(treatment_features, outcome):
    """
    Calculate propensity scores and weights for treatment features and outcome.

    Parameters:
    treatment_features (array-like): The features used to predict the treatment assignment.
    outcome (array-like): The binary outcome variable indicating treatment assignment (1 for treated, 0 for control).

    Returns:
    tuple: A tuple containing:
        - propensity_scores (array-like): The predicted propensity scores for each observation.
        - weights (array-like): The calculated weights based on the propensity scores.
    """
    model = LogisticRegression()
    model.fit(treatment_features, outcome)
    propensity_scores = model.predict_proba(treatment_features)[:, 1]
    weights = np.where(outcome == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))
    return propensity_scores, weights

def sensitivity_analysis(treatment_features, outcome, propensity_scores, unobserved_factor_strength=0.1):
    """
    Perform sensitivity analysis by introducing an unobserved confounder and evaluating its impact on propensity scores.

    Parameters:
    treatment_features (numpy.ndarray): The features related to the treatment.
    outcome (numpy.ndarray): The outcome variable.
    propensity_scores (numpy.ndarray): The initial propensity scores.
    unobserved_factor_strength (float, optional): The standard deviation of the unobserved confounder. Default is 0.1.

    Returns:
    None
    """
    simulated_confounder = np.random.normal(0, unobserved_factor_strength, size=treatment_features.shape[0])
    treatment_features_with_unobserved = np.hstack([treatment_features, simulated_confounder.reshape(-1, 1)])
    new_propensity_scores, _ = calculate_propensity_scores(treatment_features_with_unobserved, outcome)
    print(f"Mean change in propensity scores due to unobserved confounder: {np.mean(np.abs(new_propensity_scores - propensity_scores))}")

def check_positivity(propensity_scores, threshold=0.05):
    """
    Check the positivity assumption by identifying extreme propensity scores.

    The positivity assumption requires that all individuals have a non-zero probability of receiving each treatment.
    This function checks for extreme propensity scores that are either too close to 0 or 1, which may indicate a violation
    of the positivity assumption.

    Parameters:
    propensity_scores (array-like): An array of propensity scores.
    threshold (float, optional): The threshold for determining extreme scores. Scores below this value or above (1 - threshold) are considered extreme. Default is 0.05.

    Prints:
    Number of extreme propensity scores and a warning message if any extreme scores are found.
    """
    extreme_scores = np.sum((propensity_scores < threshold) | (propensity_scores > 1 - threshold))
    print(f"Number of extreme propensity scores (out of 100): {extreme_scores}")
    if extreme_scores > 0:
        print("Warning: Positivity assumption may be violated. Consider trimming extreme scores.")

def perform_balancing_diagnostics(treatment_features, propensity_scores, weights):
    """
    Perform balancing diagnostics by calculating and plotting standardized mean differences 
    before and after weighting.

    Parameters:
    treatment_features : numpy.ndarray
        A 2D array of treatment covariates/features with shape (n_samples, n_features).
    propensity_scores : numpy.ndarray
        A 1D array of propensity scores with shape (n_samples,).
    weights : numpy.ndarray
        A 1D array of weights with shape (n_samples,).

    Returns:
    None
    """
    standardized_mean_differences = []
    for i in range(treatment_features.shape[1]):
        original_mean_diff = np.mean(treatment_features[outcome == 1, i]) - np.mean(treatment_features[outcome == 0, i])
        weighted_mean_diff = (
            np.average(treatment_features[outcome == 1, i], weights=weights[outcome == 1]) -
            np.average(treatment_features[outcome == 0, i], weights=weights[outcome == 0])
        )
        standardized_mean_differences.append((original_mean_diff, weighted_mean_diff))

    smd_df = pd.DataFrame(standardized_mean_differences, columns=["Original", "Weighted"])
    smd_df.plot(kind="bar", figsize=(10, 6), rot=0)
    plt.title("Standardized Mean Differences Before and After Weighting")
    plt.xlabel("Covariates")
    plt.ylabel("Standardized Mean Difference")
    plt.tight_layout()
    plt.savefig("balancing_diagnostics.png")
    plt.show()

def bootstrap_uncertainty(treatment_features, outcome, embeddings, n_bootstrap=100):
    """
    Estimate the uncertainty of the causal effect using bootstrap resampling.

    Parameters:
    treatment_features (array-like): The features related to the treatment.
    outcome (array-like): The outcome variable indicating treatment effect (binary: 0 or 1).
    embeddings (array-like): The embeddings or representations of the data points.
    n_bootstrap (int, optional): The number of bootstrap samples to generate. Default is 100.

    Returns:
    None: Prints the estimated causal effect and its 95% confidence interval.
    """
    causal_effects = []
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(len(outcome), size=len(outcome), replace=True)
        sample_treatment_features = treatment_features[sample_indices]
        sample_outcome = outcome[sample_indices]
        sample_embeddings = embeddings[sample_indices]

        treated = sample_embeddings[sample_outcome == 1]
        control = sample_embeddings[sample_outcome == 0]
        causal_effects.append(np.mean(treated) - np.mean(control))

    lower_bound = np.percentile(causal_effects, 2.5)
    upper_bound = np.percentile(causal_effects, 97.5)
    print(f"Estimated causal effect: {np.mean(causal_effects):.4f} (95% CI: {lower_bound:.4f}, {upper_bound:.4f})")

print(f"Processing model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

original_embeddings = extract_embeddings(texts, tokenizer, model)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(treatment_features, original_embeddings)

predicted_treatment_components = regressor.predict(treatment_features)

alpha = 0.5
adjusted_embeddings = partial_residualization(original_embeddings, predicted_treatment_components, alpha=alpha)

propensity_scores, weights = calculate_propensity_scores(treatment_features, outcome)

sensitivity_analysis(treatment_features, outcome, propensity_scores)
check_positivity(propensity_scores)
perform_balancing_diagnostics(treatment_features, propensity_scores, weights)
bootstrap_uncertainty(treatment_features, outcome, adjusted_embeddings)

results_df = pd.DataFrame({
    "Propensity Scores": propensity_scores,
    "Weights": weights
})
results_df.to_csv("propensity_scores_and_weights.csv", index=False)

mi_treatment = mutual_info_regression(original_embeddings, treatment_features[:, 0])
mi_outcome = mutual_info_regression(original_embeddings, outcome)

plt.figure(figsize=(12, 6))
sns.lineplot(x=range(len(mi_treatment)), y=mi_treatment, label="Mutual Info with Treatment", marker="o", linestyle="--", color="blue")
sns.lineplot(x=range(len(mi_outcome)), y=mi_outcome, label="Mutual Info with Outcome", marker="s", linestyle="-", color="orange")
plt.title("Simplified Mutual Information Breakdown by Embedding Dimensions")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Mutual Information")
plt.legend()
plt.tight_layout()
plt.savefig("mutual_information_simplified.png")
plt.show()

print(f"Mutual Information with Treatment: {mi_treatment.mean()}")
print(f"Mutual Information with Outcome: {mi_outcome.mean()}")

print("Analysis complete. Results saved.")
