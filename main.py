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

# generate synthetic dataset
review_lengths = np.random.randint(5, 100, size=100) 
treatment_features = np.random.rand(100, 10) 
outcome = (treatment_features[:, 0] * 2 + np.random.normal(0, 0.1, 100) > 1).astype(int) 

texts = [
    f"This is a review of length {length} words with {'great' if treatment_features[i, 0] > 0.5 else 'poor'} content."
    for i, length in enumerate(review_lengths)
]

# function to extract embeddings from text
def extract_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy())
    return np.array(embeddings)

# function to apply partial residualization
def partial_residualization(original_embeddings, predicted_treatment_components, alpha=0.5):
    """
    Perform partial residualization to retain some treatment information while reducing bias.
    """
    return original_embeddings - alpha * predicted_treatment_components

# function to calculate propensity scores and weights
def calculate_propensity_scores(treatment_features, outcome):
    model = LogisticRegression()
    model.fit(treatment_features, outcome)
    propensity_scores = model.predict_proba(treatment_features)[:, 1]
    weights = np.where(outcome == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))
    return propensity_scores, weights

# sensitivity Analysis for unobserved confounders
def sensitivity_analysis(treatment_features, outcome, propensity_scores, unobserved_factor_strength=0.1):
    
    """
    Simulate the impact of unobserved confounders by adding noise to the treatment features
    and checking the robustness of treatment effect estimates.
    """

    simulated_confounder = np.random.normal(0, unobserved_factor_strength, size=treatment_features.shape[0])
    treatment_features_with_unobserved = np.hstack([treatment_features, simulated_confounder.reshape(-1, 1)])
    new_propensity_scores, _ = calculate_propensity_scores(treatment_features_with_unobserved, outcome)
    print(f"Mean change in propensity scores due to unobserved confounder: {np.mean(np.abs(new_propensity_scores - propensity_scores))}")

# positivity Check
def check_positivity(propensity_scores, threshold=0.05):
    """
    Check for violations of the positivity assumption.
    """
    extreme_scores = np.sum((propensity_scores < threshold) | (propensity_scores > 1 - threshold))
    print(f"Number of extreme propensity scores (out of 100): {extreme_scores}")
    if extreme_scores > 0:
        print("Warning: Positivity assumption may be violated. Consider trimming extreme scores.")

# balancing Diagnostics
def perform_balancing_diagnostics(treatment_features, propensity_scores, weights):
    """
    Compare the distributions of covariates before and after weighting.
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

# uncertainty quantification with bootstrapping
def bootstrap_uncertainty(treatment_features, outcome, embeddings, n_bootstrap=100):
    """
    Use bootstrapping to estimate confidence intervals for treatment effects.
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

alpha = 0.5  # fraction of treatment components to remove
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


#mutual information between embeddings and treatment
mi_treatment = mutual_info_regression(original_embeddings, treatment_features[:, 0])
print(f"Mutual Information with Treatment: {mi_treatment.mean()}")

#mutual information between embeddings and outcome
mi_outcome = mutual_info_regression(original_embeddings, outcome)
print(f"Mutual Information with Outcome: {mi_outcome.mean()}")


print("Analysis complete. Results saved.")