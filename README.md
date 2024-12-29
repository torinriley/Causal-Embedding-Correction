# Techniques to Reduce Treatment Leakage in Text Embeddings

## Problem Overview

### **Treatment Leakage**
Treatment leakage arises when representations (e.g., text embeddings) contain information about the treatment variable. This can distort causal analysis, making it difficult to disentangle the causal effect of treatment from confounding factors.

### **Embedding-Based Models**
Text embeddings, such as those generated by transformer models (e.g., BERT), provide high-dimensional vector representations of textual data. However, their high capacity may encode unintended correlations with treatment variables.

---

## Methodology

### **1. Extracting Text Embeddings**
We use the pretrained models **BERT-base-uncased**, **RoBERTa-base**, and **DistilBERT-base-uncased** to extract embeddings for textual data. The embeddings represent the semantic meaning of the input text in a high-dimensional space.

- **Step:** The `last_hidden_state` of the model is averaged across tokens to produce a single embedding vector for each text instance.

---

### **2. High-Dimensional Treatments**
To demonstrate the methodology, a synthetic high-dimensional treatment variable was generated with 10 independent features. These features simulate a realistic, complex treatment structure.

---

### **3. Embedding Decomposition Using Random Forest Regression**
To isolate and remove treatment-related information from embeddings, we use **Random Forest Regression**.

#### Steps:
1. **Train Regressor:**
   - A Random Forest model is trained with the high-dimensional treatment features as input and the original embeddings as the target.
   - The model captures nonlinear and high-dimensional relationships between the treatment and embeddings.
   
2. **Predict Treatment Components:**
   - The trained model predicts the treatment-related components of the embeddings.

3. **Partial Residualization:**
   - The predicted treatment components are scaled by a parameter \(\alpha\) and subtracted from the original embeddings.
   - This method balances treatment de-biasing and the retention of meaningful information.

4. **Propensity Scoring:**
   - To further enhance causal validity and address confounding bias, we integrate propensity scores into the methodology.

     **Steps:**
     
   - A logistic regression model estimates the propensity scores, which represent the probability of receiving treatment given observed covariates.
These scores are computed for each instance in the dataset based on the high-dimensional treatment features.
Evaluate Propensity Scores:

   - The AUC (Area Under the Curve) of the propensity model is calculated to validate its effectiveness.
High AUC values indicate that the model accurately captures covariate information.

   - The propensity scores are used to create inverse propensity weights. These weights adjust for covariate imbalances between treatment and control groups, mitigating confounding bias.

---

### **4. Validation and Visualizations**

#### **Correlation Analysis**
We calculate the mean absolute correlation between the embedding dimensions and the treatment dimensions to assess treatment leakage:
- **Original Embeddings:** Higher correlations with treatment.
- **Adjusted Embeddings:** Reduced correlations, confirming the removal of treatment-related signals.

#### **Variance Comparison**
A comparison of total variance in the embeddings before and after partial residualization demonstrates the extent to which treatment-related variance is removed while preserving overall variability.

#### **Embedding Scatter Plots**
`t-SNE` scatter plots visualize the structure of embeddings:
- **Original Embeddings:** Show strong clustering based on treatment, indicating leakage.
- **Adjusted Embeddings:** Scatter with reduced clustering, indicating treatment-agnostic embeddings.

#### **Distribution of Residuals**
Histograms of residuals confirm the removal of treatment-related components, as residuals are tightly centered around zero.

---

## Results

- **Correlation Metrics:**
  - Original Embeddings: Moderate correlation with treatment variables.
  - Adjusted Embeddings: Significantly reduced correlations, validating the effectiveness of partial residualization.

- **Variance Analysis:**
  - The adjusted embeddings retain a meaningful proportion of variance while eliminating treatment-related components.

- **Visualizations:**
  - Scatter plots of original and adjusted embeddings and the distribution of residuals provide qualitative and quantitative evidence of the methodology's success.

---

# Embedding Analysis Visualizations


## Variance and Correlation Comparison
<table>
    <tr>
        <th>Variance Comparison</th>
        <th>Correlation Comparison</th>
    </tr>
    <tr>
        <td>
           <img width="978" alt="Screenshot 2024-12-29 at 12 05 05 AM" src="https://github.com/user-attachments/assets/0aca8d2a-02ba-4189-9759-e0e20ffb7747" />
        </td>
        <td>
           <img width="986" alt="Screenshot 2024-12-29 at 12 04 56 AM" src="https://github.com/user-attachments/assets/1ac5d6bb-a086-4296-9620-19c0774af69f" />
        </td>
    </tr>
</table>



## Embedding Scatter Plots
<table>
    <tr>
        <th>Original Embeddings</th>
        <th>Adjusted Embeddings</th>
    </tr>
    <tr>
        <td>
            <img src="https://github.com/user-attachments/assets/78a37edd-59fc-4093-8037-cd81daef0e20" alt="Original Embeddings" width="789"/>
        </td>
        <td>
           <img width="789" alt="Screenshot 2024-12-28 at 11 40 40 PM" src="https://github.com/user-attachments/assets/6dee08a4-b8e5-4830-8b5b-1c9d818588d1" />
        </td>
    </tr>
</table>

---

## Conclusion

This project demonstrates a robust methodology to mitigate treatment leakage in text embeddings using the following:
1. **Random Forest Regression** to model and remove treatment-related components.
2. **Partial Residualization** to balance treatment de-biasing with the retention of meaningful data relationships.
3. **Validation** through statistical metrics and robust visualizations.

By ensuring embeddings are treatment-agnostic but not treatment-blind, this approach enhances the reliability of causal inference models, enabling more accurate estimation of causal effects.

For complete results, see: [Results Summary](https://github.com/torinriley/Causal-Embedding-Correction/blob/main/results/Results_Summary.md)

