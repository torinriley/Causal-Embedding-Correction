# Techniques to Reduce Treatment Leakage in Text Embeddings

## Problem Overview

### **Treatment Leakage**
Treatment leakage arises when representations (e.g., text embeddings) contain information about the treatment variable. This can distort causal analysis, making it difficult to disentangle the causal effect of treatment from confounding factors.

### **Embedding-Based Models**
Text embeddings, such as those generated by BERT, provide high-dimensional vector representations of textual data. However, their high capacity may encode unintended correlations with treatment variables.

---

## Methodology

### **1. Extracting Text Embeddings**
We use the pretrained **BERT-base-uncased** model to extract embeddings for textual data. The embeddings represent the semantic meaning of the input text in a high-dimensional space.

- **Step:** The `last_hidden_state` of the BERT model is averaged across tokens to produce a single embedding vector for each text instance.

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

3. **Residualization:**
   - The predicted treatment components are subtracted from the original embeddings to compute the residuals.
   - These residuals are referred to as **treatment-agnostic embeddings**, as they no longer encode treatment information.

---

### **4. Validating the Residualization Process**
To evaluate the success of the residualization:
1. **Correlation Analysis:**
   - We calculate the mean absolute correlation between the embedding dimensions and the treatment dimensions.
   - **Result:**
     - Original Embeddings: High correlation with treatment.
     - Confounder Embeddings: Near-zero correlation, indicating successful removal of treatment-related signals.
   
2. **t-SNE Visualization:**
   - **t-SNE** (t-Distributed Stochastic Neighbor Embedding) is applied to visualize the embeddings in two dimensions.
   - Graphs demonstrate:
     - **Original Embeddings:** Strong clustering based on treatment, showing leakage.
     - **Confounder Embeddings:** Clusters are no longer aligned with treatment, confirming treatment-agnostic embeddings.

---

## Results

- **Correlation Metrics:**
  - Original Embeddings: High correlation with treatment variables.
  - Confounder Embeddings: Near-zero correlation, confirming the efficacy of the residualization process.

- **Visualizations:**
  - Side-by-side scatter plots of the original and confounder embeddings visually validate the removal of treatment leakage.
    
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/948b1825-5d10-4c8a-886f-40f8b56f0215" alt="Confounder Embeddings Visualization" width="400"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/78a37edd-59fc-4093-8037-cd81daef0e20" alt="Original Embeddings Visualization" width="400"/>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">Confounder Embeddings Visualization</td>
    <td style="text-align: center;">Original Embeddings Visualization</td>
  </tr>
</table>



- **Distribution of Confounder Embedding Residuals:**
  
  - These histograms show the distribution of residual values in the confounder embeddings after removing treatment-related components using a Random Forest regressor. The residuals are tightly centered around zero, indicating minimal correlation with the treatment features. This result supports the efficacy of the technique by demonstrating that the treatment-related signal has been effectively removed, ensuring the embeddings are treatment-agnostic and suitable for unbiased causal analysis.
    
<table>
  <tr>
    <td>
      <img width="300" alt="Screenshot 2024-12-28 at 5 08 47 PM" src="https://github.com/user-attachments/assets/9e5f6adb-5c49-4166-8da1-0c74d4dc0430" />
    </td>
    <td>
      <img width="300" alt="Screenshot 2024-12-28 at 5 09 05 PM" src="https://github.com/user-attachments/assets/26328cec-3977-419c-bb6e-e3f67f1faaa2" />
    </td>
    <td>
      <img width="300" alt="Screenshot 2024-12-28 at 5 09 20 PM" src="https://github.com/user-attachments/assets/95ca97ce-c09a-4e05-9f81-4c932b833f51" />
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">bert-base-uncased</td>
    <td style="text-align: center;">roberta-base</td>
    <td style="text-align: center;">distilbert-base-uncased</td>
  </tr>
</table>


---

## Conclusion

This project demonstrates a robust methodology to mitigate treatment leakage in text embeddings using the following:
1. **Random Forest Regression** to model and remove treatment-related components.
2. **Residualization** to compute treatment-agnostic embeddings.
3. **Validation** through statistical metrics and visualizations.

By ensuring embeddings are treatment-agnostic, this approach enhances the reliability of causal inference models, enabling more accurate estimation of causal effects.

---
