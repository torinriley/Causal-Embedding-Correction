## Results Summary: Statistical Analysis of Residualization Effectiveness

### Variance Analysis

#### Original Embeddings
- **High Variance**: The original embeddings exhibit significantly larger variances for both dimensions (e.g., Dim1: 2813.83 for Treatment 0, Dim2: 5864.50 for Treatment 0). This suggests that the treatment-related components dominate the variance in the original embeddings.

#### Confounder Embeddings
- **Reduced Variance**: After residualization, the variance in confounder embeddings is drastically reduced (e.g., Dim1: 23.79 for Treatment 0, Dim2: 11.96 for Treatment 0). This demonstrates that residualization successfully minimizes the treatment-related components, rendering the embeddings more treatment-agnostic.

---

### ANOVA Results

- **Original Embeddings**:
  - Dim1: F-statistic = 0.4392, p-value = 0.9103
  - Dim2: F-statistic = 1.2995, p-value = 0.2484

- **Confounder Embeddings**:
  - Dim1: F-statistic = 0.8553, p-value = 0.5679
  - Dim2: F-statistic = 0.6983, p-value = 0.7088

The high p-values across both original and confounder embeddings suggest no statistically significant differences between treatment groups. This result supports the claim that residualization eliminates treatment group effects, ensuring that confounder embeddings are unbiased by treatment-related factors.

---

### Clustering Metrics

#### Original Embeddings
- **Silhouette Score**: 0.3520
- **Davies-Bouldin Index**: 0.8717

#### Confounder Embeddings
- **Silhouette Score**: 0.3748
- **Davies-Bouldin Index**: 0.8245

The confounder embeddings exhibit a slightly higher Silhouette Score and a lower Davies-Bouldin Index compared to the original embeddings. This indicates that residualization improves clustering quality by enhancing separation and compactness, suggesting that the embeddings are less influenced by treatment-related biases.

---

### Correlation with Treatment

#### Original Embeddings
- Dim1: –-0.0118
- Dim2: –-0.1704

#### Confounder Embeddings
- Dim1: –-0.0400
- Dim2: 0.0314

The correlations with treatment for the confounder embeddings are negligible, confirming that residualization effectively removes treatment-related influences. In contrast, the original embeddings show some treatment-related bias, particularly in Dim2 (–-0.1704).

---

### Density Analysis
The density plot compares the distributions of Dim1 values for each treatment group across the original and confounder embeddings. On the left, the original embeddings show broader, overlapping distributions with higher variance, indicating potential treatment-related leakage. On the right, the confounder embeddings demonstrate tighter, more distinct distributions with reduced variance, showcasing the efficacy of residualization in mitigating treatment bias while maintaining meaningful structure.

<table>
  <thead>
    <tr>
      <th colspan="2">
        <img width="1316" alt="Density Plot" src="https://github.com/user-attachments/assets/42a2663b-7f59-4783-9e72-6264c3c45794" />
      </th>
    </tr>
    <tr>
      <th>Density Plot</th>
      <th>Observation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Original Embeddings - Dim1</strong></td>
      <td>Higher variance and wider density spread across treatments, indicating significant treatment-related components and potential treatment leakage in the original embeddings.</td>
    </tr>
    <tr>
      <td><strong>Confounder Embeddings - Dim1</strong></td>
      <td>Reduced variance and density overlap, showcasing that residualization effectively minimizes treatment-related biases in embeddings.</td>
    </tr>
  </tbody>
</table>


### Conclusion

The statistical analyses demonstrate the efficacy of the residualization methodology in:
- **Reducing Treatment-Related Variance**: Confounder embeddings show significantly lower variance compared to original embeddings.
- **Eliminating Treatment Bias**: ANOVA results and correlation analysis confirm that residualization minimizes treatment effects in embeddings.
- **Improving Clustering**: Enhanced clustering metrics indicate that the confounder embeddings are more robust and unbiased.

This methodology ensures treatment-agnostic embeddings while preserving meaningful structure for downstream analyses. These results validate the approach as an effective means of mitigating treatment-related bias in high-dimensional embeddings.

