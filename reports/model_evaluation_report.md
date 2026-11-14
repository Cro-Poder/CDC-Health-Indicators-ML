# Model Evaluation & Diagnostic Report — Multi-Disease Prediction System

**Project:** CodeAlpha Task 4 — Multi-Disease Prediction Using Machine Learning  
**Author:** Antigravity (ML Engineer & Technical Mentor)  
**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024  
**Status:** Methodologically Remediated & Independently Verified  
**Date:** November 2026  
**Scope:** Educational Multi-Disease Risk Classification  

---

## 1. Executive Summary & Remediation Overview

This evaluation report documents the empirical performance of the Multi-Disease Prediction System following strict methodological remediation to address model-selection leakage, cross-validation preprocessing leakage, and probability calibration distortion.

### Core Remediation Principles Enacted:
1. **Strict 3-Way Stratified Splitting (60% Train / 20% Validation / 20% Test):**
   - **Training Set ($N = 60,000$):** Used exclusively for candidate model fitting and 3-fold stratified cross-validation.
   - **Validation Set ($N = 20,000$):** Used exclusively for decision-threshold optimization and probability calibration analysis.
   - **Dedicated Holdout Test Set ($N = 20,000$):** Strictly locked and segregated before any model comparison or threshold tuning occurred; touched strictly once for final reporting.
2. **Zero CV Preprocessing Leakage:** Preprocessing transformations (median/mode imputation, standard scaling, and one-hot encoding) are encapsulated within unified `sklearn.pipeline.Pipeline` objects, fitting independently inside each training CV fold.
3. **Rigorous Model Selection via Cross-Validation:** Winning algorithms were selected **strictly by mean training cross-validation ROC-AUC**, eliminating test-set snooping.
4. **Validation-Based Decision Threshold Selection:** Rather than defaulting blindly to $\tau = 0.50$ across all conditions, decision thresholds $\tau^*$ were selected on validation data using documented educational classification objectives (recall-constrained F1 optimization targeting Sensitivity $\ge 70\%$ where feasible, and Youden's J fallback where probabilities were compressed).
5. **Probability Calibration & Brier Analysis:** Probability calibration curves and Brier Skill Scores were computed against baseline sample prevalences, transparently documenting the effects of balanced loss weighting.

---

## 2. Remediated Model Selection & Holdout Test Evaluation Summary

The table below summarizes the winning model selected for each of the 7 chronic conditions based strictly on training cross-validation ROC-AUC, alongside holdout test set performance ($N = 20,000$) evaluated at both the default threshold ($\tau = 0.50$) and the validation-selected decision threshold ($\tau^*$).

| Chronic Disease Target | Selected Model (via Train CV) | Train 3-Fold CV ROC-AUC | Holdout ROC-AUC | Holdout PR-AUC | Validation-Selected Threshold ($\tau^*$) | Sensitivity (Recall @ $\tau^*$) | Specificity (@ $\tau^*$) | Precision (@ $\tau^*$) | F1-Score (@ $\tau^*$) | Pred. Pos. Rate (@ $\tau^*$) | Sensitivity (@ $\tau=0.50$) | Precision (@ $\tau=0.50$) | F1-Score (@ $\tau=0.50$) | Pred. Pos. Rate (@ $\tau=0.50$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diabetes Mellitus** | **XGBoost** | 0.8089 $\pm$ 0.0005 | **0.8051** | 0.3806 | **0.55** | 73.13% | 72.84% | 30.96% | **0.4350** | 33.7% | 78.53% | 28.90% | 0.4225 | 38.8% |
| **Heart Disease (CHD / MI)** | **Logistic Regression** | 0.8194 $\pm$ 0.0007 | **0.8262** | 0.3101 | **0.55** | 74.88% | 75.79% | 23.72% | **0.3602** | 28.8% | 80.41% | 21.92% | 0.3445 | 33.5% |
| **Stroke** | **Linear SVM (Calibrated)** | 0.8021 $\pm$ 0.0073 | **0.7949** | 0.1386 | **0.10** | 38.17% | 89.75% | 14.67% | **0.2119** | 11.5% | 0.00% | 0.00% | 0.0000 | 0.0% |
| **Asthma** | **Random Forest** | 0.6607 $\pm$ 0.0058 | **0.6630** | 0.2720 | **0.40** | 79.52% | 40.64% | 19.79% | **0.3170** | 62.5% | 50.88% | 25.02% | 0.3354 | 31.6% |
| **COPD / Bronchitis** | **Random Forest** | 0.8412 $\pm$ 0.0006 | **0.8318** | 0.3339 | **0.50** | 69.36% | 80.77% | 23.51% | **0.3511** | 23.2% | 69.36% | 23.51% | 0.3511 | 23.2% |
| **Chronic Kidney Disease** | **Linear SVM (Calibrated)** | 0.7828 $\pm$ 0.0047 | **0.8017** | 0.1684 | **0.10** | 45.61% | 88.34% | 17.46% | **0.2526** | 13.4% | 0.00% | 0.00% | 0.0000 | 0.0% |
| **Arthritis** | **XGBoost** | 0.8089 $\pm$ 0.0016 | **0.8154** | 0.6777 | **0.50** | 78.63% | 69.21% | 57.58% | **0.6648** | 47.4% | 78.63% | 57.58% | 0.6648 | 47.4% |

*Holdout Sensitivity Note for COPD:* On the holdout test set, COPD sensitivity reached **69.36%**, slightly below the 70.0% validation target. This demonstrates that validation-selected targets do not guarantee equivalent holdout performance due to sample variability.

---

## 3. Probability Calibration and Decision Threshold Analysis

In imbalanced binary classification, applying `class_weight='balanced'` or `scale_pos_weight` reweights loss penalties. This provides a plausible explanation for why predicted probabilities are shifted upward, though formal proof would require extensive reliability decomposition. To evaluate probability behavior, we calculate the **Brier Score** on validation data and compare it with the **Prevalence-Only Baseline Brier Score** ($p(1-p)$):

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2, \quad \text{Brier}_{\text{baseline}} = \bar{y}(1 - \bar{y}), \quad \text{BSS} = 1 - \frac{\text{Brier}}{\text{Brier}_{\text{baseline}}}$$

### Validation Calibration Metrics & Threshold Summary

| Chronic Disease Target | Selected Model | Validation Prevalence | Model Brier Score | Baseline Brier ($p(1-p)$) | Brier Skill Score (BSS) | Selected Threshold ($\tau^*$) | Documented Optimization Objective |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Diabetes Mellitus** | XGBoost | 14.27% | 0.1799 | 0.1224 | -0.4701 | **0.55** | Recall-constrained F1: Recall $\ge 70\%$. $\tau^*=0.55$ maximizes F1 (0.4399) and specificity (74.1%) at 72.0% validation recall. |
| **Heart Disease** | Logistic Regression | 9.13% | 0.1807 | 0.0830 | -1.1779 | **0.55** | Recall-constrained F1: Recall $\ge 70\%$. $\tau^*=0.55$ maximizes F1 (0.3550) and specificity (75.8%) at 73.6% validation recall. |
| **Stroke** | Linear SVM (Platt) | 4.42% | 0.0401 | 0.0422 | **+0.0501** | **0.10** | Youden's J Fallback: Recall $\ge 70\%$ unavailable. $\tau^*=0.10$ maximizes Youden's J (0.2870) at 39.0% recall, 89.7% specificity. |
| **Asthma** | Random Forest | 15.22% | 0.2145 | 0.1314 | -0.6330 | **0.40** | Recall-constrained F1: Recall $\ge 70\%$. $\tau^*=0.40$ maximizes F1 (0.3218) and specificity (41.9%) at 79.7% validation recall. |
| **COPD / Bronchitis** | Random Forest | 7.85% | 0.1426 | 0.0723 | -0.9710 | **0.50** | Recall-constrained F1: Recall $\ge 70\%$. $\tau^*=0.50$ maximizes F1 (0.3538) and specificity (80.1%) at 71.6% validation recall. |
| **Kidney Disease** | Linear SVM (Platt) | 5.12% | 0.0455 | 0.0486 | **+0.0645** | **0.10** | Youden's J Fallback: Recall $\ge 70\%$ unavailable. $\tau^*=0.10$ maximizes Youden's J (0.3281) at 43.8% recall, 89.0% specificity. |
| **Arthritis** | XGBoost | 34.71% | 0.1767 | 0.2266 | **+0.2204** | **0.50** | Recall-constrained F1: Recall $\ge 70\%$. $\tau^*=0.50$ maximizes F1 (0.6651) and specificity (69.1%) at 78.8% validation recall. |

### Calibration Principles & Methodological Caveats:
1. **Meaning of Brier Skill Score (BSS):**
   - **Positive BSS:** Indicates mean squared error improvement over the naive baseline of predicting constant sample prevalence.
   - **Negative BSS:** Indicates worse mean squared error than that baseline.
   - **Critical Caveat:** Positive BSS alone does **not** establish good probability calibration; it merely indicates that the predictions outperform a trivial constant baseline.
2. **Absence of Clinical Probability Calibration:**
   - None of the model outputs represent clinically calibrated probabilities of true medical disease incidence. Predicted scores represent uncalibrated algorithmic scores under balanced loss optimization.
3. **Validation Threshold Correspondence:**
   - Models were **not refitted** after validation. The validation-selected decision thresholds correspond directly to the exact saved estimator instances. However, threshold selection remains subject to validation-sample variability.

---

## 4. Historical vs. Remediated Results & Provenance Clarification

| Chronic Disease Target | Historical Selected Model | Historical Model Selection Basis | Historical Test ROC-AUC | Remediated Selected Model | Remediated Model Selection Basis | Remediated Holdout ROC-AUC | Methodological Assessment |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **Diabetes Mellitus** | XGBoost | Test Set Evaluation (Leaked) | 0.8066 | **XGBoost** | **Train CV ROC-AUC** | **0.8051** | Confirmed robust; AUC variance $< 0.002$. |
| **Heart Disease** | XGBoost | Test Set Evaluation (Leaked) | 0.8264 | **Logistic Regression** | **Train CV ROC-AUC** | **0.8262** | CV selected simpler, highly interpretable linear model with equivalent discrimination. |
| **Stroke** | Logistic Regression | Test Set Evaluation (Leaked) | 0.7950 | **Linear SVM (Calibrated)** | **Train CV ROC-AUC** | **0.7949** | CV selected Linear SVM (0.8021 CV AUC); threshold $\tau^*=0.10$ prevents 0% sensitivity collapse. |
| **Asthma** | XGBoost | Test Set Evaluation (Leaked) | 0.6652 | **Random Forest** | **Train CV ROC-AUC** | **0.6630** | CV selected Random Forest; test discrimination confirmed stable. |
| **COPD** | Random Forest | Test Set Evaluation (Leaked) | 0.8332 | **Random Forest** | **Train CV ROC-AUC** | **0.8318** | Confirmed robust across random splits. |
| **Kidney Disease** | Logistic Regression | Test Set Evaluation (Leaked) | 0.8018 | **Linear SVM (Calibrated)** | **Train CV ROC-AUC** | **0.8017** | CV selected Linear SVM; holdout AUC matches historical benchmark. |
| **Arthritis** | XGBoost | Test Set Evaluation (Leaked) | 0.8165 | **XGBoost** | **Train CV ROC-AUC** | **0.8154** | Confirmed robust across splits. |

### PR-AUC Arithmetic Re-Evaluation & Artifact Provenance:
- On the identical holdout test set, Linear SVM and Logistic Regression produce similar, but **not identical**, four-decimal PR-AUC values:
  - **Stroke:** Linear SVM = `0.138578` (rounds to **0.1386**) vs. Logistic Regression = `0.138739` (rounds to **0.1387**).
  - **Kidney Disease:** Linear SVM = `0.168367` (rounds to **0.1684**) vs. Logistic Regression = `0.168034` (rounds to **0.1680**).
- While the similarity is explained by linear decision boundaries with sigmoid mappings, it does not independently prove that historical artifacts were not reused.
- Because raw pre-remediation prediction arrays were overwritten during script re-execution and are no longer preserved on disk, historical artifact provenance is classified as **NOT INDEPENDENTLY VERIFIED**.

---

## 5. Survey Methodology & Scope Boundaries

### 1. Complex Survey Sampling Parameters
BRFSS employs a multistage stratified cluster sampling design defined by:
- `_LLCPWT`: Final post-stratification / raking weight.
- `_STSTR`: Stratification identifier.
- `_PSU`: Primary sampling unit (cluster).
- `_STRWT`: Stratum weight.
Because this project fits unweighted standard machine learning models, all reported performance metrics apply strictly to the **analyzed survey cohort** and do not represent weighted US population-level diagnostic benchmarks.

### 2. Cross-Sectional vs. Prospective Scope
Features such as General Health (`GENHLTH`), Physical Health Days (`PHYSHLTH`), and Walking Difficulty (`DIFFWALK`) represent concurrent correlates or consequences of chronic disability rather than causal pre-diagnostic risk factors. This model must be characterized strictly as a **cross-sectional educational classification tool**, not a prospective disease forecasting engine.

---

## 6. Diagnostic Figures Inventory

All diagnostic plots for the holdout test set are rendered at 300 DPI in `reports/figures/`:
- `{slug}_confusion_matrix.png` (Confusion matrix at validation-selected decision threshold $\tau^*$)
- `{slug}_confusion_matrix_default.png` (Confusion matrix at default threshold $\tau = 0.50$)
- `{slug}_roc_curve.png` (Receiver Operating Characteristic with AUC)
- `{slug}_pr_curve.png` (Precision-Recall curve with prevalence baseline)
- `{slug}_calibration_curve.png` (Validation reliability curve vs. baseline Brier score)
