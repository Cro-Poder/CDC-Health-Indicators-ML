# Final Verification Audit — CodeAlpha Task 4

**Project:** CodeAlpha Task 4 — Multi-Disease Prediction Using Machine Learning  
**Workspace:** `D:\CodeAlpha\Medical`  
**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 (`LLCP2024.XPT`)  
**Auditor:** Independent Senior Machine Learning Auditor  
**Date:** November 2026  
**Audit Scope:** Final post-remediation verification of source code, model bundles, holdout evaluation metrics, threshold objectives, probability calibration, and deployment artifacts.  
**Strict Constraint:** Audit performed without retraining models, modifying model parameters, altering thresholds, or changing serialized artifacts.

---

## 1. Executive Summary

This independent verification audit evaluated the actual state of the CodeAlpha Task 4 repository following the execution of the methodological remediation plan.

All models and serialized artifacts in `models/` were evaluated as-is on disk. Every evaluation metric, probability calibration score, and decision threshold was independently recomputed directly from the saved pipelines and the underlying survey data split.

### Audit Verdict Summary:
**APPROVED WITH DOCUMENTED METHODOLOGICAL LIMITATIONS (VERIFIED WITH LIMITATIONS)**  
*(Approved strictly for an educational machine learning portfolio demonstration and CodeAlpha submission. Not approved for clinical screening, diagnostic triage, or population-level epidemiological forecasting).*

### Core Verification Findings:
1. **Model-Fitting Protocol (VERIFIED):** All seven models were fitted on $X_{\text{train}}$ ($N = 60,000$) and were **not refitted** after validation. The validation-selected decision thresholds correspond directly to the exact saved estimator instances. However, threshold selection remains subject to validation-sample variability.
2. **Test-Set Isolation (VERIFIED):** Code inspection confirmed that candidate algorithm selection is governed strictly by training 3-fold cross-validation. Threshold selection and Brier score analysis are performed strictly on validation data ($X_{\text{val}}$, $N = 20,000$). The holdout test set ($X_{\text{test}}$, $N = 20,000$) was locked and evaluated once in Phase 4.
3. **Metric Recomputation (VERIFIED):** All reported holdout test metrics were independently recalculated directly from the serialized `.pkl` pipelines with zero numerical discrepancy ($0.00000000$ difference).
4. **Historical Artifact Provenance (NOT INDEPENDENTLY VERIFIED):** An arithmetic correction confirms that Logistic Regression and Linear SVM produce similar, but **not identical**, four-decimal PR-AUC values on Stroke (0.1387 vs. 0.1386) and Kidney Disease (0.1680 vs. 0.1684). Because raw pre-remediation prediction arrays were overwritten during pipeline re-execution and are no longer preserved on disk, historical artifact provenance cannot be independently verified.
5. **Threshold Objectives (VERIFIED WITH LIMITATIONS):** Five targets followed the documented recall-constrained F1 optimization objective on validation data. Stroke and Kidney Disease followed the documented Youden's J fallback because the 70% recall target could not be achieved under Platt calibration. COPD holdout sensitivity reached 69.36%, falling slightly below the 70% validation target, demonstrating that validation optimization targets do not guarantee equivalent holdout performance.
6. **Probability Calibration (VERIFIED WITH LIMITATIONS):** Brier scores and Brier Skill Scores (BSS) were verified against validation predictions. Three models exhibit positive BSS (improvement over constant prevalence), while four models exhibit negative BSS. Class weighting is a plausible explanation for probability inflation, but is not conclusively proven. Positive BSS alone does not establish good probability calibration, and no model output can be described as a clinically calibrated probability.
7. **Application & Mock Profiles (VERIFIED WITH LIMITATIONS):** All seven saved models load successfully, and live inference executes properly across all 16 features. However, mock-profile predictions demonstrate code execution only and must not be presented as evidence of clinical or medical validity.

---

## 2. Verification of Final Model-Fitting Procedure

For each of the seven chronic conditions, the data partition, cross-validation protocol, fitting sequence, and threshold pairing were audited in `src/model_training.py` and across the serialized `.pkl` files in `models/`:

### Observations Used Across Workflow Phases:
- **Active Population Pool:** $N = 100,000$ complete records sampled from `data/processed/brfss_cleaned.parquet` using `random_state = 42`.
- **Outer Split (80% / 20%):** Stratified partition into Development Set ($N = 80,000$) and Dedicated Holdout Test Set ($N = 20,000$).
- **Inner Split (75% / 25% of Dev):** Stratified partition into Training Set ($N = 60,000$) and Validation Set ($N = 20,000$).

| Disease Target | Cross-Validation Data ($N$) | Validation Data ($N$) | Final Model Fitting Data ($N$) | Model Refitted After Validation? | Selected Threshold ($\tau^*$) Correspondence |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Diabetes Mellitus** | $X_{\text{train}}$ (60,000) | $X_{\text{val}}$ (20,000) | $X_{\text{train}}$ (60,000) | **NO** | Corresponds to exact saved estimator instance. |
| **Heart Disease** | $X_{\text{train}}$ (60,000) | $X_{\text{val}}$ (20,000) | $X_{\text{train}}$ (60,000) | **NO** | Corresponds to exact saved estimator instance. |
| **Stroke** | $X_{\text{train}}$ (60,000) | $X_{\text{val}}$ (20,000) | $X_{\text{train}}$ (60,000) | **NO** | Corresponds to exact saved estimator instance. |
| **Asthma** | $X_{\text{train}}$ (60,000) | $X_{\text{val}}$ (20,000) | $X_{\text{train}}$ (60,000) | **NO** | Corresponds to exact saved estimator instance. |
| **COPD** | $X_{\text{train}}$ (60,000) | $X_{\text{val}}$ (20,000) | $X_{\text{train}}$ (60,000) | **NO** | Corresponds to exact saved estimator instance. |
| **Kidney Disease** | $X_{\text{train}}$ (60,000) | $X_{\text{val}}$ (20,000) | $X_{\text{train}}$ (60,000) | **NO** | Corresponds to exact saved estimator instance. |
| **Arthritis** | $X_{\text{train}}$ (60,000) | $X_{\text{val}}$ (20,000) | $X_{\text{train}}$ (60,000) | **NO** | Corresponds to exact saved estimator instance. |

### Methodological Assessment:
- **Non-Refitted Estimator Status:** Models were **not refitted** on combined development data ($X_{\text{dev}} = X_{\text{train}} \cup X_{\text{val}}$) after threshold optimization.
- **Threshold Validity:** The validation-selected decision thresholds correspond directly to the exact saved estimator instances, ensuring internal consistency between the fitted pipelines and their assigned thresholds.
- **Limitation Disclosed:** Because threshold selection was conducted on a single 20,000-sample validation split, the selected thresholds remain subject to validation-sample variability and finite-sample estimation error.

---

## 3. Verification of Test-Set Isolation

A line-by-line static and dynamic code audit of `src/model_training.py`, `src/model_evaluation.py`, and `src/prediction.py` confirmed strict test-set isolation:

### Code Evidence Checklist:
1. **Candidate Algorithm Selection Uses Training-Only CV:**
   - *Code Evidence:* `src/model_training.py` lines 205–238.
   - `cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)` is executed strictly on `X_train` ($N = 60,000$).
   - `best_model_name = max(cv_results, key=lambda k: cv_results[k]['cv_auc_mean'])` selects the winner based purely on training cross-validation.
2. **Threshold Selection Uses Validation Data Only:**
   - *Code Evidence:* `src/model_training.py` lines 248–266.
   - `val_probs = winning_pipeline.predict_proba(X_val)[:, 1]`.
   - `select_optimal_threshold(threshold_df)` processes only predictions generated on `X_val` ($N = 20,000$).
3. **Probability Calibration Analysis Uses Validation Data Only:**
   - *Code Evidence:* `src/model_training.py` lines 249–260 and 268–278.
   - `brier_metrics = calculate_brier_metrics(y_val, val_probs)` and `plot_calibration_curve(y_true=y_val, y_prob=val_probs, ...)` operate strictly on validation observations.
4. **Holdout Test Set Isolation:**
   - *Code Evidence:* `src/model_training.py` lines 281–304 (`Phase 4: Final Evaluation on Locked Test Set`).
   - `X_test` and `y_test` are accessed strictly after candidate comparison, model selection, model fitting, validation calibration, and threshold optimization are complete.
   - Test set performance metrics are used strictly for reporting and do not influence hyperparameters, feature selection, or decision thresholds.
5. **Zero CV Preprocessing Leakage:**
   - *Code Evidence:* `src/model_training.py` lines 135–144.
   - `ColumnTransformer` is encapsulated inside `Pipeline`. During `cross_val_score`, imputers and scalers fit exclusively on each fold's training slice ($N = 40,000$), preventing any feature summary leakage into the holdout fold ($N = 20,000$).

**Verdict on Test-Set Isolation: VERIFIED**

---

## 4. Independent Metric Recomputation & PR-AUC Investigation

### 4.1 Independently Recomputed Test Set Metrics ($N = 20,000$)

Using an independent audit script (`scratch/audit_verification.py`), every metric was recalculated by loading the serialized `.pkl` pipeline files directly from `models/` and evaluating against the 20,000 holdout test records:

| Disease Target | Serialized Model Architecture | Selected $\tau^*$ | Holdout ROC-AUC | Holdout PR-AUC | Accuracy (@ $\tau^*$) | Balanced Acc (@ $\tau^*$) | Sensitivity (@ $\tau^*$) | Specificity (@ $\tau^*$) | Precision (@ $\tau^*$) | F1-Score (@ $\tau^*$) | Recomputation Discrepancy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diabetes** | XGBClassifier | **0.55** | 0.805133 | 0.380587 | 72.88% | 72.99% | 73.13% | 72.84% | 30.96% | 0.4350 | **0.00000000** |
| **Heart Disease** | LogisticRegression | **0.55** | 0.826177 | 0.310133 | 75.71% | 75.33% | 74.88% | 75.79% | 23.72% | 0.3602 | **0.00000000** |
| **Stroke** | Calibrated LinearSVC | **0.10** | 0.794936 | 0.138578 | 87.47% | 63.96% | 38.17% | 89.75% | 14.67% | 0.2119 | **0.00000000** |
| **Asthma** | RandomForestClassifier | **0.40** | 0.663038 | 0.272036 | 46.56% | 60.08% | 79.52% | 40.64% | 19.79% | 0.3170 | **0.00000000** |
| **COPD** | RandomForestClassifier | **0.50** | 0.831792 | 0.333923 | 79.88% | 75.07% | 69.36% | 80.77% | 23.51% | 0.3511 | **0.00000000** |
| **Kidney Disease** | Calibrated LinearSVC | **0.10** | 0.801728 | 0.168367 | 86.15% | 66.98% | 45.61% | 88.34% | 17.46% | 0.2526 | **0.00000000** |
| **Arthritis** | XGBClassifier | **0.50** | 0.815350 | 0.677726 | 72.48% | 73.92% | 78.63% | 69.21% | 57.58% | 0.6648 | **0.00000000** |

*Verification Result:* The discrepancy between the recomputed metrics and the values stored in `models/model_metadata.json` and `reports/model_comparison_summary.csv` is **0.00000000**. The reported metrics accurately reflect the serialized pipeline artifacts.

---

### 4.2 PR-AUC Arithmetic Correction & Historical Artifact Provenance

The previous audit asserted that Logistic Regression and Linear SVM produce identical four-decimal PR-AUC values on Stroke and Kidney Disease. This assertion was mathematically inaccurate.

#### Arithmetic Re-Evaluation:
Evaluating both classifiers on the identical 20,000-sample holdout test set yields:

| Disease Target | Algorithm | Raw PR-AUC | Correct Four-Decimal Rounding | Discrepancy Between Classifiers |
| :--- | :--- | :---: | :---: | :---: |
| **Stroke** | **Linear SVM (Platt)** | **0.138578** | **0.1386** | \- |
| **Stroke** | **Logistic Regression** | **0.138739** | **0.1387** | **0.000161** (Differs at 4th decimal) |
| **Kidney Disease** | **Linear SVM (Platt)** | **0.168367** | **0.1684** | \- |
| **Kidney Disease** | **Logistic Regression** | **0.168034** | **0.1680** | **0.000333** (Differs at 4th decimal) |

#### Methodological Findings:
1. **Classifier Ranking Similarity:** Because both Logistic Regression and Linear Support Vector Machines with Platt calibration construct linear hyperplanes separating the identical 16 tabular features, they produce similar ranking metrics. However, they do **not** round to identical four-decimal values (`0.1386 != 0.1387` and `0.1684 != 0.1680`).
2. **Artifact Provenance Status:** While the newly saved model bundles produce the recomputed test metrics, this numerical similarity does **not** independently prove that historical evaluation artifacts were not reused during prior reporting transitions.
3. **Absence of Historical Prediction Arrays:** The raw, uncompressed floating-point prediction arrays and intermediate files generated prior to remediation were overwritten during pipeline execution. Because original historical prediction arrays are no longer preserved on disk for bit-level comparison, historical artifact provenance must be classified as **NOT INDEPENDENTLY VERIFIED**.

---

## 5. Verification of Decision Threshold Objectives

The audit examined the threshold selection logic (`src/model_evaluation.py`, lines 145–185) and compared it with validation results:

### Audit Findings by Target:
1. **Recall-Constrained F1 Optimization (5 Targets):**
   - Diabetes, Heart Disease, Asthma, COPD, and Arthritis applied the documented validation objective:
     $$\text{Maximize } F_1(\tau) \quad \text{subject to } \quad \text{Recall}(\tau) \ge 0.70$$
   - *Validation Results:* Diabetes achieved 72.0% recall ($\tau^*=0.55$); Heart Disease achieved 73.6% recall ($\tau^*=0.55$); Asthma achieved 79.7% recall ($\tau^*=0.40$); COPD achieved 71.6% recall ($\tau^*=0.50$); Arthritis achieved 78.8% recall ($\tau^*=0.50$).
   - *Holdout Performance & Validation Target Gap:* On the holdout test set, COPD achieved **69.36% sensitivity**, falling slightly below the 70.0% validation target. This demonstrates that threshold optimization on validation data does not guarantee that holdout sensitivity will strictly meet or exceed validation targets.
2. **Youden's J Fallback (2 Targets):**
   - For Stroke and Kidney Disease under Platt calibration, predicted probabilities remained concentrated in low ranges ($< 0.25$). No threshold in the evaluation grid achieved the 70% sensitivity target.
   - As documented in `src/model_evaluation.py`, the code triggered the **Youden's J fallback**:
     $$\text{Maximize } (\text{Recall} + \text{Specificity} - 1)$$
   - Selecting $\tau^* = 0.10$ maximized Youden's J ($J = 0.2870$ for Stroke, $J = 0.3281$ for Kidney Disease) with $\sim 89\%$ specificity.
   - *Holdout Outcome:* On the test set, $\tau^* = 0.10$ produced **38.17% sensitivity for Stroke** and **45.61% sensitivity for Kidney Disease** with $88\% - 90\%$ specificity, avoiding the 0.0% sensitivity collapse observed at default $\tau = 0.50$.
3. **Test-Set Isolation:** Thresholds were selected entirely on validation data without reference to holdout test labels.

**Verdict on Threshold Objectives: VERIFIED WITH LIMITATIONS**  
*(Verified as implemented; subject to the caveat that validation targets do not guarantee equivalent holdout performance, as seen in COPD).*

---

## 6. Verification of Probability Calibration

Calibration metrics were audited against validation set predictions ($N = 20,000$):

$$\text{Brier} = \frac{1}{N} \sum (\hat{p}_i - y_i)^2, \quad \text{Brier}_{\text{baseline}} = \bar{y}(1 - \bar{y}), \quad \text{BSS} = 1 - \frac{\text{Brier}}{\text{Brier}_{\text{baseline}}}$$

### Comprehensive Calibration Audit:

| Disease Target | Selected Model | Val. Prevalence | Validation Brier Score | Prevalence Base Brier | Brier Skill Score (BSS) | Skill Score Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Diabetes** | XGBoost | 14.27% | 0.1799 | 0.1224 | **-0.4701** | **Negative BSS** (Worse MSE than constant prevalence) |
| **Heart Disease** | Logistic Regression | 9.13% | 0.1807 | 0.0830 | **-1.1779** | **Negative BSS** (Worse MSE than constant prevalence) |
| **Stroke** | Linear SVM (Platt) | 4.42% | 0.0401 | 0.0422 | **+0.0501** | **Positive BSS** (Improvement over constant prevalence) |
| **Asthma** | Random Forest | 15.22% | 0.2145 | 0.1314 | **-0.6330** | **Negative BSS** (Worse MSE than constant prevalence) |
| **COPD** | Random Forest | 7.85% | 0.1426 | 0.0723 | **-0.9710** | **Negative BSS** (Worse MSE than constant prevalence) |
| **Kidney Disease** | Linear SVM (Platt) | 5.12% | 0.0455 | 0.0486 | **+0.0645** | **Positive BSS** (Improvement over constant prevalence) |
| **Arthritis** | XGBoost | 34.71% | 0.1767 | 0.2266 | **+0.2204** | **Positive BSS** (Improvement over constant prevalence) |

### Calibration Principles & Limitations:
1. **Meaning of Brier Skill Score:**
   - **Positive BSS:** Indicates mean squared error improvement over the naive baseline of predicting constant sample prevalence.
   - **Negative BSS:** Indicates worse mean squared error than that constant-prevalence baseline.
   - **Critical Caveat:** Positive BSS alone does **not** establish that a model outputs well-calibrated probabilities across its entire range; it merely indicates that the predictions outperform a trivial constant baseline.
2. **Class Weighting as a Plausible Explanation:**
   - Applying `class_weight='balanced'` and `scale_pos_weight` penalizes minority misclassification, which is a plausible explanation for probability inflation and negative BSS on four models. However, this is an empirical hypothesis and is not conclusively proven without formal decomposition of reliability and resolution.
3. **Absence of Clinical Calibration:**
   - None of the model outputs represent clinically calibrated probabilities of true disease incidence. Predicted scores represent uncalibrated algorithmic scores under balanced loss optimization.

**Verdict on Probability Calibration: VERIFIED WITH LIMITATIONS**

---

## 7. Verification of Application & Evidence Boundaries

### 7.1 Distinguishing Verification Modalities:
To maintain rigorous audit standards, the following evidence boundaries are explicitly defined:
1. **Code Inspection (VERIFIED):** Confirmed by direct static analysis that `src/model_training.py` isolates training CV from validation threshold tuning and locks the test set.
2. **Metric Recomputation (VERIFIED):** Confirmed by independent execution that test-set metrics match serialized model outputs to eight decimal places.
3. **Unit Tests (VERIFIED):** Confirmed by automated execution that all 11 unit tests in `tests/test_pipeline.py` pass cleanly.
4. **Artifact Freshness (VERIFIED):** Current disk files reflect post-remediation generation. However, matching timestamps do not prove complete reproducibility across arbitrary environments or historical provenance.
5. **Demonstration Mock Profiles (DEMONSTRATION ONLY):** Testing mock profiles demonstrates code execution and threshold mechanics. **Mock profiles must not be presented as evidence of medical or clinical validity.**

### 7.2 Application Terminology Review:
The Streamlit interface (`app.py`) was reviewed to remove clinical screening claims, diagnostic assertions, and misleading uses of the term "relative risk." The interface clearly disclaims that predictions are educational classifications based on self-reported survey associations.

---

## 8. Final Status Classifications

| Audit Item | Final Classification | Evidentiary Basis |
| :--- | :---: | :--- |
| **Model-Fitting Procedure** | **VERIFIED** | Estimators fitted on $X_{\text{train}}$ ($N=60,000$); not refitted; thresholds match saved instances. Subject to validation variability. |
| **Test-Set Isolation** | **VERIFIED** | Model selection and thresholds use CV and validation data only; test set touched strictly once in Phase 4. |
| **Metric Recomputation** | **VERIFIED** | All metrics independently recomputed on holdout test set with zero numerical discrepancy ($0.00000000$). |
| **PR-AUC Arithmetic & Provenance** | **NOT INDEPENDENTLY VERIFIED** | Arithmetic error corrected (PR-AUCs differ at 4th decimal). Historical pre-remediation artifact provenance cannot be independently verified. |
| **Threshold Objectives** | **VERIFIED WITH LIMITATIONS** | Documented objectives verified; COPD holdout sensitivity (69.36%) discloses the gap between validation target and holdout outcome. |
| **Probability Calibration** | **VERIFIED WITH LIMITATIONS** | Positive and negative BSS accurately computed; positive BSS caveat and class-weighting hypothesis documented. |
| **Survey & Medical Scope** | **VERIFIED** | Formally defined as educational cross-sectional classification with unweighted survey design disclosures. |
| **Application & Model Artifacts** | **VERIFIED** | All seven bundles load cleanly; live inference functions; demonstration-only nature of mock profiles clarified. |

---

## 9. Final Publication Recommendation

### Official Recommendation: **APPROVED WITH DOCUMENTED METHODOLOGICAL LIMITATIONS (VERIFIED WITH LIMITATIONS)**

The CodeAlpha Task 4 Multi-Disease Prediction repository demonstrates high technical competence, thorough data preprocessing, zero test-set leakage in candidate selection, and comprehensive documentation.

It is **approved for GitHub portfolio publication and CodeAlpha internship submission** with the clear understanding that:
- It is an educational classification demonstration on self-reported survey data.
- It does not perform clinical screening or diagnostic triage.
- Threshold selection remains subject to validation-sample variability.
- Four models exhibit negative Brier Skill Scores due to probability inflation.
- Historical pre-remediation artifact provenance is not independently verified.
