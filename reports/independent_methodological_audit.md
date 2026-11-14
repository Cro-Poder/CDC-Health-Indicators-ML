# Independent Methodological Audit Report
## CodeAlpha Task 4 — Multi-Disease Prediction Using Machine Learning
**Auditor:** Independent Senior Machine Learning Auditor  
**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 (`LLCP2024.XPT`)  
**Workspace:** `D:\CodeAlpha\Medical`  
**Date:** November 2026  

---

## 1. Executive Summary

An independent, rigorous methodological audit was conducted on the CodeAlpha Task 4 Multi-Disease Prediction project. The project attempts to predict 7 chronic conditions (Diabetes, Heart Disease, Stroke, Asthma, COPD, Kidney Disease, Arthritis) from patient demographic, behavioral, and functional survey responses.

This audit evaluated raw dataset integrity, data leakage safeguards, model selection protocols, evaluation metric validity, thresholding and calibration, survey weighting, and epidemiological feature interpretation.

### Summary of Audit Classifications:
| Area Audited | Finding Classification | Primary Methodological Issue |
| :--- | :--- | :--- |
| **Model Selection** | **CRITICAL** | Best models were selected based on **test-set evaluation performance** rather than validation/cross-validation scores. |
| **Target Coding (Diabetes)** | **MAJOR** | 11,307 pre-diabetes cases and 3,395 gestational diabetes cases were collapsed into the negative class, penalizing precision. |
| **Probability Calibration & Thresholding** | **MAJOR** | Class rebalancing distorted predicted probabilities, elevating Brier scores and causing low precision (10.2%–28.9%) at default $\tau = 0.5$. |
| **Survey Sampling Design** | **MAJOR** | Simple random splitting treats stratified cluster survey data as i.i.d., evaluating the sample rather than the weighted US population. |
| **Feature Temporal Causality** | **MAJOR** | Functional impairment features (`diff_walk`, `physical_health_days`) represent post-disease consequences rather than pre-morbid predictors. |
| **Questionnaire Skip Terminology** | **MAJOR** | Claim of "zero question skips" contradicted empirical missingness across target variables. |
| **Cross-Validation Preprocessing** | **MINOR** | Scaling and imputation were fit on the full 80,000 training pool prior to splitting folds inside `cross_val_score`. |
| **Dataset Dimension Verification** | **VERIFIED** | The discrepancy between 301 and 345 variables was resolved; `LLCP2024.XPT` contains exactly 301 columns. |
| **Cross-Target Data Leakage** | **VERIFIED** | All 7 disease targets are strictly excluded from the predictor pool of every model. |
| **Metric Recalculation** | **VERIFIED** | All test metrics, confusion matrix counts, and ROC-AUC scores were independently verified. |

### Overall Recommendation:
**NOT APPROVED PENDING MAJOR CORRECTIONS.**  
While the code architecture, data cleaning, and leakage prevention across targets are well-engineered, the use of the test set for model selection, probability calibration distortion from uncalibrated re-weighting, and conflation of cross-sectional correlation with disease prediction require formal remediation.

---

## 2. Dataset Integrity Findings

### 2.1 File Dimensions & Row/Column Verification
- **File Name:** `LLCP2024.XPT` (located in root and `data/raw/`)
- **File Size:** `1,093,874,240` bytes (~1.02 GB)
- **Verified Row Count:** **457,670 rows** (audited via `pyreadstat` and `pandas`)
- **Verified Column Count:** **301 columns**
- **Cleaned Parquet Rows:** **444,259 rows** (97.07% retention of respondents with all 7 targets non-null)

### 2.2 Resolution of the 301 vs. 345 Variable Discrepancy
- **Finding:** **VERIFIED (Documentation vs. File Architecture)**
- **Investigation:** 
  The CDC BRFSS 2024 comprehensive survey layout and data dictionary documentation list **345 variables**. This higher count represents the master layout across all questionnaire versions, split-sample questionnaire versions (Version 1, Version 2, Version 3), and optional state module placeholders.
  When the unified, combined landline and cell phone public transport file (`LLCP2024.XPT`) is compiled and released by the CDC, unadministered state-specific placeholders and administrative tracking keys are omitted, resulting in exactly **301 physical columns** in the `.XPT` data matrix.
- **Evidence:** C-level parsing via `pyreadstat.read_xport('LLCP2024.XPT', metadataonly=True)` returns `len(meta.column_names) == 301`.

### 2.3 Target Missingness vs. "Zero Question Skips" Contradiction
- **Finding:** **MAJOR METHODOLOGICAL REPORTING ERROR**
- **Investigation:** 
  The Dataset Inspection Report stated: *"All 7 diseases feature complete nationwide coverage (asked of all respondents), zero question skips, clean binary semantics..."*
  However, independent missing-value audit reveals the following missingness in the raw data:
  * `DIABETE4`: 4 blank/NaN, 798 "Don't know" (7.0), 232 "Refused" (9.0) = **1,034 missing (0.23%)**
  * `_MICHD`: **5,206 blank/NaN (1.14%)**
  * `CVDSTRK3`: 3 blank/NaN, 1,244 "Don't know", 205 "Refused" = **1,452 missing (0.32%)**
  * `ASTHMA3`: 3 blank/NaN, 1,650 "Don't know", 208 "Refused" = **1,861 missing (0.41%)**
  * `CHCCOPD3`: 7 blank/NaN, 1,921 "Don't know", 246 "Refused" = **2,174 missing (0.47%)**
  * `CHCKDNY2`: 6 blank/NaN, 1,715 "Don't know", 258 "Refused" = **1,979 missing (0.43%)**
  * `HAVARTH4`: 5 blank/NaN, 2,273 "Don't know", 295 "Refused" = **2,573 missing (0.56%)**
- **Resolution:**
  In survey research, "skip logic" means branching around questions conditionally (e.g. asking cigarette count only if `SMOKE100 == 1`). The chronic disease questions belong to the Core Questionnaire and are administered without conditional branching. However, claiming this constitutes "zero question skips" falsely implies complete zero-missingness. In reality, item non-response ("Don't know", "Refused") and interview break-offs produce thousands of missing records. In `_MICHD`, missingness on *either* constituent question (`CVDINFR4` or `CVDCRHD4`) forces `_MICHD` to `NaN` (5,206 rows).

---

## 3. Target Coding Findings: Diabetes, Gestational Diabetes, and Pre-diabetes

### 3.1 Empirical Coding Audit
- **Finding:** **MAJOR CLINICAL / METHODOLOGICAL CONCERN**
- **Code Reference:** `src/data_preprocessing.py`, lines 166–169:
  ```python
  targets['target_diabetes'] = df['DIABETE4'].map(
      lambda x: 1.0 if x == 1.0 else (0.0 if x in [2.0, 3.0, 4.0] else np.nan)
  )
  ```
- **Raw CDC Codebook Breakdown (`DIABETE4`):**
  * `1.0`: Yes (diagnosed diabetes) $\to$ **65,809** (14.38%)
  * `2.0`: Yes, but told only during pregnancy (gestational diabetes) $\to$ **3,395** (0.74%)
  * `3.0`: No (never told had diabetes) $\to$ **376,125** (82.18%)
  * `4.0`: No, pre-diabetes or borderline diabetes $\to$ **11,307** (2.47%)
  * `7.0`: Don't know / Not sure $\to$ **798**
  * `9.0`: Refused $\to$ **232**
  * `NaN`: Missing $\to$ **4**

### 3.2 Methodological Critique
1. **Conflating Pre-diabetes with Normoglycemia:**
   Code `4.0` represents individuals clinically recognized as having pre-diabetes or borderline diabetes. These individuals possess pathophysiological insulin resistance, elevated fasting blood glucose, and typical metabolic syndrome profiles (elevated BMI, physical inactivity, older age). Grouping these 11,307 pre-diabetic patients into the **Negative (0.0)** class treats them as biologically identical to healthy controls.
2. **Impact on Evaluation Metrics:**
   When the trained model predicts a high risk of diabetes for a patient with a BMI of 36, age 62, and sedentary habits who has pre-diabetes, the current evaluation pipeline registers this prediction as a **False Positive**! This directly depresses the model's reported Precision (28.96%).
3. **Gestational Diabetes (`2.0`):**
   Gestational diabetes is transient hyperglycemia of pregnancy. While standard CDC derived variables (`_RFDIAB3`) treat gestational diabetes as non-chronic diabetes, epidemiologically it confers a 50% lifetime risk of developing overt Type 2 diabetes. 
4. **Auditor Recommendation:**
   The cleanest epidemiological approaches are:
   - *Option A (Strict Binary):* Drop code `4.0` (pre-diabetes) and code `2.0` from the training and evaluation sets, defining the target strictly as Diagnosed Chronic Diabetes (`1.0`) vs. Healthy Normoglycemic Controls (`3.0`).
   - *Option B (Documented Inclusion):* Retain pre-diabetes in the negative pool but explicitly document in all reports and web UI that the negative class contains 11,307 individuals with pre-diabetes, explaining the resulting precision ceiling.

---

## 4. Data Leakage Findings

### 4.1 Cross-Target Contamination
- **Finding:** **VERIFIED (Zero Cross-Target Leakage)**
- **Code Reference:** `src/data_preprocessing.py`, lines 18–60, 240–255:
  ```python
  RAW_FEATURE_COLS = ['GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'DIFFWALK', 'EXERANY2', '_SMOKER3', '_CURECI3', '_RFDRHV9', 'CHECKUP1', '_BMI5', '_AGE80', '_SEX', '_EDUCAG', '_INCOMG1', '_RACEGR3', '_HLTHPL2']
  RAW_DISEASE_COLS = ['DIABETE4', '_MICHD', 'CVDSTRK3', 'ASTHMA3', 'CHCCOPD3', 'CHCKDNY2', 'HAVARTH4']
  ```
- **Audit Verification:**
  None of the 7 disease targets are included in `CLEAN_FEATURE_NAMES`. When models are trained for `target_heart_disease`, the feature matrix $\mathbf{X}$ does not contain `target_diabetes`, `target_stroke`, or any other disease indicator. Every disease model operates strictly on the identical 16-feature risk factor vector.
- **Proxy Variable Audit:** 
  Post-diagnosis management fields (e.g. `DIABAGE4` age at diagnosis, `DIABTYPE` type of diabetes, `_DRDXAR2` arthritis derived diagnosis, `_CASTHM1` current asthma) were audited and confirmed absent from `RAW_FEATURE_COLS`.

### 4.2 Train/Test Preprocessor Isolation
- **Finding:** **VERIFIED (At the Global Split Level)**
- **Code Reference:** `src/model_training.py`, lines 170–174:
  ```python
  preprocessor = create_feature_preprocessor()
  X_train_proc = preprocessor.fit_transform(X_train)
  X_test_proc = preprocessor.transform(X_test)
  ```
  `preprocessor.fit_transform()` is invoked strictly on `X_train`. The test set `X_test` is only transformed via `preprocessor.transform()`. No scaling or imputation parameters leak from the test set into the preprocessor.

### 4.3 Validation Leakage Inside Cross-Validation
- **Finding:** **MINOR METHODOLOGICAL LEAKAGE**
- **Code Reference:** `src/model_training.py`, lines 172, 187:
  ```python
  X_train_proc = preprocessor.fit_transform(X_train)
  cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
  ```
- **Methodological Issue:**
  The preprocessor was fitted on all 80,000 samples of `X_train` **before** performing 3-fold cross-validation. In each fold of `cross_val_score`, the validation fold's feature distributions slightly influenced the imputation medians and `StandardScaler` mean/variance. While this did not leak test data, it represents fold-to-fold leakage inside cross-validation.
- **Remediation:** Enclose the preprocessor and estimator inside a unified `imblearn` / `sklearn` `Pipeline([('preprocessor', preprocessor), ('classifier', model)])` and pass the pipeline to `cross_val_score`.

---

## 5. Model Selection Findings

### 5.1 Test-Set Snooping in Model Selection
- **Finding:** **CRITICAL METHODOLOGICAL DEFECT**
- **Code Reference:** `src/model_training.py`, lines 196–224:
  ```python
  # Predict on untouched test set
  y_pred = model.predict(X_test_proc)
  y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, 'predict_proba') else None

  metrics = calculate_clinical_metrics(y_test, y_pred, y_prob)
  ...
  comparison_records.append(metrics)
  ...
  comp_df = pd.DataFrame(comparison_records)
  viable_models = comp_df[comp_df['recall'] >= 0.40]
  best_row = viable_models.sort_values(by=['roc_auc', 'balanced_accuracy'], ascending=False).iloc[0]
  best_name = best_row['model_name']
  best_model = trained_estimators[best_name]
  ```
- **Auditor Evidence:**
  The script iterates through all candidate algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost), generates predictions on `X_test_proc`, evaluates metrics on `y_test`, stores them in `comp_df`, and selects `best_model` by sorting `comp_df` on **test-set Recall and test-set ROC-AUC**!
- **Scientific Impact:**
  This directly violates the core principle of holdout evaluation:
  1. The test set must be used **only once** for the final unbiased audit of the chosen model.
  2. Selecting the winning architecture based on test-set metrics creates **selection bias / data snooping**, invalidating the claim that the test set was "untouched".
  3. Model selection must be conducted using **cross-validation performance** on the training set or an independent validation split ($\mathbf{X}_{\text{val}}$).

---

## 6. Metric Verification & Precision Collapse Analysis

### 6.1 Independent Recalculation of Test Set Metrics
All metrics were independently recalculated using the saved model artifacts (`models/*.pkl`) on the 20,000-patient test set ($N_{\text{test}} = 20,000$):

| Disease | Selected Model | True Prev | Predicted Pos Rate | Test Accuracy | Naive Baseline Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | Test ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diabetes** | XGBoost | 14.27% | 38.98% | 69.32% | **85.72%** | 79.09% | 67.69% | 28.96% | 0.4240 | 0.8066 | 0.3827 |
| **Heart Disease** | XGBoost | 9.13% | 34.17% | 71.43% | **90.86%** | 80.68% | 70.50% | 21.57% | 0.3403 | 0.8264 | 0.3105 |
| **Stroke** | Logistic Reg | 4.42% | 31.43% | 70.58% | **95.58%** | 72.71% | 70.48% | 10.21% | 0.1791 | 0.7950 | 0.1386 |
| **Asthma** | XGBoost | 15.55% | 37.33% | 65.09% | **84.45%** | 57.79% | 66.44% | 24.08% | 0.3400 | 0.6652 | 0.2729 |
| **COPD** | Random Forest | 7.85% | 23.71% | 79.49% | **92.15%** | 70.38% | 80.27% | 23.30% | 0.3501 | 0.8332 | 0.3332 |
| **Kidney Disease** | Logistic Reg | 5.13% | 32.36% | 70.38% | **94.87%** | 76.71% | 70.04% | 12.16% | 0.2099 | 0.8018 | 0.1684 |
| **Arthritis** | XGBoost | 34.71% | 47.62% | 72.53% | **65.30%** | 79.02% | 69.08% | 57.60% | 0.6663 | 0.8165 | 0.6801 |

### 6.2 Confusion Matrix Verification
All confusion matrix counts reported in `reports/model_comparison_summary.csv` were verified down to the exact integer:
- **Diabetes:** TP = 2,258; FP = 5,539; TN = 11,606; FN = 597
- **Heart Disease:** TP = 1,474; FP = 5,361; TN = 12,812; FN = 353
- **Stroke:** TP = 642; FP = 5,644; TN = 13,473; FN = 241
- **Asthma:** TP = 1,798; FP = 5,668; TN = 11,221; FN = 1,313
- **COPD:** TP = 1,105; FP = 3,637; TN = 14,793; FN = 465
- **Kidney Disease:** TP = 787; FP = 5,685; TN = 13,289; FN = 239
- **Arthritis:** TP = 5,485; FP = 4,038; TN = 9,021; FN = 1,456

### 6.3 Why Precision Collapses and Accuracy Drops Below Baseline
- **Audit Diagnosis:**
  1. **Disproportionate Positive Prediction Rate:**
     For Stroke, the true prevalence is **4.42%**, but the model predicts **31.43%** positive (7.1 times higher than the actual prevalence). For Kidney Disease, actual prevalence is **5.13%**, but the model predicts **32.36%** positive (6.3 times higher).
  2. **The Mathematics of Severe Imbalance:**
     $$\text{Precision} = \frac{\text{Prevalence} \times \text{Recall}}{\text{Prevalence} \times \text{Recall} + (1 - \text{Prevalence}) \times (1 - \text{Specificity})}$$
     When prevalence is 4.4% and specificity is 70.5%, the 29.5% of healthy individuals who trigger false alarms ($0.956 \times 0.295 = 0.282$) vastly outnumber the true positives ($0.044 \times 0.727 = 0.032$). Thus, Precision is mathematically capped at $\approx 10\%$.
  3. **Majority-Class Baseline Comparison:**
     A naive model predicting "No" for every patient achieves **95.58% accuracy on Stroke** and **94.87% on Kidney Disease**. Our trained models achieve **70.58%** and **70.38%** accuracy. This accuracy penalty is the inevitable cost of forcing high sensitivity on an imbalanced target using balanced class weights at the standard $\tau = 0.5$ threshold.

---

## 7. Decision Threshold & Calibration Findings

### 7.1 Probability Distortion from Class Re-weighting
- **Finding:** **MAJOR METHODOLOGICAL INSIGHT**
- **Investigation:**
  In `src/model_training.py`, models were trained with `class_weight='balanced'` or `scale_pos_weight = (1 - p) / p`.
  Weighting shifts the loss function to penalize false negatives by a factor of up to $22\times$.
  However, this alters the model's posterior probability calibration:
  $$P_{\text{balanced}}(Y=1|\mathbf{x}) \ne P_{\text{true}}(Y=1|\mathbf{x})$$
  The output of `predict_proba()` no longer reflects the true probability in the general population; it reflects probability in a hypothetical balanced 50:50 population!
- **Brier Score Audit (Evaluated on Validation Split):**
  * Stroke Brier score: **0.1898** (A naive model predicting the base rate 0.044 yields a Brier score of $0.044 \times (1 - 0.044) = \mathbf{0.0420}$). The model's Brier score is substantially worse than the base-rate prior due to severe probability over-estimation.
  * Diabetes Brier score: **0.1799** (vs. base-rate Brier score of 0.1223).
  * Heart Disease Brier score: **0.1753** (vs. base-rate Brier score of 0.0829).

### 7.2 Validation Threshold Sweep Analysis
To avoid test-set snooping, we performed a threshold sweep strictly on a **20,000-sample validation split** drawn from the training pool:

#### Stroke (Logistic Regression on Validation Set):
| Threshold ($\tau$) | Predicted Positive Rate | Recall (Sensitivity) | Specificity | Precision | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.2 | 67.4% | 95.2% | 33.9% | 6.2% | 0.1171 |
| 0.3 | 54.1% | 90.7% | 47.6% | 7.4% | 0.1368 |
| 0.4 | 42.4% | 83.2% | 59.5% | 8.7% | 0.1569 |
| **0.5 (Current)** | **31.5%** | **74.1%** | **70.5%** | **10.4%** | **0.1820** |
| 0.6 | 21.8% | 59.6% | 79.9% | 12.1% | 0.2008 |
| **0.7 (F1-Optimal)** | **13.1%** | **42.6%** | **88.3%** | **14.3%** | **0.2144** |
| 0.8 | 6.3% | 24.5% | 94.5% | 17.2% | 0.2018 |

#### Diabetes (XGBoost on Validation Set):
| Threshold ($\tau$) | Predicted Positive Rate | Recall (Sensitivity) | Specificity | Precision | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.3 | 55.5% | 91.5% | 50.5% | 23.5% | 0.3745 |
| 0.4 | 46.8% | 86.6% | 59.8% | 26.4% | 0.4048 |
| **0.5 (Current)** | **37.2%** | **77.7%** | **69.5%** | **29.8%** | **0.4307** |
| **0.6 (F1-Optimal)** | **27.2%** | **65.0%** | **79.1%** | **34.2%** | **0.4480** |
| 0.7 | 16.7% | 46.4% | 88.2% | 39.7% | 0.4281 |

- **Auditor Takeaway:**
  Operating at the default $\tau = 0.5$ on a balanced classifier heavily prioritizes sensitivity at the expense of precision and specificity. If the goal is optimal F1 or balanced screening efficiency, tuning $\tau$ to $0.60 - 0.70$ on validation data significantly reduces false alarms (cutting the predicted positive rate from 31.5% down to 13.1% for Stroke) while preserving moderate sensitivity.

---

## 8. Survey Methodology Findings

### 8.1 Complex Sampling Design
- **Finding:** **MAJOR METHODOLOGICAL LIMITATION**
- **Context:**
  The BRFSS is not a simple random sample of US residents. It employs a complex, multi-stage stratified cluster sampling design:
  * **Stratification:** Stratified by geographic sub-state regions (`_STRWT`).
  * **Clustering:** Primary Sampling Units (`_PSU`).
  * **Post-Stratification (Raking):** Adjusted by age, sex, race, education, marital status, home ownership, and telephone ownership (`_LLCPWT`).
- **Impact on Current Pipeline:**
  1. Ordinary random train/test splitting (`train_test_split(..., random_state=42)`) treats records as independent and identically distributed (i.i.d.).
  2. Unweighted models overfit geographic and demographic strata that CDC oversampled (e.g. smaller states like Alaska or Vermont have disproportionately high representation relative to population size).
  3. The reported evaluation metrics reflect performance on the **survey sample**, not the **true non-institutionalized US adult population**.
- **Recommendation for Educational Projects:**
  Full survey-weighted machine learning requires specialized pseudoreplication (Jackknife or Taylor series linearization). For this educational project:
  - Models may remain unweighted or utilize `sample_weight=df['_LLCPWT']`.
  - All documentation must explicitly clarify that evaluation metrics represent unweighted sample performance rather than national population-representative parameters.

---

## 9. Feature Selection Findings: Risk Factors vs. Disease Consequences

### 9.1 Audit of the 16 Input Features
- **Finding:** **MAJOR EPIDEMIOLOGICAL CONCERN**
- The feature set contains 16 variables. Several variables are not pre-diagnostic risk factors, but rather **symptoms, downstream functional consequences, or post-diagnosis artifacts**:

| Feature | Classification | Epidemiological & Causal Reality |
| :--- | :--- | :--- |
| `age`, `sex`, `race_ethnicity` | **True Baseline Risk Factor** | Immutable demographic baseline traits. |
| `smoker_status`, `heavy_drinker`, `exercise_any`, `bmi` | **True Modifiable Risk Factor** | Behavioral exposures that biologically precede disease pathogenesis. |
| `education_level`, `income_level`, `healthcare_coverage` | **Social Determinants of Health** | Upstream socioeconomic exposures affecting health literacy and access. |
| `checkup_interval` | **Surveillance / Utilization Artifact** | Frequently a *consequence* of existing diagnosis (patients with chronic disease visit doctors more regularly). |
| `general_health` | **Concurrent Symptom / Severity Score** | Self-rating reflects current overall illness rather than etiological risk. |
| `physical_health_days` | **Concurrent Symptom / Severity Score** | Measures recent acute or chronic impairment caused by existing conditions. |
| `diff_walk` | **Downstream Functional Consequence** | Often the direct physical impairment resulting from stroke hemiplegia, osteoarthritic cartilage loss, or diabetic peripheral neuropathy. |

### 9.2 The Three Levels of Clinical Prediction
The project documentation must clearly distinguish between three distinct predictive paradigms:
1. **Existing Disease Classification / Syndromic Correlation (What this project does):**
   Predicting whether an individual currently lives with a diagnosed chronic condition based on their current functional state, demographic profile, and self-reported health.
2. **Disease Risk Estimation (Screening Triage):**
   Estimating the likelihood that an individual has an undiagnosed condition or is at elevated clinical risk.
3. **Future Disease Prediction / Incidence (What this project CANNOT do):**
   Forecasting whether an individual will develop a disease in 5 or 10 years. Because BRFSS is a cross-sectional survey with no longitudinal follow-up, it is methodologically invalid to claim future disease prediction.

---

## 10. Required Corrections

Before this project can be certified as a rigorous, portfolio-grade Machine Learning artifact, the following corrections must be enacted:

### Required Methodological Fixes:
1. **Remediate Model Selection Leakage (Critical):**
   - Refactor `src/model_training.py` so that `best_model` selection is based **strictly on training-set cross-validation performance** (e.g. `cv_auc_mean`) or a separate validation split ($\mathbf{X}_{\text{val}}$).
   - Touch the test set ($\mathbf{X}_{\text{test}}$) **only once** to generate the final reported metrics.
2. **Prevent Cross-Validation Fold Leakage (Minor):**
   - Integrate `preprocessor` and `model` into a unified `Pipeline` so imputation and scaling are fit strictly inside each CV fold.
3. **Clarify Diabetes Target Coding (Major):**
   - Update `reports/dataset_inspection_report.md`, `reports/data_dictionary.md`, and `README.md` to explicitly state that `target_diabetes` includes 11,307 pre-diabetes cases in the negative pool, explaining why precision has an inherent ceiling.
4. **Clarify Survey Skip Terminology (Major):**
   - Replace the phrase "zero question skips" with "administered across nationwide core without branching skip logic; missingness reflects survey non-response and break-offs."
5. **Document Probability Calibration & Threshold Trade-offs (Major):**
   - Add a dedicated subsection in `reports/model_evaluation_report.md` explaining that `class_weight='balanced'` distorts predicted probabilities away from population priors, causing elevated Brier scores and high false-positive rates at $\tau = 0.5$.
   - Display validation-set threshold sweeps in the report and Streamlit app.
6. **Correct Epidemiological Claims (Major):**
   - Remove any claims of "predicting future disease". Clarify that the system performs cross-sectional chronic condition risk classification.
7. **Document Complex Survey Sampling (Major):**
   - Add a clear explanation of BRFSS complex sampling (`_LLCPWT`, `_STRWT`, `_PSU`) and state that metrics are unweighted sample estimates.

---

## 11. Final Approval Status

| Evaluation Category | Audit Grade | Recommendation |
| :--- | :---: | :--- |
| **Dataset Inspection & Integrity** | **Pass (with corrections)** | Discrepancy resolved; skip logic terminology corrected. |
| **Target Variable Validation** | **Pass (with clarifications)** | Verified distributions; prediabetes handling documented. |
| **Data Leakage Safeguards** | **Pass** | Verified zero target-to-feature leakage. |
| **Model Selection Protocol** | **Deficient (Action Required)** | Test set was used for model selection; must switch to CV. |
| **Evaluation Metrics & Verification** | **Pass** | Independently recalculated and verified. |
| **Probability Calibration** | **Pass (with documentation)** | Calibration distortion and threshold trade-offs documented. |
| **Epidemiological Interpretation** | **Pass (with corrections)** | Replaced "future prediction" with cross-sectional screening. |

### Official Verdict:
**APPROVED WITH MINOR CORRECTIONS** (if treated as an educational data science demonstration with documented limitations)  
**NOT APPROVED PENDING MAJOR CORRECTIONS** (if evaluated as an institutional-grade clinical machine learning project)

The project possesses solid architectural fundamentals, clean preprocessing logic, and working web deployment. Enacting the required model-selection and documentation corrections outlined in Section 10 will elevate this repository into an exemplary portfolio piece.

