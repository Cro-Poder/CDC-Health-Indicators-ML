# Multi-Disease Classification System Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-brightgreen.svg)](https://streamlit.io/)
[![Tests: 11 Passed](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)](https://docs.python.org/3/library/unittest.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, methodologically audited Machine Learning multi-disease classification system built on the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024** annual survey dataset (457,670 nationwide records).

---

## 1. Project Introduction & Educational Scope

Chronic non-communicable diseases—including diabetes, cardiovascular disease, stroke, chronic pulmonary disease, kidney failure, and arthritis—represent the leading causes of morbidity and healthcare costs worldwide. 

This project implements an end-to-end machine learning system designed as part of **CodeAlpha Task 4**. It is characterized as **an educational multi-disease classification system using cross-sectional, self-reported CDC BRFSS survey data**. It classifies whether a respondent's self-reported lifestyle, demographic, and health indicators statistically associate with existing diagnoses across **7 major chronic conditions** simultaneously.

It is **not** a prospective disease forecasting engine, does **not** predict future disease onset, does **not** provide clinical diagnosis, and does **not** replace laboratory biomarker testing.

---

## 2. Methodological Remediation Highlights

Following an independent methodological audit and verification, the system was refactored to eliminate all test-set snooping and data leakage:

1. **Strict 3-Way Stratified Data Splitting (60% Train / 20% Validation / 20% Test):**
   - **Training Set ($N = 60,000$):** Used exclusively for candidate model fitting and 3-fold stratified cross-validation.
   - **Validation Set ($N = 20,000$):** Used exclusively for probability calibration analysis and decision-threshold optimization.
   - **Dedicated Holdout Test Set ($N = 20,000$):** Strictly locked and segregated before model comparison or threshold tuning; accessed strictly once for final reporting.
2. **Zero Preprocessing Leakage Inside Cross-Validation:**
   - Preprocessing steps (median/mode imputation, standard scaling, and one-hot encoding) are enclosed within unified `sklearn.pipeline.Pipeline` objects.
   - Imputers and scalers fit independently inside each training fold during `cross_val_score`, completely preventing validation fold leakage.
3. **Model Selection Exclusively via Cross-Validation:**
   - Winning algorithms are selected **strictly by mean training cross-validation ROC-AUC**. Test-set metrics never influence model selection, hyperparameters, or feature choices.
4. **Validation-Based Threshold Optimization:**
   - Decision thresholds $\tau^*$ were selected on validation data using recall-constrained F1 optimization (target Recall $\ge 70\%$ while maximizing F1, with Youden's J fallback for Stroke and Kidney Disease), before evaluating on the test set.
   - Note: Operating on independent test samples is subject to validation-sample variability (e.g., COPD holdout sensitivity reached 69.36% vs. the 70.0% validation target).
5. **Probability Calibration Transparency:**
   - Evaluated Brier scores against baseline prevalences ($p(1-p)$). Positive Brier Skill Scores (Stroke, Kidney, Arthritis) indicate MSE improvement over naive base rates but do not establish clinical calibration. Raw model probabilities reflect class-weighted rank-ordering indices.

---

## 3. Dataset Source & Description

- **Source:** Centers for Disease Control and Prevention (CDC) — Behavioral Risk Factor Surveillance System (BRFSS) 2024 Annual Release.
- **Official Documentation & Codebook:** [CDC BRFSS 2024 Annual Data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
- **Raw File Format:** SAS XPORT transport format (`LLCP2024.XPT`, ~1.02 GB).
- **Raw Dimensions:** 457,670 survey respondents, 301 variables.
- **Clean Complete Sample:** 444,259 respondents (97.07%) with valid, non-missing responses across all 7 targets.

### Target Availability & Survey Rotation Schedule:
- **Hypertension & Cholesterol Absence:** In accordance with the CDC rotating core schedule, questions on high blood pressure (`BPHIGH4`) and cholesterol screening (`TOLDHI3`) are collected in **odd-numbered survey years only** (2021, 2023, 2025). They are absent from the 2024 core module and are not artificially imputed.
- **The 7 Verified Chronic Disease Targets:**
  1. **Diabetes Mellitus** (`DIABETE4`): 14.38% prevalence
  2. **Heart Disease (CHD / MI)** (`_MICHD`): 9.25% prevalence
  3. **Stroke** (`CVDSTRK3`): 4.51% prevalence
  4. **Asthma** (`ASTHMA3`): 15.74% prevalence
  5. **COPD / Chronic Bronchitis / Emphysema** (`CHCCOPD3`): 8.07% prevalence
  6. **Chronic Kidney Disease** (`CHCKDNY2`): 5.19% prevalence
  7. **Arthritis** (`HAVARTH4`): 34.59% prevalence

### Survey Administration & Missingness:
The chronic disease questions are administered through the nationwide core questionnaire without conditional branching skip logic. Missing values nevertheless occur due to item non-response, "Don't know" responses, refusals, and interview break-offs. Respondents answering "Don't know" or "Refused" are excluded from target evaluation and never treated as negative cases.

### Diabetes Coding Clarification:
In official CDC coding (`DIABETE4`):
- Code 1: Positive (diagnosed chronic diabetes).
- Codes 2 (gestational only, N=3,395), 3 (never told, N=376,950), and 4 (prediabetes / borderline, N=11,307): Negative class.
- The negative class includes respondents reporting prediabetes and gestational-only diabetes; it must not be described as a "healthy" or "normoglycemic" control pool because the dataset does not contain laboratory biomarker measurements (fasting glucose, HbA1c) needed to establish normoglycemia.

### Complex Survey Sampling Design:
BRFSS utilizes a complex, multistage stratified cluster sampling design defined by:
- `_LLCPWT`: Final post-stratification / raking sampling weight assigned to each respondent.
- `_STSTR`: Stratification identifier.
- `_PSU`: Primary sampling unit (cluster).
- `_STRWT`: Stratum weight (not the stratification identifier).
*Model Scope:* Models are trained unweighted. Reported metrics apply to the analyzed survey cohort and are **not population-representative national estimates**.

---

## 4. Final Remediated Evaluation Results (Holdout Test Set, N=20,000)

| Chronic Condition | Winning Model (Selected by CV) | Train CV ROC-AUC | Holdout ROC-AUC | Holdout PR-AUC | Selected Threshold ($\tau^*$) | Recall (@ $\tau^*$) | Specificity (@ $\tau^*$) | Precision (@ $\tau^*$) | F1-Score (@ $\tau^*$) | Pred. Pos. Rate (@ $\tau^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diabetes Mellitus** | **XGBoost** | 0.8089 $\pm$ 0.0005 | **0.8051** | 0.3806 | **0.55** | 73.13% | 72.84% | 30.96% | **0.4350** | 33.7% |
| **Heart Disease (CHD / MI)** | **Logistic Regression** | 0.8194 $\pm$ 0.0007 | **0.8262** | 0.3101 | **0.55** | 74.88% | 75.79% | 23.72% | **0.3602** | 28.8% |
| **Stroke** | **Linear SVM (Calibrated)** | 0.8021 $\pm$ 0.0073 | **0.7949** | 0.1386 | **0.10** | 38.17% | 89.75% | 14.67% | **0.2119** | 11.5% |
| **Asthma** | **Random Forest** | 0.6607 $\pm$ 0.0058 | **0.6630** | 0.2720 | **0.40** | 79.52% | 40.64% | 19.79% | **0.3170** | 62.5% |
| **COPD / Chronic Bronchitis** | **Random Forest** | 0.8412 $\pm$ 0.0006 | **0.8318** | 0.3339 | **0.50** | 69.36% | 80.77% | 23.51% | **0.3511** | 23.2% |
| **Chronic Kidney Disease** | **Linear SVM (Calibrated)** | 0.7828 $\pm$ 0.0047 | **0.8017** | 0.1684 | **0.10** | 45.61% | 88.34% | 17.46% | **0.2526** | 13.4% |
| **Arthritis** | **XGBoost** | 0.8089 $\pm$ 0.0016 | **0.8154** | 0.6777 | **0.50** | 78.63% | 69.21% | 57.58% | **0.6648** | 47.4% |

*Holdout Operational Notes:*
- **COPD Sensitivity:** Reached 69.36% on the holdout test set (slightly below the 70.0% validation target) due to expected validation-sample variability.
- **Stroke & Kidney Disease:** Platt-calibrated probabilities rarely cross 0.50; thresholds $\tau^* = 0.10$ were selected via Youden's J optimization on validation data to achieve balanced classification utility.

---

## 5. Project Architecture & Directory Layout

```
CodeAlpha_Multi_Disease_Prediction/
├── data/
│   ├── raw/
│   │   └── LLCP2024.XPT              # Raw SAS Transport dataset (~1.02 GB)
│   └── processed/
│       ├── brfss_cleaned.parquet     # Processed clean dataset (444,259 rows, ~3.3 MB)
│       └── brfss_cleaned_sample.csv  # 25,000-row representative CSV sample
│
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb   # Executed EDA with rendered plots & tables
│   └── 02_Model_Training.ipynb              # Executed training, CV & benchmark results
│
├── models/
│   ├── diabetes_model.pkl            # Remediated Pipeline bundle (XGBoost)
│   ├── heart_disease_model.pkl       # Remediated Pipeline bundle (Logistic Regression)
│   ├── stroke_model.pkl              # Remediated Pipeline bundle (Calibrated Linear SVM)
│   ├── asthma_model.pkl              # Remediated Pipeline bundle (Random Forest)
│   ├── copd_model.pkl                # Remediated Pipeline bundle (Random Forest)
│   ├── kidney_disease_model.pkl      # Remediated Pipeline bundle (Calibrated Linear SVM)
│   ├── arthritis_model.pkl           # Remediated Pipeline bundle (XGBoost)
│   └── model_metadata.json           # Model parameters & holdout evaluation metrics
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Cleaning, CDC code translation, ColumnTransformer
│   ├── model_training.py             # Pipeline CV, threshold tuning, holdout evaluation
│   ├── model_evaluation.py           # Metrics, Brier score, threshold grid, diagnostic plots
│   └── prediction.py                 # Multi-disease inference engine & demo profiles
│
├── reports/
│   ├── dataset_inspection_report.md          # Phase 1 empirical audit & codebook verification
│   ├── data_dictionary.md                    # Complete feature-by-feature dictionary
│   ├── model_evaluation_report.md            # Diagnostic report & benchmark analysis
│   ├── independent_methodological_audit.md   # Independent audit identifying methodological defects
│   ├── methodological_remediation_report.md  # Comprehensive remediation verification report
│   ├── final_verification_audit.md           # Independent post-remediation audit verification
│   ├── final_documentation_correction_report.md # Final audit correction synthesis & verification
│   ├── model_comparison_summary.csv          # Holdout test comparison table
│   ├── threshold_calibration_summary.csv     # Brier scores & threshold rationale
│   └── figures/                              # 35 high-resolution 300 DPI diagnostic plots
│
├── tests/
│   └── test_pipeline.py              # Unit & integration test suite (11 tests, all passing)
│
├── app.py                            # Streamlit 5-section healthcare web application
├── requirements.txt                  # Python dependency specifications
├── README.md                         # Comprehensive project documentation
└── .gitignore                        # Git exclusion rules
```

---

## 6. How to Install & Run

### 6.1 Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/CodeAlpha_Multi_Disease_Prediction.git
cd CodeAlpha_Multi_Disease_Prediction

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Data Preprocessing & Model Training
```bash
# 1. Clean raw CDC BRFSS dataset
python src/data_preprocessing.py

# 2. Train models with CV model selection, validation threshold tuning, and holdout evaluation
python src/model_training.py
```

### 6.3 Run Automated Test Suite
```bash
python -m unittest tests/test_pipeline.py
```

### 6.4 Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Limitations & Clinical Boundary Disclaimer

> [!CAUTION]
> **Educational Demonstration Only:**
> Predictions generated by these models are statistical associations based on self-reported population survey data. They **are not clinical diagnoses** and must never replace consultation with a licensed healthcare provider, diagnostic laboratory testing (e.g., HbA1c, eGFR, lipid panel), or medical imaging.

### Known Limitations:
1. **Cross-Sectional Scope:** Features like `GENHLTH`, `PHYSHLTH`, and `DIFFWALK` are concurrent correlates of existing illness, not pre-diagnostic risk factors. The model cannot forecast future disease onset.
2. **Observational Self-Reported Data:** Subject to recall bias and lack of access to care (individuals without routine checkups may be underdiagnosed).
3. **Absence of Biomarkers:** BRFSS does not measure continuous clinical lab values (e.g., serum creatinine, fasting blood glucose).
4. **Uncalibrated Model Scores:** Class weighting shifts predicted probabilities; model outputs represent relative ranking scores rather than clinically calibrated probabilities of absolute disease incidence.
5. **Threshold Generalization:** Thresholds optimized on validation data are subject to sampling variability and do not guarantee identical sensitivity on independent test samples (e.g., COPD sensitivity fell from 71.59% on validation to 69.36% on test).
6. **Unweighted Models:** Metrics describe the survey sample, not weighted US population rates.
