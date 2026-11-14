# CDC BRFSS 2024 Dataset Inspection and Validation Report

**Project:** CodeAlpha Task 4 — Multi-Disease Prediction Using Machine Learning  
**Author:** Antigravity (ML Engineer & Technical Mentor)  
**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 Annual Survey  
**Official Source:** [CDC BRFSS 2024 Documentation](https://www.cdc.gov/brfss/annual_data/annual_2024.html)  
**Status:** Methodologically Remediated & Verified  

---

## 1. Executive Summary

This inspection report provides a comprehensive, empirical audit of the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 annual survey dataset stored locally in SAS Transport (`.XPT`) format (`LLCP2024.XPT`).

### Educational Scope & System Characterization
This project is designed as **an educational multi-disease classification system using cross-sectional, self-reported CDC BRFSS survey data**. It classifies existing self-reported chronic disease status based on demographic, behavioral, socioeconomic, and current health indicators.

It is **not** a prospective disease forecasting model, does **not** predict future disease onset, does **not** provide clinical diagnosis, does **not** replace laboratory biomarker testing, and does **not** estimate an individual's medically calibrated absolute clinical risk.

### Key Inspection Findings:
1. **File Specifications:** The raw file `LLCP2024.XPT` is 1,093,874,240 bytes (~1.02 GB / 1,043 MB).
2. **Dimensions:** The raw dataset contains **457,670 respondent records** and **301 columns**. (The 345 count mentioned in CDC documentation layout tables represents all possible module slots across all interview versions; the unified release file contains 301).
3. **Core Administration & Missingness:** The chronic disease questions are administered through the nationwide core questionnaire without conditional branching skip logic. Missing values nevertheless occur due to item non-response, "Don't know" responses, refusals, and interview break-offs. Complete data across all 7 verified targets is present for **444,259 respondents (97.07%)**, with 13,411 records (2.93%) exhibiting non-response on at least one condition.
4. **Target Availability Verification:**
   - **Confirmed Core Diseases (7 Targets):** Diabetes (`DIABETE4`), Heart Disease (`_MICHD`), Stroke (`CVDSTRK3`), Asthma (`ASTHMA3`), Chronic Obstructive Pulmonary Disease (`CHCCOPD3`), Kidney Disease (`CHCKDNY2`), and Arthritis (`HAVARTH4`).
   - **Survey Rotation Absence (Hypertension & Cholesterol):** In accordance with the CDC rotating core schedule, questions on high blood pressure (`BPHIGH4`) and cholesterol screening (`TOLDHI3`) are collected in **odd-numbered survey years only** (e.g., 2021, 2023, 2025). They are completely absent from the 2024 core dataset. We do not synthesize or impute nonexistent variables.
5. **Complex Survey Sampling Design:** BRFSS employs a complex, multistage stratified cluster sampling design defined by:
   - `_LLCPWT`: Final post-stratification / raking sampling weight assigned to each respondent.
   - `_STSTR`: Stratification identifier.
   - `_PSU`: Primary sampling unit (cluster).
   - `_STRWT`: Stratum weight (not the stratification identifier).
   All predictive models in this educational demonstration are unweighted. Consequently, unweighted metrics describe classification performance strictly on the analyzed survey sample and are **not population-representative estimates**.

---

## 2. Dataset Overview and Technical Profile

| Metric / Dimension | Value | Methodological Notes |
| :--- | :--- | :--- |
| **File Name** | `LLCP2024.XPT` | Preserved read-only in `data/raw/` and workspace root |
| **File Format** | SAS XPORT (V8/V9 Transport) | Read using C-optimized `pyreadstat` (`encoding='latin1'`) |
| **Raw File Size** | 1.02 GB (1,093,874,240 bytes) | Uncompressed binary SAS format |
| **Total Survey Respondents** | **457,670** | Non-institutionalized US adults aged $\ge 18$ |
| **Total Raw Columns** | **301** | Unified public release layout |
| **Clean Complete Records** | **444,259** (97.07%) | Non-missing across all 7 selected chronic disease targets |
| **Clean Feature Set** | **16 features** | Demographic, behavioral, lifestyle, and health indicators |
| **Survey Year** | 2024 | Data collected across all 50 US States, DC, PR, Guam, USVI |

---

## 3. Disease Target Investigation and Coding Verification

Each disease outcome was audited against the official CDC 2024 variable codebook and empirical distributions in the dataset.

### Detailed Target Breakdown & Missingness Audit

| Disease Target | CDC Variable | Description | Positive Count / % | Missing / Excluded (N / %) | Target Coding & Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | `DIABETE4` | (Ever told) you had diabetes | 65,809 (14.38%) | 1,607 (0.35%) | **Positive (1):** Diagnosed diabetes (`1`).<br>**Negative (0):** Gestational diabetes only (`2`, N=3,395), Never told (`3`, N=376,950), Prediabetes / borderline (`4`, N=11,307).<br>**Excluded:** Codes `7`, `9`, Blank $\to$ `NaN`. |
| **Heart Disease** | `_MICHD` | Reported CHD or Myocardial Infarction | 42,338 (9.25%) | 5,508 (1.20%) | **Positive (1):** Reported CHD or MI (`1`).<br>**Negative (0):** Did not report CHD or MI (`2`).<br>**Excluded:** Missing blanks $\to$ `NaN`. |
| **Stroke** | `CVDSTRK3` | Ever diagnosed with a stroke | 20,661 (4.51%) | 2,139 (0.47%) | **Positive (1):** Yes (`1`).<br>**Negative (0):** No (`2`).<br>**Excluded:** Codes `7`, `9`, Blank $\to$ `NaN`. |
| **Asthma** | `ASTHMA3` | Ever told had asthma | 72,037 (15.74%) | 2,476 (0.54%) | **Positive (1):** Yes (`1`).<br>**Negative (0):** No (`2`).<br>**Excluded:** Codes `7`, `9`, Blank $\to$ `NaN`. |
| **COPD** | `CHCCOPD3` | Ever told had COPD, emphysema, bronchitis | 36,921 (8.07%) | 3,195 (0.70%) | **Positive (1):** Yes (`1`).<br>**Negative (0):** No (`2`).<br>**Excluded:** Codes `7`, `9`, Blank $\to$ `NaN`. |
| **Kidney Disease** | `CHCKDNY2` | Ever told had kidney disease (excl. stones) | 23,752 (5.19%) | 2,906 (0.63%) | **Positive (1):** Yes (`1`).<br>**Negative (0):** No (`2`).<br>**Excluded:** Codes `7`, `9`, Blank $\to$ `NaN`. |
| **Arthritis** | `HAVARTH4` | Told had arthritis, gout, lupus, or FM | 158,319 (34.59%) | 3,745 (0.82%) | **Positive (1):** Yes (`1`).<br>**Negative (0):** No (`2`).<br>**Excluded:** Codes `7`, `9`, Blank $\to$ `NaN`. |

### Critical Clarifications on Diabetes Coding
- `DIABETE4 = 1` is assigned to the positive class (diagnosed chronic diabetes).
- `DIABETE4 = 2` (gestational diabetes only, 3,395 respondents), `DIABETE4 = 3` (never told, 376,950 respondents), and `DIABETE4 = 4` (prediabetes or borderline diabetes, 11,307 respondents) are assigned to the negative class.
- **Important:** The negative class includes respondents reporting prediabetes and gestational-only diabetes. Therefore, the negative class cannot and must not be described as "healthy" or "normoglycemic" controls, because the dataset does not contain laboratory measurements (e.g. fasting plasma glucose or HbA1c) necessary to establish normoglycemia.
- Including prediabetes in the negative pool introduces clinical heterogeneity into the non-diabetic class, which is transparently documented.

---

## 4. Feature Space: Demographic, Behavioral, and Health Features

A standardized pool of **16 predictor features** was curated across demographic, lifestyle, and health indicators.

| Feature Name | CDC Variable | Type | Raw Survey Coding | Handled Representation | Epidemiological Caveat |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **General Health** | `GENHLTH` | Ordinal | 1=Excellent ... 5=Poor. 7/9=Missing. | 1 to 5 scale; 7/9 $\to$ NaN | **Correlate / Consequence:** Often a reflection of existing morbidity rather than pre-diagnostic cause. |
| **Physical Health Days** | `PHYSHLTH` | Numerical | 1–30 days. **88=None (0 days)**. 77/99=Missing. | **88 $\to$ 0.0**. 1–30 kept as continuous days. | **Symptom Burden:** High values indicate active disability and chronic disease burden. |
| **Mental Health Days** | `MENTHLTH` | Numerical | 1–30 days. **88=None (0 days)**. 77/99=Missing. | **88 $\to$ 0.0**. 1–30 kept as continuous days. | Confounding distress / comorbidity factor. |
| **Difficulty Walking** | `DIFFWALK` | Binary | 1=Yes, 2=No. 7/9=Missing. | Binary: 1 = Difficulty, 0 = No. | **Functional Consequence:** Highly correlated with arthritis, stroke, and vascular disease. |
| **Physical Activity** | `EXERANY2` | Binary | 1=Yes, 2=No. 7/9=Missing. | Binary: 1 = Active, 0 = Inactive. | Behavioral risk factor / functional capability. |
| **Smoking Status** | `_SMOKER3` | Categorical | 1=Everyday, 2=Somedays, 3=Former, 4=Never. 9=Missing. | One-hot encoded across 4 levels. | Established lifestyle etiology. |
| **E-Cigarette Status** | `_CURECI3` | Categorical | 1=Never, 2=Current, 3=Former. 9=Missing. | One-hot encoded across 3 levels. | Established lifestyle etiology. |
| **Heavy Alcohol Drinker** | `_RFDRHV9` | Binary | 1=No, 2=Yes (>14/wk men, >7/wk women). 9=Missing. | Binary: 1 = Heavy drinker, 0 = No. | Behavioral risk factor. |
| **Routine Checkup** | `CHECKUP1` | Categorical | 1=<1yr, 2=1-2yrs, 3=2-5yrs, 4=5+yrs, 8=Never. | Categorical 1 to 5 (mapping 8=5). | Healthcare access / diagnostic opportunity. |
| **BMI** | `_BMI5` | Numerical | Value multiplied by 100. Blank=Missing. | Float: `_BMI5 / 100.0` (kg/m²). Standard scaled. | Established metabolic risk factor. |
| **Age** | `_AGE80` | Numerical | Continuous years 18 to 80 (80 = $\ge 80$). | Continuous integer (18–80). Standard scaled. | Dominant demographic determinant. |
| **Biological Sex** | `_SEX` | Binary | 1=Male, 2=Female. | Binary: 1 = Male, 0 = Female. | Demographic baseline. |
| **Education Completed** | `_EDUCAG` | Categorical | 1=<HS, 2=HS Grad, 3=Some College, 4=College Grad. | Ordinal scale 1 to 4. Mode imputed. | Socioeconomic determinant. |
| **Household Income** | `_INCOMG1` | Categorical | 1 (<$15k) to 7 ($200k+). 9=Missing. | Ordinal scale 1 to 7. Mode imputed. | Socioeconomic determinant. |
| **Race / Ethnicity** | `_RACEGR3` | Categorical | 1=White, 2=Black, 3=Other, 4=Multi, 5=Hispanic. | One-hot encoded across 5 groups. | Demographic / social health determinant. |
| **Healthcare Coverage** | `_HLTHPL2` | Binary | 1=Yes, 2=No. 9=Missing. | Binary: 1 = Insured, 0 = Uninsured. | Diagnostic detection bias factor. |

---

## 5. Critical Missing Value Codes and Survey Semantics

Survey data contains specific codes that must not be treated as literal numerical values:

1. **The Code `88` Trap in `PHYSHLTH` and `MENTHLTH`:**
   - In CDC BRFSS, respondents reporting zero days of poor physical or mental health are assigned code `88` ("None").
   - Treating `88` as a numeric value would falsely imply that a healthy individual suffered 88 days of illness in a 30-day month.
   - Conversely, treating `88` as missing (`NaN`) would incorrectly drop or mis-impute over 267,000 respondents.
   - **Correct Treatment:** Explicitly remap `88 -> 0.0`.
2. **Codes `7`, `77`, `777` (Don't Know / Not Sure):**
   - Represents cognitive uncertainty; mapped to `np.nan` and imputed via median (numerical) or mode (categorical).
3. **Codes `9`, `99`, `999` (Refused):**
   - Represents survey non-response; mapped to `np.nan` and imputed via median or mode.
4. **Calculated Metric Offsets (`_BMI5`):**
   - In `_BMI5`, two implied decimal places are used by SAS (`2854` = `28.54`). We divide by 100 to yield legitimate SI units ($\text{kg/m}^2$).

---

## 6. Strict Data Leakage Prevention Protocol

Data leakage occurs when target or post-diagnosis information enters the feature space or training pipeline:

1. **Target Exclusion:** All 7 disease variables are strictly excluded from the predictor feature set $\mathbf{X}$. Model A predicting Heart Disease never receives Diabetes, Stroke, or any other disease indicator as a predictor.
2. **Preprocessing Pipeline Isolation:** All imputers, encoders, and scalers are enclosed inside scikit-learn `Pipeline` objects. Preprocessing parameters are fitted strictly inside each training fold during cross-validation, and never on validation or holdout test sets.
3. **Dedicated Final Holdout Test Set:** 20% of the active dataset is strictly segregated before any model comparison or threshold tuning occurs, and is accessed only once for final reporting.

---

## 7. Suitability and Methodological Boundaries

### Why BRFSS 2024 is Suitable for Educational Classification:
- **Large Sample Size:** Over 444,000 complete records provide statistical power and prevent overfitting.
- **Realistic Chronic Disease Prevalences:** Matches real-world epidemiology (e.g. Diabetes ~14%, Heart Disease ~9%, Stroke ~4.5%).
- **Standardized Survey Administration:** Nationwide coverage provides demographic and socioeconomic diversity.

### Crucial Methodological Boundaries & Limitations:
1. **Self-Reported Survey Data:** Diagnoses are self-reported ("Has a doctor ever told you..."), introducing recall bias and healthcare access detection bias.
2. **Cross-Sectional Design:** Data captures a single point in time. It classifies existing condition status, but **cannot predict future disease incidence** or establish longitudinal causality.
3. **Absence of Clinical Biomarkers:** BRFSS contains no laboratory values (HbA1c, fasting glucose, eGFR, lipid panels, continuous blood pressure).
4. **Unweighted Survey Scope:** Models are trained without complex sampling weights (`_LLCPWT`), meaning performance metrics apply to the survey sample rather than the true weighted US population.
5. **Class Imbalance & Probability Calibration:** Due to class imbalance and balanced weighting, raw probability outputs are not calibrated population probabilities and must be interpreted as relative risk indices.

---

## 8. Conclusion and Sign-Off

The BRFSS 2024 dataset provides a robust, real-world foundation for educational multi-disease classification. The data has been rigorously inspected, CDC coding rules have been applied, and strict methodological safeguards against data leakage and over-interpretation have been enacted.
