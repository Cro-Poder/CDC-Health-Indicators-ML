# Data Dictionary — CDC BRFSS 2024 Multi-Disease Prediction System

**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 Annual Release  
**Project:** CodeAlpha Task 4 — Multi-Disease Prediction Using Machine Learning  
**Official Codebook:** [CDC 2024 BRFSS Codebook](https://www.cdc.gov/brfss/annual_data/annual_2024.html)  
**Methodological Scope:** Cross-sectional educational chronic disease classification  

---

## 1. Disease Targets (Outputs / Dependent Variables)

The multi-disease prediction system predicts each of the following 7 binary outcomes independently. All disease outcomes are **strictly excluded** from the input predictor feature set of every model to prevent data leakage.

The chronic disease questions are administered through the nationwide core questionnaire without conditional branching skip logic. Missing values nevertheless occur due to item non-response, "Don't know" responses, refusals, and interview break-offs. Complete data across all 7 targets is present for **444,259 respondents (97.07%)**, with 13,411 records (2.93%) exhibiting missingness on one or more conditions.

| Target Variable | CDC Raw Variable | Clinical Disease | Raw CDC Survey Coding | Transformed Binary Target | Full Dataset Counts / Prevalence | Missing / Excluded (N / %) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `target_diabetes` | `DIABETE4` | **Diabetes Mellitus** | `1`: Yes<br>`2`: Yes, pregnancy only (N=3,395)<br>`3`: No (N=376,950)<br>`4`: Prediabetes / borderline (N=11,307)<br>`7`: Don't know<br>`9`: Refused | `1`: Positive (`1`)<br>`0`: Negative (`2`, `3`, `4`)<br>`NaN`: Excluded (`7`, `9`, Blank) | 65,809 / 14.38% | 1,607 (0.35%) |
| `target_heart_disease` | `_MICHD` | **Heart Disease (CHD / MI)** | `1`: Reported CHD or MI<br>`2`: Did not report CHD or MI<br>`Blank`: Missing | `1`: Positive (`1`)<br>`0`: Negative (`2`)<br>`NaN`: Excluded (`Blank`) | 42,338 / 9.25% | 5,508 (1.20%) |
| `target_stroke` | `CVDSTRK3` | **Stroke** | `1`: Yes<br>`2`: No<br>`7`: Don't know<br>`9`: Refused | `1`: Positive (`1`)<br>`0`: Negative (`2`)<br>`NaN`: Excluded (`7`, `9`, Blank) | 20,661 / 4.51% | 2,139 (0.47%) |
| `target_asthma` | `ASTHMA3` | **Asthma** | `1`: Yes<br>`2`: No<br>`7`: Don't know<br>`9`: Refused | `1`: Positive (`1`)<br>`0`: Negative (`2`)<br>`NaN`: Excluded (`7`, `9`, Blank) | 72,037 / 15.74% | 2,476 (0.54%) |
| `target_copd` | `CHCCOPD3` | **COPD / Bronchitis** | `1`: Yes<br>`2`: No<br>`7`: Don't know<br>`9`: Refused | `1`: Positive (`1`)<br>`0`: Negative (`2`)<br>`NaN`: Excluded (`7`, `9`, Blank) | 36,921 / 8.07% | 3,195 (0.70%) |
| `target_kidney_disease` | `CHCKDNY2` | **Kidney Disease** | `1`: Yes<br>`2`: No<br>`7`: Don't know<br>`9`: Refused | `1`: Positive (`1`)<br>`0`: Negative (`2`)<br>`NaN`: Excluded (`7`, `9`, Blank) | 23,752 / 5.19% | 2,906 (0.63%) |
| `target_arthritis` | `HAVARTH4` | **Arthritis** | `1`: Yes<br>`2`: No<br>`7`: Don't know<br>`9`: Refused | `1`: Positive (`1`)<br>`0`: Negative (`2`)<br>`NaN`: Excluded (`7`, `9`, Blank) | 158,319 / 34.59% | 3,745 (0.82%) |

### Target Coding Methodological Note (Diabetes)
- The negative class for `target_diabetes` includes respondents reporting prediabetes (11,307 respondents) and gestational-only diabetes (3,395 respondents).
- The negative class must not be described as "healthy" or "normoglycemic" controls because the dataset does not contain laboratory biomarker measurements (e.g. fasting glucose or HbA1c) necessary to establish normoglycemia.
- Respondents answering "Don't know" or "Refused" are excluded from target evaluation and are never treated as negative cases.

---

## 2. Input Predictor Features (Independent Variables)

These 16 features are collected from patient self-reports, behavioral responses, and demographic characteristics.

### 2.1 General Health & Functional Status Indicators

> [!NOTE]
> In cross-sectional survey data, general health rating, physical distress days, and walking difficulty represent **current health status, active symptom burden, or functional consequences** of existing chronic conditions rather than genuine pre-morbid risk factors.

| Feature Name | CDC Variable | Survey Question / Description | Data Type | Raw CDC Coding | Preprocessed Representation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `general_health` | `GENHLTH` | "Would you say that in general your health is:" | Ordinal | `1`: Excellent, `2`: Very good, `3`: Good, `4`: Fair, `5`: Poor. `7`/`9`: Missing | Numeric 1 to 5. Codes 7/9 $\to$ `NaN`; imputed with median. |
| `physical_health_days` | `PHYSHLTH` | "For how many days was your physical health not good?" (past 30 days) | Numerical | `1-30`: Days<br>**`88`: None (0 days)**<br>`77`/`99`: Missing | Float 0 to 30.<br>**`88` mapped to `0.0`**.<br>77/99 $\to$ `NaN`; median imputed. |
| `mental_health_days` | `MENTHLTH` | "For how many days was your mental health not good?" (past 30 days) | Numerical | `1-30`: Days<br>**`88`: None (0 days)**<br>`77`/`99`: Missing | Float 0 to 30.<br>**`88` mapped to `0.0`**.<br>77/99 $\to$ `NaN`; median imputed. |
| `diff_walk` | `DIFFWALK` | "Do you have serious difficulty walking or climbing stairs?" | Binary | `1`: Yes<br>`2`: No<br>`7`/`9`: Missing | Binary: `1` (Difficulty), `0` (No difficulty).<br>7/9 $\to$ `NaN`; mode imputed. |

### 2.2 Behavioral & Lifestyle Risk Factors

| Feature Name | CDC Variable | Survey Question / Description | Data Type | Raw CDC Coding | Preprocessed Representation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `exercise_any` | `EXERANY2` | "During the past month, did you participate in any physical activities?" | Binary | `1`: Yes<br>`2`: No<br>`7`/`9`: Missing | Binary: `1` (Active), `0` (Inactive).<br>7/9 $\to$ `NaN`; mode imputed. |
| `smoker_status` | `_SMOKER3` | Computed four-level smoking status | Categorical | `1`: Everyday smoker<br>`2`: Somedays smoker<br>`3`: Former smoker<br>`4`: Never smoked<br>`9`: Missing | Categorical 1–4. 9 $\to$ `NaN`; mode imputed. One-hot encoded. |
| `ecig_user` | `_CURECI3` | Computed e-cigarette usage status | Categorical | `1`: Never used<br>`2`: Current user<br>`3`: Former user<br>`9`: Missing | Categorical 1–3. 9 $\to$ `NaN`; mode imputed. One-hot encoded. |
| `heavy_drinker` | `_RFDRHV9` | Computed heavy alcohol consumption | Binary | `1`: No<br>`2`: Yes (Men >14/wk, Women >7/wk)<br>`9`: Missing | Binary: `1` (Heavy drinker), `0` (Not heavy).<br>9 $\to$ `NaN`; mode imputed. |
| `checkup_interval` | `CHECKUP1` | "About how long has it been since you last visited a doctor for a routine checkup?" | Categorical | `1`: Past year, `2`: Past 2 yrs, `3`: Past 5 yrs, `4`: 5+ yrs, `8`: Never. `7`/`9`: Missing | Categorical 1 to 5 (mapping 8=5).<br>7/9 $\to$ `NaN`; mode imputed. |

### 2.3 Anthropometric & Demographic Indicators

| Feature Name | CDC Variable | Survey Question / Description | Data Type | Raw CDC Coding | Preprocessed Representation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bmi` | `_BMI5` | Computed Body Mass Index ($\text{kg/m}^2$) | Numerical | Value multiplied by 100 (e.g. 2854 = 28.54). Blank = Missing | Continuous float: `_BMI5 / 100.0`. Valid range [12.0, 70.0]. Standard scaled. |
| `age` | `_AGE80` | Imputed age collapsed at 80 | Numerical | `18–79`: Exact age in years<br>`80`: Age $\ge 80$ | Continuous integer (18–80). Standard scaled. |
| `sex` | `_SEX` | Biological sex | Binary | `1`: Male<br>`2`: Female | Binary: `1` = Male, `0` = Female. |
| `education_level` | `_EDUCAG` | Computed education completed | Categorical / Ordinal | `1`: < HS, `2`: HS Grad, `3`: Some College, `4`: College Grad. `9`: Missing | Ordinal scale 1 to 4. 9 $\to$ `NaN`; mode imputed. |
| `income_level` | `_INCOMG1` | Computed annual household income | Categorical / Ordinal | `1`: < \$15k ... `7`: \$200k+. `9`: Missing | Ordinal scale 1 to 7. 9 $\to$ `NaN`; mode imputed. |
| `race_ethnicity` | `_RACEGR3` | Computed five-level race/ethnicity | Categorical | `1`: White, `2`: Black, `3`: Other, `4`: Multi, `5`: Hispanic. `9`: Missing | Categorical 1–5. 9 $\to$ `NaN`; mode imputed. One-hot encoded. |
| `healthcare_coverage` | `_HLTHPL2` | "Do you have any kind of health care coverage?" | Binary | `1`: Yes<br>`2`: No<br>`9`: Missing | Binary: `1` (Insured), `0` (Uninsured).<br>9 $\to$ `NaN`; mode imputed. |

---

## 3. Complex Survey Sampling Variables & Design Parameters

The BRFSS survey utilizes a complex, multistage stratified cluster sampling design. Official CDC 2024 variable definitions:

- `_LLCPWT`: **Final sampling weight**. Post-stratification / iterative proportional fitting (raking) weight assigned to each respondent.
- `_STSTR`: **Stratification identifier**. Identifies geographic and sampling strata within each state.
- `_PSU`: **Primary sampling unit**. Identifies clustering in the survey sampling frame.
- `_STRWT`: **Stratum weight**. Represents the stratum sampling weight (not the stratification identifier).

### Unweighted Machine Learning Model Scope:
The predictive models in this educational repository are trained without survey weights. Therefore:
- Model performance metrics (Accuracy, ROC-AUC, Recall, Precision, Brier score) describe classification performance **strictly on the analyzed survey sample**.
- These metrics are **not population-representative estimates** of disease prevalence or diagnostic accuracy across the entire United States.
- The system should not be claimed to possess national clinical validity or population-level predictive performance.
