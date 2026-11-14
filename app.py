"""
CodeAlpha Task 4 — Multi-Disease Prediction Healthcare Dashboard
Author: Antigravity (ML Engineer & Technical Mentor)
Dataset: CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024
Methodological Scope: Educational Multi-Disease Classification
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.prediction import MultiDiseasePredictor, get_sample_patient_profiles
from src.data_preprocessing import CLEAN_FEATURE_NAMES
from src.model_training import DISEASE_METADATA


# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Disease Educational Classification Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern healthcare aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 14px 20px;
        border-radius: 8px;
        border-left: 5px solid #ffeeba;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .info-box {
        background-color: #e8f4f8;
        color: #1a5276;
        padding: 12px 18px;
        border-radius: 8px;
        border-left: 5px solid #2980b9;
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .risk-high {
        color: #d9534f;
        font-weight: bold;
    }
    .risk-mod {
        color: #f0ad4e;
        font-weight: bold;
    }
    .risk-low {
        color: #5cb85c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# INITIALIZE PREDICTOR
# ---------------------------------------------------------
@st.cache_resource
def load_predictor():
    return MultiDiseasePredictor(models_dir="models")

predictor = load_predictor()


# ---------------------------------------------------------
# MANDATORY EDUCATIONAL & CLINICAL DISCLAIMER
# ---------------------------------------------------------
def render_disclaimer():
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Important Educational Classification & Clinical Boundary Notice:</strong><br>
        This application is an educational data science demonstration built on cross-sectional, self-reported 
        <strong>CDC BRFSS 2024</strong> survey data. 
        <ul>
            <li>Models classify whether a respondent's self-reported lifestyle and demographic indicators align with individuals reporting an existing condition.</li>
            <li>This system does <strong>not</strong> predict future disease onset, does <strong>not</strong> diagnose medical conditions, and does <strong>not</strong> replace clinical laboratory testing (such as HbA1c, fasting glucose, eGFR, or clinical imaging).</li>
            <li>Predicted probability scores reflect rank-ordering under class-weighted model optimization rather than medically calibrated population probabilities.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home / Overview",
        "Disease Classification",
        "Model Performance",
        "About the Dataset",
        "Project Limitations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**CodeAlpha Task 4 (Remediated)**\n\n"
    "• Strict 3-Way Stratified Split (60/20/20)\n"
    "• Zero Preprocessing Leakage (Pipelines)\n"
    "• Model Selection via CV ROC-AUC Only\n"
    "• Validation Threshold Optimization\n"
    "• 7 Verified CDC Chronic Diseases"
)


# ---------------------------------------------------------
# PAGE 1: HOME / OVERVIEW
# ---------------------------------------------------------
if page == "Home / Overview":
    st.markdown('<div class="main-header">🩺 Multi-Disease Classification Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Educational Cross-Sectional Chronic Disease Classification via CDC BRFSS 2024 Data</div>', unsafe_allow_html=True)
    render_disclaimer()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Project Objectives & Educational Scope
        This project provides an end-to-end Machine Learning pipeline that simultaneously evaluates chronic disease associations across **7 validated conditions** using standardized respondent demographic, behavioral, and self-reported health indicators:
        
        1. **Diabetes Mellitus** (XGBoost)
        2. **Heart Disease (CHD / Myocardial Infarction)** (Logistic Regression)
        3. **Stroke** (Calibrated Linear SVM)
        4. **Asthma** (Random Forest)
        5. **Chronic Obstructive Pulmonary Disease (COPD)** (Random Forest)
        6. **Chronic Kidney Disease** (Calibrated Linear SVM)
        7. **Arthritis** (XGBoost)
        
        ### Methodological Remediation Highlights:
        - **Model Selection Integrity:** Winning algorithms were selected **exclusively by mean cross-validation ROC-AUC on training data (60,000 samples)**, completely eliminating test-set snooping.
        - **Pipeline Preprocessing Isolation:** Imputation and scaling are encapsulated inside `sklearn.pipeline.Pipeline`, fitting strictly inside training CV folds without validation leakage.
        - **Validation Threshold Optimization:** Decision thresholds $\\tau^*$ were optimized on a separate validation set (20,000 samples) to achieve recall-constrained F1 optimization (target Recall $\\ge 70%$, with Youden's J fallback for Stroke and Kidney Disease), before touching the locked holdout test set (20,000 samples).
        - **Zero Target Leakage:** Every disease model is strictly barred from seeing any other disease outcomes.
        """)
    
    with col2:
        st.markdown("### Audited System Statistics")
        st.metric(label="Raw BRFSS Survey Respondents", value="457,670")
        st.metric(label="Clean Complete Cases Across 7 Targets", value="444,259 (97.1%)")
        st.metric(label="Dedicated Holdout Test Set", value="20,000 samples (LOCKED)")
        st.metric(label="Independent Disease Models", value="7 Specialized Classifiers")
        st.metric(label="Predictor Features", value="16 Non-Leaking Indicators")

    st.markdown("---")
    st.markdown("### Chronic Disease Prevalence in the Analyzed Sample (BRFSS 2024)")
    if os.path.exists("reports/figures/disease_prevalence.png"):
        st.image("reports/figures/disease_prevalence.png", use_container_width=True)


# ---------------------------------------------------------
# PAGE 2: DISEASE CLASSIFICATION
# ---------------------------------------------------------
elif page == "Disease Classification":
    st.markdown('<div class="main-header">🩺 Respondent Health Profile Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter self-reported demographic, lifestyle, and health indicators to evaluate multi-disease classification profiles.</div>', unsafe_allow_html=True)
    render_disclaimer()

    # Preset profiles
    st.markdown("#### Load Preset Respondent Profiles")
    preset_col1, preset_col2, preset_col3 = st.columns([2, 2, 4])
    sample_profiles = get_sample_patient_profiles()

    selected_preset = None
    if preset_col1.button("👤 Young Adult Profile (Lower Score)"):
        selected_preset = sample_profiles['Healthy Young Adult (Low Risk)']
    if preset_col2.button("⚠️ Senior Profile (Higher Score)"):
        selected_preset = sample_profiles['Senior with Multiple Risk Factors (High Risk)']

    if selected_preset:
        st.session_state.update(selected_preset)
        st.success("Loaded demo profile successfully!")

    st.markdown("---")
    st.markdown("### Respondent Health & Lifestyle Questionnaire")

    with st.form("patient_input_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("##### 🧬 Demographics & Body Metrics")
            age = st.slider("Age (Years)", 18, 80, int(st.session_state.get('age', 45)))
            sex_choice = st.selectbox(
                "Biological Sex",
                ["Male", "Female"],
                index=0 if st.session_state.get('sex', 1.0) == 1.0 else 1
            )
            bmi = st.number_input(
                "Body Mass Index (BMI in kg/m²)",
                min_value=12.0,
                max_value=65.0,
                value=float(st.session_state.get('bmi', 26.5)),
                step=0.1,
                help="Healthy: 18.5 - 24.9 | Overweight: 25 - 29.9 | Obese: 30+"
            )
            race_options = {
                "White non-Hispanic": 1.0,
                "Black non-Hispanic": 2.0,
                "Other race non-Hispanic": 3.0,
                "Multiracial non-Hispanic": 4.0,
                "Hispanic": 5.0
            }
            race_choice = st.selectbox("Race / Ethnicity", list(race_options.keys()))

        with col_b:
            st.markdown("##### 🩺 Current Health Status (Concurrent Correlates)")
            gh_options = {
                "1 - Excellent": 1.0,
                "2 - Very Good": 2.0,
                "3 - Good": 3.0,
                "4 - Fair": 4.0,
                "5 - Poor": 5.0
            }
            gh_choice = st.selectbox(
                "General Health Self-Rating",
                list(gh_options.keys()),
                index=int(st.session_state.get('general_health', 3.0)) - 1
            )
            phys_days = st.slider(
                "Poor Physical Health Days (past 30 days)",
                0, 30,
                int(st.session_state.get('physical_health_days', 0))
            )
            ment_days = st.slider(
                "Poor Mental Health Days (past 30 days)",
                0, 30,
                int(st.session_state.get('mental_health_days', 0))
            )
            diff_walk_choice = st.selectbox(
                "Difficulty Walking or Climbing Stairs?",
                ["No", "Yes"],
                index=int(st.session_state.get('diff_walk', 0.0))
            )

        with col_c:
            st.markdown("##### 🏃 Lifestyle & Healthcare Access")
            exer_choice = st.selectbox(
                "Physical Activity / Exercise (past 30 days)",
                ["Yes", "No"],
                index=0 if st.session_state.get('exercise_any', 1.0) == 1.0 else 1
            )
            smk_options = {
                "Never smoked": 4.0,
                "Former smoker": 3.0,
                "Current smoker (some days)": 2.0,
                "Current smoker (every day)": 1.0
            }
            smk_choice = st.selectbox("Smoking Status", list(smk_options.keys()))
            
            ecig_options = {
                "Never used e-cigarettes": 1.0,
                "Former e-cigarette user": 3.0,
                "Current e-cigarette user": 2.0
            }
            ecig_choice = st.selectbox("E-Cigarette Status", list(ecig_options.keys()))
            
            heavy_drink_choice = st.selectbox(
                "Heavy Alcohol Consumption (Men >14/wk, Women >7/wk)",
                ["No", "Yes"],
                index=int(st.session_state.get('heavy_drinker', 0.0))
            )

            checkup_options = {
                "Within past year": 1.0,
                "Within past 2 years": 2.0,
                "Within past 5 years": 3.0,
                "5 or more years ago": 4.0,
                "Never": 5.0
            }
            checkup_choice = st.selectbox("Routine Checkup Interval", list(checkup_options.keys()))

            ins_choice = st.selectbox(
                "Have Healthcare Insurance Coverage?",
                ["Yes", "No"],
                index=0 if st.session_state.get('healthcare_coverage', 1.0) == 1.0 else 1
            )

            edu_options = {
                "Graduated College / Technical School": 4.0,
                "Attended College / Technical School": 3.0,
                "Graduated High School": 2.0,
                "Did Not Graduate High School": 1.0
            }
            edu_choice = st.selectbox("Education Level", list(edu_options.keys()))

            inc_options = {
                "$100,000 to < $200,000": 6.0,
                "$50,000 to < $100,000": 5.0,
                "$35,000 to < $50,000": 4.0,
                "$25,000 to < $35,000": 3.0,
                "$15,000 to < $25,000": 2.0,
                "Less than $15,000": 1.0,
                "$200,000 or more": 7.0
            }
            inc_choice = st.selectbox("Household Income Level", list(inc_options.keys()))

        submitted = st.form_submit_button("🚀 Run Multi-Disease Classification", use_container_width=True)

    if submitted:
        patient_dict = {
            'general_health': gh_options[gh_choice],
            'physical_health_days': float(phys_days),
            'mental_health_days': float(ment_days),
            'diff_walk': 1.0 if diff_walk_choice == "Yes" else 0.0,
            'exercise_any': 1.0 if exer_choice == "Yes" else 0.0,
            'smoker_status': smk_options[smk_choice],
            'ecig_user': ecig_options[ecig_choice],
            'heavy_drinker': 1.0 if heavy_drink_choice == "Yes" else 0.0,
            'checkup_interval': checkup_options[checkup_choice],
            'bmi': float(bmi),
            'age': float(age),
            'sex': 1.0 if sex_choice == "Male" else 0.0,
            'education_level': edu_options[edu_choice],
            'income_level': inc_options[inc_choice],
            'race_ethnicity': race_options[race_choice],
            'healthcare_coverage': 1.0 if ins_choice == "Yes" else 0.0
        }

        try:
            results = predictor.predict_all_diseases(patient_dict)
            st.markdown("---")
            st.markdown("### 📊 Multi-Disease Classification Results")

            st.markdown("""
            <div class="info-box">
                ℹ️ <strong>Classification Semantics:</strong><br>
                • <strong>Positive:</strong> Respondent profile aligns with survey respondents reporting an existing condition under the validation-selected decision threshold.<br>
                • <strong>Negative:</strong> Respondent profile aligns with the non-reporting reference group.<br>
                • <strong>Score Notice:</strong> Probability scores are uncalibrated ranking scores under class-weighted model optimization and must not be interpreted as clinical probabilities or absolute diagnostic incidence.
            </div>
            """, unsafe_allow_html=True)

            res_cols = st.columns(len(results))
            for idx, (target_col, res) in enumerate(results.items()):
                with res_cols[idx]:
                    tier = res['risk_tier']
                    prob = res['probability_pct']
                    tau = res['selected_threshold']
                    label = res['prediction_label']

                    st.markdown(f"**{res['disease_name']}**")
                    if label == "Positive":
                        st.error(f"⚠️ Positive ({prob})")
                    else:
                        st.success(f"✅ Negative ({prob})")
                    st.caption(f"Model: {res['model_name']}\n\nThreshold: $\\tau^*={tau:.2f}$")

            # Detailed results table
            st.markdown("#### Classification Summary Table")
            table_data = []
            for target_col, res in results.items():
                table_data.append({
                    "Chronic Condition": res['disease_name'],
                    "Winning Model (CV)": res['model_name'],
                    "Classification Result": res['prediction_label'],
                    "Selected Threshold (τ*)": f"{res['selected_threshold']:.2f}",
                    "Default Result (τ=0.50)": res['prediction_default_label'],
                    "Model Probability Score": res['probability_pct'],
                    "Score Category": res['risk_tier']
                })
            st.table(pd.DataFrame(table_data))

        except Exception as e:
            st.error(f"Error executing prediction: {e}")


# ---------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# ---------------------------------------------------------
elif page == "Model Performance":
    st.markdown('<div class="main-header">📈 Methodologically Validated Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Holdout test set evaluation (20,000 samples) following cross-validation model selection and validation threshold tuning.</div>', unsafe_allow_html=True)
    render_disclaimer()

    # Load summary CSV if exists
    summary_path = os.path.join("reports", "model_comparison_summary.csv")
    if os.path.exists(summary_path):
        st.markdown("### 🏆 Remediated Model Selection & Holdout Test Evaluation Summary")
        st.caption("Models were chosen strictly based on Train CV ROC-AUC. Test set (20,000 samples) was touched strictly once for final reporting.")
        sum_df = pd.read_csv(summary_path)
        st.dataframe(sum_df, use_container_width=True)

    calib_summary_path = os.path.join("reports", "threshold_calibration_summary.csv")
    if os.path.exists(calib_summary_path):
        st.markdown("### 🎯 Decision Threshold Optimization & Calibration Summary")
        calib_df = pd.read_csv(calib_summary_path)
        st.dataframe(calib_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Diagnostic Plots (Confusion Matrices, ROC, PR, and Calibration Curves)")

    disease_tabs = st.tabs([
        "Diabetes", "Heart Disease", "Stroke", "Asthma", "COPD", "Kidney Disease", "Arthritis"
    ])
    slugs = ["diabetes", "heart_disease", "stroke", "asthma", "copd", "kidney_disease", "arthritis"]

    for tab, slug in zip(disease_tabs, slugs):
        with tab:
            cm_opt_path = f"reports/figures/{slug}_confusion_matrix.png"
            cm_def_path = f"reports/figures/{slug}_confusion_matrix_default.png"
            roc_path = f"reports/figures/{slug}_roc_curve.png"
            pr_path = f"reports/figures/{slug}_pr_curve.png"
            cal_path = f"reports/figures/{slug}_calibration_curve.png"

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if os.path.exists(cm_opt_path):
                    st.image(cm_opt_path, caption="Confusion Matrix @ Selected Decision Threshold (τ*)", use_container_width=True)
                if os.path.exists(roc_path):
                    st.image(roc_path, caption="ROC Curve (Holdout Test Set)", use_container_width=True)
            with col_p2:
                if os.path.exists(cm_def_path):
                    st.image(cm_def_path, caption="Confusion Matrix @ Default Threshold (τ = 0.50)", use_container_width=True)
                if os.path.exists(pr_path):
                    st.image(pr_path, caption="Precision-Recall Curve (Holdout Test Set)", use_container_width=True)

            if os.path.exists(cal_path):
                st.markdown("#### Probability Calibration (Reliability)")
                st.image(cal_path, caption="Calibration Curve (Validation Set) vs. Base Prevalence", use_container_width=True)


# ---------------------------------------------------------
# PAGE 4: ABOUT THE DATASET
# ---------------------------------------------------------
elif page == "About the Dataset":
    st.markdown('<div class="main-header">📋 About the CDC BRFSS 2024 Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data provenance, survey methodology, and variable specifications.</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("""
    ### Data Provenance & Administration
    The **Behavioral Risk Factor Surveillance System (BRFSS)** is the nation's premier system of health-related telephone surveys that collect state-level data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services.
    
    - **Sponsor:** Centers for Disease Control and Prevention (CDC)
    - **Year:** 2024 Annual Public Release
    - **Official Documentation:** [CDC BRFSS 2024 Annual Data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
    - **Total Records:** 457,670 survey respondents
    - **Total Columns in Public File:** 301 columns

    ### Complex Survey Sampling Design Variables
    Official CDC 2024 design variable definitions:
    - **`_LLCPWT`**: Final sampling weight (post-stratification / raking weight assigned to each respondent).
    - **`_STSTR`**: Stratification identifier (geographic and sampling stratum).
    - **`_PSU`**: Primary sampling unit (cluster identifier).
    - **`_STRWT`**: Stratum weight (represents the stratum sampling weight, not the stratification identifier).

    *Note on Model Scope:* All models in this educational system are unweighted. Performance metrics describe classification performance strictly on the analyzed survey sample and are **not population-representative national estimates**.

    ### CDC Survey Rotation Schedule
    - **Hypertension & Cholesterol:** Questions regarding high blood pressure (`BPHIGH4`) and cholesterol screening (`TOLDHI3`) are collected on **odd-numbered years only** (e.g. 2021, 2023, 2025). They are absent from the 2024 core module. Rather than imputing or synthesizing nonexistent data, this constraint is honestly documented.
    """)

    st.markdown("---")
    st.markdown("### Risk Factor Distributions & Demographics")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if os.path.exists("reports/figures/bmi_and_age_distributions.png"):
            st.image("reports/figures/bmi_and_age_distributions.png", caption="Age & BMI Distributions", use_container_width=True)
    with col_d2:
        if os.path.exists("reports/figures/lifestyle_risk_factors.png"):
            st.image("reports/figures/lifestyle_risk_factors.png", caption="Lifestyle Risk Factor Distributions", use_container_width=True)


# ---------------------------------------------------------
# PAGE 5: PROJECT LIMITATIONS
# ---------------------------------------------------------
elif page == "Project Limitations":
    st.markdown('<div class="main-header">⚠️ Methodological Limitations & Epidemiological Scope</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transparent disclosure of clinical, algorithmic, and survey boundaries.</div>', unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("""
    ### 1. Cross-Sectional Nature vs. Prospective Prediction
    - **No Future Prediction:** The BRFSS survey captures a single cross-sectional snapshot in time. It measures statistical associations between self-reported health factors and current disease status. It **cannot predict future disease incidence** or establish temporal causality.
    - **Symptoms / Consequences as Predictors:** Features like General Health (`GENHLTH`), Physical Health Days (`PHYSHLTH`), and Walking Difficulty (`DIFFWALK`) often reflect consequences or active symptoms of existing disease rather than genuine pre-morbid risk factors.

    ### 2. Observational Self-Reported Survey Data
    - **Recall Bias & Detection Bias:** All medical diagnoses are self-reported by respondents over the telephone without clinical record verification.
    - **Healthcare Access Confounding:** Individuals without insurance or regular doctor visits may have undiagnosed conditions, introducing differential detection bias.

    ### 3. Diabetes Negative Class Heterogeneity
    - In official CDC coding (`DIABETE4`), respondents reporting **prediabetes / borderline diabetes (11,307 respondents)** and gestational diabetes (3,395 respondents) are mapped to `0.0` (non-chronic diabetes).
    - The negative class must not be interpreted as a "healthy" or "normoglycemic" control pool, as laboratory biomarker testing (fasting plasma glucose, HbA1c) is absent from the survey.

    ### 4. Probability Calibration & Class Weighting
    - In rare conditions (Stroke 4.5%, Kidney Disease 5.1%), applying balanced class weighting shifts raw predicted probabilities upward.
    - Consequently, raw probability scores reflect relative ranking scores rather than true population-level absolute incidence rates. Decision threshold tuning was used to align operating points with educational classification objectives.

    ### 5. Unweighted Model Estimates
    - The models are evaluated on unweighted survey splits rather than via survey-weighted pseudo-maximum-likelihood algorithms. The metrics describe sample accuracy, not weighted US population diagnostic performance.
    """)
