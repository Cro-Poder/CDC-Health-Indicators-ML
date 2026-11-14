"""
Multi-Disease Classification System
Dataset: CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.prediction import MultiDiseasePredictor, get_sample_patient_profiles
from src.data_preprocessing import CLEAN_FEATURE_NAMES


# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Disease Classification",
    page_icon="Health",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, clean typography and adaptive neutral palette
st.markdown("""
<style>
    /* Global font & typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Main container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 960px;
    }
    
    /* Header styling */
    .app-title, h1.app-title {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important; /* light blue */
        margin-bottom: 0.35rem !important;
        letter-spacing: -0.02em !important;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: var(--text-color, #94a3b8);
        opacity: 0.85;
        margin-bottom: 1.75rem;
        line-height: 1.5;
    }
    
    /* Section headers in forms */
    .section-title, h2.section-title, h3.section-title {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #38bdf8 !important; /* light blue */
        border-bottom: 2px solid #38bdf8 !important;
        padding-bottom: 0.4rem !important;
        margin-top: 1.75rem !important;
        margin-bottom: 1.15rem !important;
    }
    
    /* Conditions list in overview */
    .condition-item {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        font-weight: 500;
        color: var(--text-color, #334155);
    }
    
    /* Clean result display */
    .result-card {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Footer */
    .app-footer {
        margin-top: 3.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        font-size: 0.85rem;
        color: var(--text-color, #64748b);
        opacity: 0.75;
        text-align: center;
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
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Overview"

st.sidebar.markdown("### Navigation")
selected_page = st.sidebar.radio(
    "Navigation Menu",
    ["Overview", "Assessment"],
    index=0 if st.session_state.page == "Overview" else 1,
    label_visibility="collapsed"
)
st.session_state.page = selected_page


# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
if st.session_state.page == "Overview":
    st.markdown('<div class="app-title" style="color: #38bdf8 !important;">Multi-Disease Classification System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        'An independent data science application analyzing statistical patterns in the '
        'CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 survey.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    This application evaluates self-reported demographic, health, and lifestyle indicators against statistical 
    classification models trained on CDC BRFSS survey data.
    
    Visitors can complete a brief questionnaire to explore how specific combinations of reported characteristics 
    statistically associate with existing diagnoses across seven chronic conditions.
    """)

    st.markdown('<div class="section-title" style="color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important;">Supported Conditions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="condition-item">Diabetes Mellitus</div>', unsafe_allow_html=True)
        st.markdown('<div class="condition-item">Heart Disease (CHD / Myocardial Infarction)</div>', unsafe_allow_html=True)
        st.markdown('<div class="condition-item">Stroke</div>', unsafe_allow_html=True)
        st.markdown('<div class="condition-item">Asthma</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="condition-item">Chronic Obstructive Pulmonary Disease (COPD)</div>', unsafe_allow_html=True)
        st.markdown('<div class="condition-item">Chronic Kidney Disease</div>', unsafe_allow_html=True)
        st.markdown('<div class="condition-item">Arthritis</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Begin Health Assessment", type="primary"):
        st.session_state.page = "Assessment"
        st.rerun()


# ---------------------------------------------------------
# PAGE 2: ASSESSMENT (QUESTIONNAIRE & RESULTS)
# ---------------------------------------------------------
elif st.session_state.page == "Assessment":
    st.markdown('<div class="app-title" style="color: #38bdf8 !important;">Health & Lifestyle Questionnaire</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        'Enter demographic, lifestyle, and health indicators to evaluate statistical classification outcomes.'
        '</div>',
        unsafe_allow_html=True
    )

    # Optional preset loader
    with st.expander("Load sample profile (optional)", expanded=False):
        preset_cols = st.columns([1, 1, 2])
        sample_profiles = get_sample_patient_profiles()

        if preset_cols[0].button("Young adult profile"):
            st.session_state.update(sample_profiles['Healthy Young Adult (Low Risk)'])
            st.rerun()

        if preset_cols[1].button("Senior profile"):
            st.session_state.update(sample_profiles['Senior with Multiple Risk Factors (High Risk)'])
            st.rerun()

    with st.form("assessment_form"):
        # Section A: Demographic Information
        st.markdown('<div class="section-title" style="color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important;">Section A — Demographic Information</div>', unsafe_allow_html=True)
        col_a1, col_a2 = st.columns(2)

        with col_a1:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=80,
                value=int(st.session_state.get('age', 45)),
                step=1
            )
            sex_choice = st.selectbox(
                "Biological sex",
                ["Male", "Female"],
                index=0 if st.session_state.get('sex', 1.0) == 1.0 else 1
            )
            race_options = {
                "White (non-Hispanic)": 1.0,
                "Black or African American (non-Hispanic)": 2.0,
                "Other Race (non-Hispanic)": 3.0,
                "Multiracial (non-Hispanic)": 4.0,
                "Hispanic or Latino": 5.0,
                "Prefer not to say / Unreported": np.nan
            }
            # Determine initial index
            current_race = st.session_state.get('race_ethnicity', 1.0)
            race_default_idx = 0
            if pd.isna(current_race):
                race_default_idx = 5
            elif current_race in [1.0, 2.0, 3.0, 4.0, 5.0]:
                race_default_idx = int(current_race) - 1

            race_choice = st.selectbox(
                "Race / Ethnicity",
                list(race_options.keys()),
                index=race_default_idx
            )

        with col_a2:
            edu_options = {
                "Did not graduate high school": 1.0,
                "Graduated high school or equivalent (GED)": 2.0,
                "Attended college or technical school": 3.0,
                "Graduated from college or technical school": 4.0
            }
            edu_choice = st.selectbox(
                "Education level",
                list(edu_options.keys()),
                index=int(st.session_state.get('education_level', 3.0)) - 1
            )
            inc_options = {
                "Less than $15,000": 1.0,
                "$15,000 to < $25,000": 2.0,
                "$25,000 to < $35,000": 3.0,
                "$35,000 to < $50,000": 4.0,
                "$50,000 to < $100,000": 5.0,
                "$100,000 to < $200,000": 6.0,
                "$200,000 or more": 7.0
            }
            inc_choice = st.selectbox(
                "Annual household income",
                list(inc_options.keys()),
                index=int(st.session_state.get('income_level', 5.0)) - 1
            )

        # Section B: General Health
        st.markdown('<div class="section-title" style="color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important;">Section B — General Health</div>', unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)

        with col_b1:
            gh_options = {
                "Excellent": 1.0,
                "Very good": 2.0,
                "Good": 3.0,
                "Fair": 4.0,
                "Poor": 5.0
            }
            gh_choice = st.selectbox(
                "General health self-rating",
                list(gh_options.keys()),
                index=int(st.session_state.get('general_health', 3.0)) - 1
            )
            diff_walk_choice = st.selectbox(
                "Difficulty walking or climbing stairs",
                ["No", "Yes"],
                index=int(st.session_state.get('diff_walk', 0.0))
            )

        with col_b2:
            phys_days = st.number_input(
                "Days of poor physical health (past 30 days)",
                min_value=0,
                max_value=30,
                value=int(st.session_state.get('physical_health_days', 0)),
                step=1
            )
            ment_days = st.number_input(
                "Days of poor mental health (past 30 days)",
                min_value=0,
                max_value=30,
                value=int(st.session_state.get('mental_health_days', 0)),
                step=1
            )

        # Section C: Lifestyle
        st.markdown('<div class="section-title" style="color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important;">Section C — Lifestyle</div>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            exer_choice = st.selectbox(
                "Physical activity or exercise (past 30 days)",
                ["Yes", "No"],
                index=0 if st.session_state.get('exercise_any', 1.0) == 1.0 else 1
            )
            smk_options = {
                "Never smoked": 4.0,
                "Former smoker": 3.0,
                "Current smoker (some days)": 2.0,
                "Current smoker (every day)": 1.0
            }
            smk_choice = st.selectbox(
                "Smoking status",
                list(smk_options.keys()),
                index=0
            )
            bmi = st.number_input(
                "Body Mass Index (BMI in kg/m²)",
                min_value=12.0,
                max_value=70.0,
                value=float(st.session_state.get('bmi', 26.5)),
                step=0.1
            )

        with col_c2:
            ecig_options = {
                "Never used e-cigarettes": 1.0,
                "Former e-cigarette user": 3.0,
                "Current e-cigarette user": 2.0
            }
            ecig_choice = st.selectbox(
                "E-cigarette usage",
                list(ecig_options.keys()),
                index=0
            )
            heavy_drink_choice = st.selectbox(
                "Heavy alcohol consumption (Men >14/wk, Women >7/wk)",
                ["No", "Yes"],
                index=int(st.session_state.get('heavy_drinker', 0.0))
            )

        # Section D: Healthcare Access
        st.markdown('<div class="section-title" style="color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important;">Section D — Healthcare Access</div>', unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            checkup_options = {
                "Within past year": 1.0,
                "Within past 2 years": 2.0,
                "Within past 5 years": 3.0,
                "5 or more years ago": 4.0,
                "Never": 5.0
            }
            checkup_choice = st.selectbox(
                "Last routine medical checkup",
                list(checkup_options.keys()),
                index=0
            )

        with col_d2:
            ins_choice = st.selectbox(
                "Healthcare insurance coverage",
                ["Yes", "No"],
                index=0 if st.session_state.get('healthcare_coverage', 1.0) == 1.0 else 1
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Assessment", type="primary", use_container_width=True)

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
            st.markdown('<div class="section-title" style="color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important;">Assessment Results</div>', unsafe_allow_html=True)

            table_rows = []
            for target_col, res in results.items():
                table_rows.append({
                    "Condition": res['disease_name'],
                    "Classification": res['prediction_label'],
                    "Operating Threshold": f"{res['selected_threshold']:.2f}",
                    "Model Score": res['probability_pct']
                })

            df_results = pd.DataFrame(table_rows)
            st.table(df_results)

            st.caption(
                "Classification outcomes indicate whether the reported profile aligns with survey respondents "
                "reporting existing chronic conditions under validation-selected decision thresholds. Model scores "
                "represent relative rank-ordering metrics rather than clinical diagnostic probabilities."
            )

        except Exception as e:
            st.error(f"Error executing prediction: {e}")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    '<div class="app-footer">'
    'Educational classification project using self-reported survey data. Not intended for medical use.'
    '</div>',
    unsafe_allow_html=True
)
