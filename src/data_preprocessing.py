"""
Data Preprocessing Module for Multi-Disease Prediction
Dataset: CDC BRFSS 2024 (LLCP2024.XPT)
Author: Antigravity (ML Engineer & Technical Mentor)

This module handles:
1. Loading the raw SAS transport (.XPT) file using pyreadstat / pandas.
2. Filtering relevant demographic, behavioral, and general health features.
3. Translating CDC special missing-value codes (e.g. 88 -> 0 for days, 7/77/99 -> NaN).
4. Creating standardized binary target indicators for verified chronic diseases.
5. Strict prevention of target leakage (excluding all target columns from predictor features).
6. Building scikit-learn preprocessing pipelines for numerical scaling and categorical encoding.
7. Providing reproducible train/test splits.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ---------------------------------------------------------
# CONSTANTS & VARIABLE MAPPINGS
# ---------------------------------------------------------

# CDC Raw Column Names
RAW_FEATURE_COLS = [
    'GENHLTH',    # General health (1-5, 7/9 missing)
    'PHYSHLTH',   # Physical health days (1-30, 88=0, 77/99 missing)
    'MENTHLTH',   # Mental health days (1-30, 88=0, 77/99 missing)
    'DIFFWALK',   # Difficulty walking/climbing stairs (1=Yes, 2=No)
    'EXERANY2',   # Exercise in past 30 days (1=Yes, 2=No)
    '_SMOKER3',   # Four-level smoker status (1-4, 9 missing)
    '_CURECI3',   # E-cigarette usage status (1-3, 9 missing)
    '_RFDRHV9',   # Heavy alcohol drinker (1=No, 2=Yes, 9 missing)
    'CHECKUP1',   # Last routine checkup interval (1-4, 8=Never, 7/9 missing)
    '_BMI5',      # Body Mass Index * 100 (e.g. 2854 = 28.54)
    '_AGE80',     # Imputed age 18-80
    '_SEX',       # Sex (1=Male, 2=Female)
    '_EDUCAG',    # Four-level education completed (1-4, 9 missing)
    '_INCOMG1',   # Seven-level income bracket (1-7, 9 missing)
    '_RACEGR3',   # Five-level race/ethnicity (1-5, 9 missing)
    '_HLTHPL2',   # Health care coverage (1=Yes, 2=No, 9 missing)
]

RAW_DISEASE_COLS = [
    'DIABETE4',   # Diabetes
    '_MICHD',     # Coronary Heart Disease or Myocardial Infarction
    'CVDSTRK3',   # Stroke
    'ASTHMA3',    # Asthma
    'CHCCOPD3',   # COPD / Emphysema / Chronic Bronchitis
    'CHCKDNY2',   # Kidney Disease
    'HAVARTH4',   # Arthritis
]

# Standardized Feature Names for Clean Modeling and Streamlit UI
CLEAN_FEATURE_NAMES = [
    'general_health',
    'physical_health_days',
    'mental_health_days',
    'diff_walk',
    'exercise_any',
    'smoker_status',
    'ecig_user',
    'heavy_drinker',
    'checkup_interval',
    'bmi',
    'age',
    'sex',
    'education_level',
    'income_level',
    'race_ethnicity',
    'healthcare_coverage'
]

TARGET_NAMES = [
    'target_diabetes',
    'target_heart_disease',
    'target_stroke',
    'target_asthma',
    'target_copd',
    'target_kidney_disease',
    'target_arthritis'
]

# Subsets for Preprocessing Pipelines
NUMERICAL_FEATURES = ['bmi', 'age', 'physical_health_days', 'mental_health_days']
ORDINAL_FEATURES = ['general_health', 'education_level', 'income_level', 'checkup_interval']
BINARY_FEATURES = ['diff_walk', 'exercise_any', 'heavy_drinker', 'sex', 'healthcare_coverage']
CATEGORICAL_FEATURES = ['smoker_status', 'ecig_user', 'race_ethnicity']


# ---------------------------------------------------------
# DATA LOADING & CLEANING FUNCTIONS
# ---------------------------------------------------------

def find_raw_dataset_path() -> str:
    """Locates the raw BRFSS SAS XPORT file."""
    candidate_paths = [
        os.path.join("data", "raw", "LLCP2024.XPT"),
        "LLCP2024.XPT",
        os.path.join("..", "data", "raw", "LLCP2024.XPT"),
        os.path.join("..", "LLCP2024.XPT")
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find LLCP2024.XPT in project directories.")


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Reads the required columns from LLCP2024.XPT using pyreadstat or pandas.
    """
    if filepath is None:
        filepath = find_raw_dataset_path()

    needed_cols = RAW_FEATURE_COLS + RAW_DISEASE_COLS

    try:
        import pyreadstat
        df, _ = pyreadstat.read_xport(
            filepath,
            usecols=needed_cols,
            encoding='latin1'
        )
    except Exception as e:
        print(f"pyreadstat error or not installed, falling back to pandas read_sas: {e}")
        df = pd.read_sas(filepath, format='xport', encoding='latin1')
        df = df[needed_cols]

    return df


def clean_brfss_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies official CDC BRFSS 2024 codebook rules to clean input features:
    - Code 88 in PHYSHLTH and MENTHLTH -> 0.0 (None / 0 days)
    - Codes 7, 77, 99 (Don't Know / Refused) -> np.nan
    - _BMI5 divided by 100.0 to obtain actual BMI (kg/m^2)
    - Recodes binary indicators to 0 and 1
    """
    cleaned = pd.DataFrame(index=df.index)

    # 1. General Health: 1=Excellent ... 5=Poor. 7, 9 = Missing
    cleaned['general_health'] = df['GENHLTH'].map(
        lambda x: x if x in [1.0, 2.0, 3.0, 4.0, 5.0] else np.nan
    )

    # 2. Physical Health Days: 1-30 days; 88 = None (0 days); 77, 99 = Missing
    cleaned['physical_health_days'] = df['PHYSHLTH'].map(
        lambda x: 0.0 if x == 88.0 else (x if 1.0 <= x <= 30.0 else np.nan)
    )

    # 3. Mental Health Days: 1-30 days; 88 = None (0 days); 77, 99 = Missing
    cleaned['mental_health_days'] = df['MENTHLTH'].map(
        lambda x: 0.0 if x == 88.0 else (x if 1.0 <= x <= 30.0 else np.nan)
    )

    # 4. Difficulty Walking / Climbing stairs: 1=Yes -> 1, 2=No -> 0; 7, 9 = Missing
    cleaned['diff_walk'] = df['DIFFWALK'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    # 5. Physical Activity / Exercise in past 30 days: 1=Yes -> 1, 2=No -> 0; 7, 9 = Missing
    cleaned['exercise_any'] = df['EXERANY2'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    # 6. Smoker Status: 1=Every day, 2=Some days, 3=Former, 4=Never; 9 = Missing
    cleaned['smoker_status'] = df['_SMOKER3'].map(
        lambda x: x if x in [1.0, 2.0, 3.0, 4.0] else np.nan
    )

    # 7. E-Cigarette Status: 1=Never, 2=Current, 3=Former; 9 = Missing
    cleaned['ecig_user'] = df['_CURECI3'].map(
        lambda x: x if x in [1.0, 2.0, 3.0] else np.nan
    )

    # 8. Heavy Alcohol Drinker: 1=No -> 0, 2=Yes -> 1; 9 = Missing
    cleaned['heavy_drinker'] = df['_RFDRHV9'].map(
        lambda x: 0.0 if x == 1.0 else (1.0 if x == 2.0 else np.nan)
    )

    # 9. Routine Checkup Interval: 1=<1yr, 2=1-2yrs, 3=2-5yrs, 4=5+yrs, 8=Never (map to 5); 7,9 = Missing
    cleaned['checkup_interval'] = df['CHECKUP1'].map(
        lambda x: 5.0 if x == 8.0 else (x if x in [1.0, 2.0, 3.0, 4.0] else np.nan)
    )

    # 10. BMI: divide by 100 to get kg/m^2; filter physiological limits (12 - 70)
    cleaned['bmi'] = df['_BMI5'].map(
        lambda x: (x / 100.0) if (pd.notna(x) and 1200.0 <= x <= 7000.0) else np.nan
    )

    # 11. Age: 18 - 80 continuous years
    cleaned['age'] = df['_AGE80'].map(
        lambda x: x if (pd.notna(x) and 18.0 <= x <= 80.0) else np.nan
    )

    # 12. Sex: 1=Male -> 1, 2=Female -> 0
    cleaned['sex'] = df['_SEX'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    # 13. Education Completed: 1=<HS, 2=HS Grad, 3=Some College, 4=College Grad; 9 = Missing
    cleaned['education_level'] = df['_EDUCAG'].map(
        lambda x: x if x in [1.0, 2.0, 3.0, 4.0] else np.nan
    )

    # 14. Annual Household Income: 1 (<$15k) to 7 ($200k+); 9 = Missing
    cleaned['income_level'] = df['_INCOMG1'].map(
        lambda x: x if x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] else np.nan
    )

    # 15. Race/Ethnicity: 1=White, 2=Black, 3=Other, 4=Multiracial, 5=Hispanic; 9 = Missing
    cleaned['race_ethnicity'] = df['_RACEGR3'].map(
        lambda x: x if x in [1.0, 2.0, 3.0, 4.0, 5.0] else np.nan
    )

    # 16. Healthcare Coverage: 1=Yes -> 1, 2=No -> 0; 9 = Missing
    cleaned['healthcare_coverage'] = df['_HLTHPL2'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    return cleaned


def clean_brfss_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies official CDC BRFSS 2024 coding to build clean binary targets (0 or 1).
    All 'Don't know', 'Refused', and blank responses are mapped to np.nan.

    Target Coding Clarifications:
    - DIABETE4 (Diabetes):
      * Code 1 -> 1.0 (Positive: diagnosed chronic diabetes)
      * Codes 2, 3, 4 -> 0.0 (Negative: includes gestational diabetes [2], never told [3],
        and prediabetes/borderline diabetes [4])
      * Note: The negative class includes respondents reporting prediabetes and gestational-only
        diabetes; it should not be described as a 'healthy' or 'normoglycemic' control pool.
      * Codes 7, 9, Blank -> np.nan (excluded from training and evaluation).
    - _MICHD (Heart Disease): Code 1 -> 1.0, Code 2 -> 0.0, Blank -> np.nan.
    - CVDSTRK3 (Stroke): Code 1 -> 1.0, Code 2 -> 0.0, Codes 7, 9, Blank -> np.nan.
    - ASTHMA3 (Asthma): Code 1 -> 1.0, Code 2 -> 0.0, Codes 7, 9, Blank -> np.nan.
    - CHCCOPD3 (COPD): Code 1 -> 1.0, Code 2 -> 0.0, Codes 7, 9, Blank -> np.nan.
    - CHCKDNY2 (Kidney Disease): Code 1 -> 1.0, Code 2 -> 0.0, Codes 7, 9, Blank -> np.nan.
    - HAVARTH4 (Arthritis): Code 1 -> 1.0, Code 2 -> 0.0, Codes 7, 9, Blank -> np.nan.
    """
    targets = pd.DataFrame(index=df.index)

    # Diabetes: 1 = Yes (1), 2/3/4 = No chronic diabetes (0), 7/9 = NaN
    targets['target_diabetes'] = df['DIABETE4'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x in [2.0, 3.0, 4.0] else np.nan)
    )

    # Heart Disease (MI or Coronary Heart Disease): 1 = Yes (1), 2 = No (0), Blank = NaN
    targets['target_heart_disease'] = df['_MICHD'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    # Stroke: 1 = Yes (1), 2 = No (0), 7/9 = NaN
    targets['target_stroke'] = df['CVDSTRK3'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    # Asthma: 1 = Yes (1), 2 = No (0), 7/9 = NaN
    targets['target_asthma'] = df['ASTHMA3'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    # COPD: 1 = Yes (1), 2 = No (0), 7/9 = NaN
    targets['target_copd'] = df['CHCCOPD3'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    # Kidney Disease: 1 = Yes (1), 2 = No (0), 7/9 = NaN
    targets['target_kidney_disease'] = df['CHCKDNY2'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    # Arthritis: 1 = Yes (1), 2 = No (0), 7/9 = NaN
    targets['target_arthritis'] = df['HAVARTH4'].map(
        lambda x: 1.0 if x == 1.0 else (0.0 if x == 2.0 else np.nan)
    )

    return targets


def prepare_full_processed_dataset(output_dir: str = os.path.join("data", "processed")) -> pd.DataFrame:
    """
    Loads raw data, performs full cleaning on features and targets,
    filters valid complete records, and saves the result to parquet and CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_raw_data()
    print(f"Loaded raw records: {len(raw_df):,}")

    clean_X = clean_brfss_features(raw_df)
    clean_y = clean_brfss_targets(raw_df)

    combined = pd.concat([clean_X, clean_y], axis=1)

    # Keep records where all 7 targets are valid (non-null)
    valid_mask = clean_y.notna().all(axis=1)
    combined_clean = combined[valid_mask].copy()

    # Reset index
    combined_clean.reset_index(drop=True, inplace=True)
    print(f"Clean records with all 7 valid targets: {len(combined_clean):,} ({len(combined_clean)/len(raw_df)*100:.2f}%)")

    parquet_path = os.path.join(output_dir, "brfss_cleaned.parquet")
    combined_clean.to_parquet(parquet_path, index=False)
    print(f"Saved processed dataset to: {parquet_path}")

    return combined_clean


# ---------------------------------------------------------
# SCIKIT-LEARN PREPROCESSING PIPELINE
# ---------------------------------------------------------

def create_feature_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer that cleanly handles:
    - Numerical features: Median Imputation + StandardScaler
    - Categorical features: Most Frequent Imputation + OneHotEncoder (handle_unknown='ignore')
    - Binary & Ordinal features: Most Frequent Imputation
    
    CRITICAL: This pipeline fits strictly on training data to prevent data leakage.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    bin_ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES),
            ('bin_ord', bin_ord_pipeline, BINARY_FEATURES + ORDINAL_FEATURES)
        ],
        remainder='drop'
    )

    return preprocessor


def get_stratified_train_test_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Extracts predictor features X and target series y for a specific disease.
    Guarantees that ALL disease targets are strictly excluded from X.
    Performs stratified train/test split.
    """
    if target_col not in TARGET_NAMES:
        raise ValueError(f"Target '{target_col}' not recognized. Must be one of {TARGET_NAMES}")

    # Exclude ALL disease targets from features to prevent any data leakage
    X = df[CLEAN_FEATURE_NAMES].copy()
    y = df[target_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Testing data preprocessing module...")
    df_clean = prepare_full_processed_dataset()
    print("Processed dataframe preview:")
    print(df_clean.head())
    print("\nTarget prevalences:")
    for t in TARGET_NAMES:
        print(f"  {t:25s}: {df_clean[t].mean()*100:.2f}% positive ({df_clean[t].sum():,})")

