"""
Prediction and Inference Service for Multi-Disease Prediction
Author: Antigravity (ML Engineer & Technical Mentor)
Dataset: CDC BRFSS 2024 (LLCP2024.XPT)

This module handles:
1. Loading remediated model bundles from the models/ directory.
2. Validating incoming patient feature profiles against the 16 verified risk factors.
3. Generating standalone and multi-disease risk predictions using calibrated decision thresholds.
4. Categorizing probability scores into relative risk tiers:
   - Low Relative Risk (< 30%)
   - Moderate Relative Risk (30% - 60%)
   - High Relative Risk (>= 60%)
5. Providing clear methodological disclosures on class-weighted probability calibration.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.data_preprocessing import CLEAN_FEATURE_NAMES
from src.model_training import DISEASE_METADATA


class MultiDiseasePredictor:
    """
    Inference engine that loads trained pipelines for all 7 verified chronic diseases
    and generates independent probability scores and risk categorizations using
    predefined clinical screening thresholds.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_all_models()

    def _load_all_models(self) -> None:
        """Loads all saved model bundles from disk."""
        meta_path = os.path.join(self.models_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)

        for target_col, meta in DISEASE_METADATA.items():
            model_file = os.path.join(self.models_dir, meta['file_name'])
            if os.path.exists(model_file):
                try:
                    bundle = joblib.load(model_file)
                    self.models[target_col] = bundle
                except Exception as e:
                    print(f"Warning: Failed to load model for {target_col} from {model_file}: {e}")
            else:
                print(f"Notice: Model file {model_file} not found. Train models first.")

    @property
    def available_diseases(self) -> List[str]:
        return list(self.models.keys())

    def validate_input(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates that all required 16 features are provided and formatted properly.
        Returns a single-row DataFrame.
        """
        missing = [f for f in CLEAN_FEATURE_NAMES if f not in patient_data]
        if missing:
            raise ValueError(f"Missing required patient features: {missing}")

        row = {f: [patient_data[f]] for f in CLEAN_FEATURE_NAMES}
        df = pd.DataFrame(row)
        return df

    def predict_disease(self, patient_data: Dict[str, Any], target_col: str) -> Dict[str, Any]:
        """
        Predicts the risk outcome for a single specified disease using the
        validated pipeline and calibrated decision threshold.
        """
        if target_col not in self.models:
            raise KeyError(f"No trained model loaded for disease: {target_col}")

        df_input = self.validate_input(patient_data)
        bundle = self.models[target_col]

        disease_name = bundle.get('disease_name', target_col)
        model_name = bundle.get('model_name', 'Trained Classifier')
        threshold = float(bundle.get('selected_threshold', 0.50))
        rationale = bundle.get('threshold_rationale', 'Default screening threshold')

        # Run inference via Pipeline or preprocessor+model fallback
        if 'pipeline' in bundle:
            pipeline = bundle['pipeline']
            prob_arr = pipeline.predict_proba(df_input)[0]
            probability = float(prob_arr[1])
        else:
            model = bundle['model']
            preprocessor = bundle['preprocessor']
            X_proc = preprocessor.transform(df_input)
            prob_arr = model.predict_proba(X_proc)[0]
            probability = float(prob_arr[1])

        # Prediction under calibrated screening threshold tau*
        prediction_optimal = 1 if probability >= threshold else 0
        prediction_default = 1 if probability >= 0.50 else 0

        # Relative risk tiering based on class-weighted probability
        if probability < 0.30:
            risk_tier = "Low"
            status_color = "green"
        elif probability < 0.60:
            risk_tier = "Moderate"
            status_color = "orange"
        else:
            risk_tier = "High"
            status_color = "red"

        result = {
            'target_col': target_col,
            'disease_name': disease_name,
            'model_name': model_name,
            'selected_threshold': threshold,
            'threshold_rationale': rationale,
            'probability': probability,
            'probability_pct': f"{probability*100:.1f}%",
            'prediction': prediction_optimal,
            'prediction_label': 'Positive' if prediction_optimal == 1 else 'Negative',
            'prediction_default': prediction_default,
            'prediction_default_label': 'Positive' if prediction_default == 1 else 'Negative',
            'risk_tier': risk_tier,
            'status_color': status_color,
            'calibration_notice': (
                "Probability reflects relative rank-ordering under class-weighted optimization, "
                "not calibrated absolute clinical incidence."
            )
        }
        return result

    def predict_all_diseases(self, patient_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Generates independent predictions for all 7 verified chronic diseases.
        """
        results = {}
        for target_col in self.available_diseases:
            results[target_col] = self.predict_disease(patient_data, target_col)
        return results


def get_sample_patient_profiles() -> Dict[str, Dict[str, Any]]:
    """Returns realistic mock profiles for demonstration and testing."""
    healthy_young = {
        'general_health': 1.0,        # Excellent
        'physical_health_days': 0.0,  # 0 days
        'mental_health_days': 1.0,    # 1 day
        'diff_walk': 0.0,             # No difficulty
        'exercise_any': 1.0,          # Active
        'smoker_status': 4.0,         # Never smoked
        'ecig_user': 1.0,             # Never used
        'heavy_drinker': 0.0,         # No
        'checkup_interval': 1.0,      # Past year
        'bmi': 22.4,                  # Normal BMI
        'age': 28.0,                  # 28 years old
        'sex': 1.0,                   # Male
        'education_level': 4.0,       # College graduate
        'income_level': 6.0,          # $100k-$200k
        'race_ethnicity': 1.0,        # White
        'healthcare_coverage': 1.0    # Insured
    }

    high_risk_senior = {
        'general_health': 4.0,        # Fair
        'physical_health_days': 15.0, # 15 days poor physical health
        'mental_health_days': 5.0,    # 5 days
        'diff_walk': 1.0,             # Has difficulty walking
        'exercise_any': 0.0,          # Inactive
        'smoker_status': 1.0,         # Current everyday smoker
        'ecig_user': 3.0,             # Former ecig
        'heavy_drinker': 1.0,         # Heavy drinker
        'checkup_interval': 1.0,      # Past year
        'bmi': 34.2,                  # Obese (Class I)
        'age': 68.0,                  # 68 years old
        'sex': 1.0,                   # Male
        'education_level': 2.0,       # High school graduate
        'income_level': 2.0,          # $15k-$25k
        'race_ethnicity': 1.0,        # White
        'healthcare_coverage': 1.0    # Insured
    }

    return {
        'Healthy Young Adult (Low Risk)': healthy_young,
        'Senior with Multiple Risk Factors (High Risk)': high_risk_senior
    }


if __name__ == "__main__":
    print("Testing MultiDiseasePredictor...")
    try:
        predictor = MultiDiseasePredictor()
        print(f"Loaded {len(predictor.available_diseases)} disease models: {predictor.available_diseases}")
        samples = get_sample_patient_profiles()
        for name, profile in samples.items():
            print(f"\n--- Testing Profile: {name} ---")
            res = predictor.predict_all_diseases(profile)
            for target_col, d in res.items():
                print(f"  {d['disease_name']:28s}: {d['prediction_label']} ({d['probability_pct']} | tau={d['selected_threshold']:.2f}) -> Risk: {d['risk_tier']}")
    except Exception as e:
        print(f"Prediction test notice: {e}")
