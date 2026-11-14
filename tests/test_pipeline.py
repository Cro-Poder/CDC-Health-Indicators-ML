"""
Unit & Integration Test Suite for Multi-Disease Prediction System
Author: Antigravity (ML Engineer & Technical Mentor)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import (
    CLEAN_FEATURE_NAMES,
    TARGET_NAMES,
    create_feature_preprocessor,
    clean_brfss_features,
    clean_brfss_targets
)
from src.model_evaluation import calculate_clinical_metrics
from src.prediction import MultiDiseasePredictor, get_sample_patient_profiles


class TestMultiDiseasePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        cls.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'brfss_cleaned.parquet')
        cls.predictor = MultiDiseasePredictor(models_dir=cls.models_dir)

    def test_01_processed_data_exists(self):
        """Verify processed dataset exists and has valid shape."""
        self.assertTrue(os.path.exists(self.data_path), f"Processed data missing at {self.data_path}")
        df = pd.read_parquet(self.data_path)
        self.assertGreater(len(df), 100000, "Dataset has fewer rows than expected.")
        self.assertEqual(len(df.columns), 23, "Cleaned dataset should have 16 features + 7 targets = 23 columns.")

    def test_02_zero_target_leakage(self):
        """Verify that NO disease outcome is included in predictor features."""
        for target in TARGET_NAMES:
            self.assertNotIn(
                target,
                CLEAN_FEATURE_NAMES,
                f"DATA LEAKAGE DETECTED: Target {target} is inside CLEAN_FEATURE_NAMES!"
            )
            # Check raw variants as well
            self.assertNotIn(target.replace('target_', '').upper(), CLEAN_FEATURE_NAMES)

    def test_03_preprocessor_pipeline(self):
        """Verify that the ColumnTransformer transforms features into valid numeric array."""
        preprocessor = create_feature_preprocessor()
        df = pd.read_parquet(self.data_path).head(100)
        X = df[CLEAN_FEATURE_NAMES]
        X_proc = preprocessor.fit_transform(X)
        self.assertIsInstance(X_proc, np.ndarray)
        self.assertEqual(X_proc.shape[0], 100)
        self.assertFalse(np.isnan(X_proc).any(), "Preprocessed array contains unexpected NaNs.")

    def test_04_all_saved_models_exist(self):
        """Verify that model files exist for all 7 verified diseases."""
        expected_files = [
            'diabetes_model.pkl',
            'heart_disease_model.pkl',
            'stroke_model.pkl',
            'asthma_model.pkl',
            'copd_model.pkl',
            'kidney_disease_model.pkl',
            'arthritis_model.pkl',
            'model_metadata.json'
        ]
        for f in expected_files:
            p = os.path.join(self.models_dir, f)
            self.assertTrue(os.path.exists(p), f"Missing model artifact: {p}")

    def test_05_predictor_loaded_all_diseases(self):
        """Verify that MultiDiseasePredictor loads all 7 models."""
        self.assertEqual(
            len(self.predictor.available_diseases),
            7,
            f"Expected 7 loaded models, found: {self.predictor.available_diseases}"
        )

    def test_06_low_risk_patient_prediction(self):
        """Verify healthy young profile yields predominantly low-risk results."""
        profiles = get_sample_patient_profiles()
        healthy = profiles['Healthy Young Adult (Low Risk)']
        results = self.predictor.predict_all_diseases(healthy)
        self.assertEqual(len(results), 7)
        for target_col, res in results.items():
            self.assertIn(res['risk_tier'], ['Low', 'Moderate'], f"Unexpected high risk for healthy profile on {target_col}")

    def test_07_high_risk_patient_prediction(self):
        """Verify high risk profile yields high risk on major chronic diseases."""
        profiles = get_sample_patient_profiles()
        senior = profiles['Senior with Multiple Risk Factors (High Risk)']
        results = self.predictor.predict_all_diseases(senior)
        self.assertEqual(len(results), 7)
        # Should flag positive on at least 5 conditions
        pos_count = sum(1 for res in results.values() if res['prediction'] == 1)
        self.assertGreaterEqual(pos_count, 5, f"High risk senior should have >= 5 positives, got {pos_count}")

    def test_08_clinical_metrics_calculation(self):
        """Verify accuracy, sensitivity, and confusion matrix calculation."""
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.8, 0.6, 0.1, 0.2, 0.3, 0.1])
        m = calculate_clinical_metrics(y_true, y_pred, y_prob)
        self.assertEqual(m['true_positive'], 2)
        self.assertEqual(m['false_negative'], 1)
        self.assertEqual(m['false_positive'], 1)
        self.assertEqual(m['true_negative'], 6)
        self.assertAlmostEqual(m['recall'], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(m['precision'], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(m['accuracy'], 8.0 / 10.0, places=4)

    def test_09_bundle_structure_and_thresholds(self):
        """Verify that model bundles contain remediated pipeline and threshold metadata."""
        for target_col in TARGET_NAMES:
            bundle = self.predictor.models.get(target_col)
            self.assertIsNotNone(bundle, f"Bundle for {target_col} missing")
            self.assertIn('pipeline', bundle, f"Bundle for {target_col} missing Pipeline")
            self.assertIn('selected_threshold', bundle, f"Bundle for {target_col} missing selected_threshold")
            self.assertIn('cv_roc_auc_mean', bundle, f"Bundle for {target_col} missing cv_roc_auc_mean")
            self.assertIn('split_sizes', bundle, f"Bundle for {target_col} missing split_sizes")
            self.assertEqual(bundle['split_sizes']['train'], 60000)
            self.assertEqual(bundle['split_sizes']['val'], 20000)
            self.assertEqual(bundle['split_sizes']['test'], 20000)

    def test_10_calibrated_threshold_prediction(self):
        """Verify that predictor returns both calibrated and default threshold results."""
        profiles = get_sample_patient_profiles()
        sample = profiles['Healthy Young Adult (Low Risk)']
        res = self.predictor.predict_disease(sample, 'target_diabetes')
        self.assertIn('selected_threshold', res)
        self.assertIn('prediction_default', res)
        self.assertIn('calibration_notice', res)
        self.assertIsInstance(res['probability'], float)

    def test_11_categorical_unseen_handling(self):
        """Verify preprocessor ignores unseen categorical values without crashing."""
        profiles = get_sample_patient_profiles()
        unseen_patient = profiles['Healthy Young Adult (Low Risk)'].copy()
        # Introduce an out-of-range category
        unseen_patient['smoker_status'] = 99.0
        res = self.predictor.predict_disease(unseen_patient, 'target_heart_disease')
        self.assertIn(res['risk_tier'], ['Low', 'Moderate', 'High'])


if __name__ == "__main__":
    unittest.main()

