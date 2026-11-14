"""
Model Training Module for Multi-Disease Prediction
Author: Antigravity (ML Engineer & Technical Mentor)
Dataset: CDC BRFSS 2024 (LLCP2024.XPT)

Methodological Remediation Highlights:
1. Strict 3-Way Stratified Data Splitting (60% Train / 20% Validation / 20% Holdout Test):
   - Outer Split (80/20): Dedicated, untouched holdout test set (20,000 samples).
   - Inner Split (75/25): 60,000 training samples for CV and fitting; 20,000 validation samples for threshold & calibration analysis.
2. Complete Preprocessing Leakage Prevention:
   - Scikit-learn Pipeline([('preprocessor', create_feature_preprocessor()), ('classifier', clf)])
   - Preprocessors fit strictly inside each CV fold during cross_val_score.
3. Methodologically Valid Model Selection:
   - Winning algorithm selected EXCLUSIVELY by mean cross-validation ROC-AUC on X_train.
   - Zero test data access during candidate comparison and selection.
4. Threshold Optimization & Probability Calibration:
   - Calibration curves and Brier score vs prevalence baseline evaluated on validation data.
   - Threshold sweep on validation data to select optimal screening threshold tau* (targeting Recall >= 70% while maximizing F1).
   - Frozen model and threshold evaluated once on the final untouched test set.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.data_preprocessing import (
    CLEAN_FEATURE_NAMES,
    TARGET_NAMES,
    create_feature_preprocessor
)
from src.model_evaluation import (
    calculate_clinical_metrics,
    calculate_brier_metrics,
    evaluate_threshold_grid,
    select_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve
)


DISEASE_METADATA = {
    'target_diabetes': {
        'display_name': 'Diabetes Mellitus',
        'slug': 'diabetes',
        'file_name': 'diabetes_model.pkl'
    },
    'target_heart_disease': {
        'display_name': 'Heart Disease (CHD / MI)',
        'slug': 'heart_disease',
        'file_name': 'heart_disease_model.pkl'
    },
    'target_stroke': {
        'display_name': 'Stroke',
        'slug': 'stroke',
        'file_name': 'stroke_model.pkl'
    },
    'target_asthma': {
        'display_name': 'Asthma',
        'slug': 'asthma',
        'file_name': 'asthma_model.pkl'
    },
    'target_copd': {
        'display_name': 'COPD / Chronic Bronchitis',
        'slug': 'copd',
        'file_name': 'copd_model.pkl'
    },
    'target_kidney_disease': {
        'display_name': 'Chronic Kidney Disease',
        'slug': 'kidney_disease',
        'file_name': 'kidney_disease_model.pkl'
    },
    'target_arthritis': {
        'display_name': 'Arthritis',
        'slug': 'arthritis',
        'file_name': 'arthritis_model.pkl'
    }
}


def build_candidate_pipelines(pos_ratio: float = 0.1) -> Dict[str, Pipeline]:
    """
    Constructs unified Pipelines pairing a fresh ColumnTransformer preprocessor
    with each candidate classification algorithm.
    All models use balanced class weighting to handle survey class imbalance.
    """
    scale_weight = (1.0 - pos_ratio) / max(pos_ratio, 0.01)

    classifiers = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=30,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        'Support Vector Machine (Linear)': CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight='balanced',
                dual=False,
                max_iter=2000,
                random_state=42
            ),
            cv=3
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_weight,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42
        )
    }

    pipelines = {}
    for name, clf in classifiers.items():
        pipelines[name] = Pipeline([
            ('preprocessor', create_feature_preprocessor()),
            ('classifier', clf)
        ])

    return pipelines


def train_and_evaluate_disease(
    df: pd.DataFrame,
    target_col: str,
    output_model_dir: str = "models",
    output_figure_dir: str = os.path.join("reports", "figures")
) -> Tuple[Dict[str, Any], Pipeline]:
    """
    Executes the methodologically sound training, validation, and evaluation pipeline for a single disease target:
    1. Stratified 3-way split: 60% Train, 20% Validation, 20% Holdout Test.
    2. 3-fold Stratified CV on X_train only with preprocessing inside the Pipeline.
    3. Select winning model strictly by mean CV ROC-AUC on X_train.
    4. Fit winning pipeline on X_train; evaluate calibration and optimize threshold on X_val only.
    5. Freeze pipeline and threshold; evaluate once on holdout X_test.
    6. Save diagnostic figures and serialized model bundle.
    """
    meta = DISEASE_METADATA[target_col]
    disease_name = meta['display_name']
    slug = meta['slug']

    print(f"\n{'='*70}\nMETHODOLOGICAL PIPELINE: {disease_name.upper()} ({target_col})\n{'='*70}")

    # Step 1: Feature and Target Extraction (Strict exclusion of all disease targets from X)
    X = df[CLEAN_FEATURE_NAMES].copy()
    y = df[target_col].astype(int).copy()

    # Step 2: 3-Way Stratified Data Splitting
    # Outer split: 80% dev, 20% final test (HELD OUT & LOCKED)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    # Inner split: 75% train (60% total), 25% val (20% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
    )

    pos_tr = int(y_train.sum())
    pos_val = int(y_val.sum())
    pos_te = int(y_test.sum())
    pos_ratio = pos_tr / len(y_train)

    print(f"Data Splits (random_state=42):")
    print(f"  • Train Set     : {len(X_train):,} samples (Pos: {pos_tr:,} [{pos_ratio*100:.2f}%], Neg: {len(X_train)-pos_tr:,})")
    print(f"  • Validation Set: {len(X_val):,} samples (Pos: {pos_val:,} [{pos_val/len(X_val)*100:.2f}%], Neg: {len(X_val)-pos_val:,})")
    print(f"  • Test Set (LOCKED): {len(X_test):,} samples (Pos: {pos_te:,} [{pos_te/len(X_test)*100:.2f}%], Neg: {len(X_test)-pos_te:,})")

    # Step 3: Candidate Model Cross-Validation on X_train ONLY
    pipelines = build_candidate_pipelines(pos_ratio=pos_ratio)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    cv_results = {}
    print("\n--- Phase 1: Candidate Model Cross-Validation (Training Data Only) ---")
    for name, pipe in pipelines.items():
        t0 = time.time()
        print(f"  CV scoring {name:32s}...", end=" ", flush=True)

        # Preprocessor fits independently inside each fold
        cv_scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        fit_time = time.time() - t0
        mean_auc = float(cv_scores.mean())
        std_auc = float(cv_scores.std())

        cv_results[name] = {
            'cv_auc_mean': mean_auc,
            'cv_auc_std': std_auc,
            'fit_time': fit_time
        }
        print(f"Done in {fit_time:5.1f}s | Mean CV ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    # Step 4: Model Selection EXCLUSIVELY via Training Cross-Validation
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['cv_auc_mean'])
    best_cv_stats = cv_results[best_model_name]
    winning_pipeline = pipelines[best_model_name]

    print(f"\n>>> WINNING MODEL SELECTED BY CV: {best_model_name}")
    print(f"    (Training 3-Fold CV ROC-AUC: {best_cv_stats['cv_auc_mean']:.4f} +/- {best_cv_stats['cv_auc_std']:.4f})")

    # Step 5: Fit Winning Pipeline on Full X_train
    print(f"\n--- Phase 2: Fitting {best_model_name} on X_train ({len(X_train):,} samples) ---")
    t0 = time.time()
    winning_pipeline.fit(X_train, y_train)
    print(f"Fitting completed in {time.time()-t0:.2f}s.")

    # Step 6: Validation Set Calibration & Threshold Optimization (X_val ONLY)
    print("\n--- Phase 3: Calibration & Threshold Analysis (Validation Set Only) ---")
    val_probs = winning_pipeline.predict_proba(X_val)[:, 1]
    brier_metrics = calculate_brier_metrics(y_val, val_probs)

    print(f"  Validation Prevalence : {brier_metrics['sample_prevalence']*100:.2f}%")
    print(f"  Model Brier Score     : {brier_metrics['brier_score']:.4f}")
    print(f"  Prevalence Base Brier : {brier_metrics['brier_baseline']:.4f}")
    print(f"  Brier Skill Score     : {brier_metrics['brier_skill_score']:.4f}")

    # Threshold sweep on validation set
    threshold_df = evaluate_threshold_grid(y_val, val_probs)
    tau_star, val_opt_metrics, rationale = select_optimal_threshold(
        threshold_df,
        target_recall=0.70,
        min_recall=0.50
    )

    print(f"  Optimal Decision Threshold selected on Validation: tau* = {tau_star:.2f}")
    print(f"  Clinical Rationale: {rationale}")
    print(f"  Validation Metrics at tau* = {tau_star:.2f}:")
    print(f"    Recall: {val_opt_metrics['recall']*100:.2f}% | Specificity: {val_opt_metrics['specificity']*100:.2f}% | "
          f"Precision: {val_opt_metrics['precision']*100:.2f}% | F1: {val_opt_metrics['f1_score']:.4f} | "
          f"Pred Pos Rate: {val_opt_metrics['pred_pos_rate']*100:.1f}%")

    # Save validation calibration curve plot
    os.makedirs(output_figure_dir, exist_ok=True)
    calib_plot_path = os.path.join(output_figure_dir, f"{slug}_calibration_curve.png")
    plot_calibration_curve(
        y_true=y_val,
        y_prob=val_probs,
        disease_name=disease_name,
        model_name=best_model_name,
        brier_score=brier_metrics['brier_score'],
        brier_baseline=brier_metrics['brier_baseline'],
        save_path=calib_plot_path
    )

    # Step 7: Final Evaluation on Untouched Holdout Test Set (ONCE)
    print(f"\n--- Phase 4: Final Evaluation on Locked Test Set ({len(X_test):,} samples) ---")
    test_probs = winning_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate at default threshold (tau = 0.50)
    y_test_pred_default = (test_probs >= 0.50).astype(int)
    test_metrics_default = calculate_clinical_metrics(y_test, y_test_pred_default, test_probs)

    # Evaluate at calibrated screening threshold (tau = tau_star)
    y_test_pred_optimal = (test_probs >= tau_star).astype(int)
    test_metrics_optimal = calculate_clinical_metrics(y_test, y_test_pred_optimal, test_probs)

    print(f"  Holdout Test Results @ Default Threshold (tau = 0.50):")
    print(f"    ROC-AUC: {test_metrics_default['roc_auc']:.4f} | PR-AUC: {test_metrics_default['pr_auc']:.4f} | "
          f"Recall: {test_metrics_default['recall']*100:.2f}% | Precision: {test_metrics_default['precision']*100:.2f}% | "
          f"F1: {test_metrics_default['f1_score']:.4f} | Pred Pos Rate: {test_metrics_default['predicted_positive_rate']*100:.1f}%")

    print(f"  Holdout Test Results @ Calibrated Screening Threshold (tau* = {tau_star:.2f}):")
    print(f"    ROC-AUC: {test_metrics_optimal['roc_auc']:.4f} | PR-AUC: {test_metrics_optimal['pr_auc']:.4f} | "
          f"Recall: {test_metrics_optimal['recall']*100:.2f}% | Precision: {test_metrics_optimal['precision']*100:.2f}% | "
          f"F1: {test_metrics_optimal['f1_score']:.4f} | Pred Pos Rate: {test_metrics_optimal['predicted_positive_rate']*100:.1f}%")

    # Step 8: Generate Diagnostic Plots on Holdout Test Set
    # 1. Confusion Matrix @ optimal threshold
    cm_opt_path = os.path.join(output_figure_dir, f"{slug}_confusion_matrix.png")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred_optimal,
        disease_name=disease_name,
        model_name=best_model_name,
        threshold=tau_star,
        save_path=cm_opt_path
    )

    # 2. Confusion Matrix @ default 0.50 threshold
    cm_def_path = os.path.join(output_figure_dir, f"{slug}_confusion_matrix_default.png")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred_default,
        disease_name=disease_name,
        model_name=best_model_name,
        threshold=0.50,
        save_path=cm_def_path
    )

    # 3. ROC Curve
    roc_path = os.path.join(output_figure_dir, f"{slug}_roc_curve.png")
    plot_roc_curve(
        y_true=y_test,
        y_prob=test_probs,
        disease_name=disease_name,
        model_name=best_model_name,
        roc_auc=test_metrics_default['roc_auc'],
        save_path=roc_path
    )

    # 4. PR Curve
    pr_path = os.path.join(output_figure_dir, f"{slug}_pr_curve.png")
    plot_precision_recall_curve(
        y_true=y_test,
        y_prob=test_probs,
        disease_name=disease_name,
        model_name=best_model_name,
        pr_auc=test_metrics_default['pr_auc'],
        save_path=pr_path
    )

    # Step 9: Package and Serialize Standalone Model Bundle
    os.makedirs(output_model_dir, exist_ok=True)
    model_bundle = {
        'pipeline': winning_pipeline,
        'model': winning_pipeline.named_steps['classifier'],
        'preprocessor': winning_pipeline.named_steps['preprocessor'],
        'feature_names': CLEAN_FEATURE_NAMES,
        'target_col': target_col,
        'disease_name': disease_name,
        'model_name': best_model_name,
        'selected_threshold': tau_star,
        'threshold_rationale': rationale,
        'cv_roc_auc_mean': best_cv_stats['cv_auc_mean'],
        'cv_roc_auc_std': best_cv_stats['cv_auc_std'],
        'brier_score_val': brier_metrics['brier_score'],
        'brier_baseline_val': brier_metrics['brier_baseline'],
        'brier_skill_score_val': brier_metrics['brier_skill_score'],
        'validation_metrics': val_opt_metrics,
        'test_metrics_default': test_metrics_default,
        'test_metrics_optimal': test_metrics_optimal,
        'split_sizes': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        'random_state': 42
    }

    bundle_path = os.path.join(output_model_dir, meta['file_name'])
    joblib.dump(model_bundle, bundle_path)
    print(f"\nSaved remediated model bundle to: {bundle_path}")

    # Compile result record
    disease_record = {
        'target_col': target_col,
        'disease_name': disease_name,
        'slug': slug,
        'best_model_name': best_model_name,
        'cv_roc_auc_mean': best_cv_stats['cv_auc_mean'],
        'cv_roc_auc_std': best_cv_stats['cv_auc_std'],
        'selected_threshold': tau_star,
        'threshold_rationale': rationale,
        'brier_score_val': brier_metrics['brier_score'],
        'brier_baseline_val': brier_metrics['brier_baseline'],
        'brier_skill_score_val': brier_metrics['brier_skill_score'],
        'test_metrics_default': test_metrics_default,
        'test_metrics_optimal': test_metrics_optimal,
        'cv_all_models': cv_results,
        'threshold_sweep_val': threshold_df.to_dict(orient='records')
    }

    return disease_record, winning_pipeline


def run_full_training_pipeline(
    sample_size: Optional[int] = 100000,
    output_model_dir: str = "models",
    output_report_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Executes the fully remediated training and evaluation workflow across all 7 chronic conditions.
    Generates updated comparison summaries, calibration reports, and metadata JSON.
    """
    print("="*75)
    print("STARTING REMEDIATED MULTI-DISEASE PREDICTION TRAINING WORKFLOW")
    print("Zero Leakage | CV Model Selection | Validation Threshold Optimization")
    print("="*75)

    clean_data_path = os.path.join("data", "processed", "brfss_cleaned.parquet")
    if not os.path.exists(clean_data_path):
        from src.data_preprocessing import prepare_full_processed_dataset
        df = prepare_full_processed_dataset()
    else:
        df = pd.read_parquet(clean_data_path)

    if sample_size is not None and len(df) > sample_size:
        print(f"Sampling {sample_size:,} records from {len(df):,} for training efficiency...")
        df_active = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_active = df

    print(f"Active pool size: {len(df_active):,} records across 7 targets.\n")

    all_disease_results = {}
    summary_rows = []
    threshold_calibration_rows = []

    for target in TARGET_NAMES:
        res, _ = train_and_evaluate_disease(
            df=df_active,
            target_col=target,
            output_model_dir=output_model_dir
        )
        all_disease_results[target] = res

        m_opt = res['test_metrics_optimal']
        m_def = res['test_metrics_default']
        tau = res['selected_threshold']

        summary_rows.append({
            'Disease': res['disease_name'],
            'Selected Model (by CV)': res['best_model_name'],
            'Train CV ROC-AUC': f"{res['cv_roc_auc_mean']:.4f} +/- {res['cv_roc_auc_std']:.4f}",
            'Holdout ROC-AUC': f"{m_opt['roc_auc']:.4f}",
            'Holdout PR-AUC': f"{m_opt['pr_auc']:.4f}",
            'Opt Threshold': f"{tau:.2f}",
            'Recall @ Opt': f"{m_opt['recall']*100:.2f}%",
            'Specificity @ Opt': f"{m_opt['specificity']*100:.2f}%",
            'Precision @ Opt': f"{m_opt['precision']*100:.2f}%",
            'F1 @ Opt': f"{m_opt['f1_score']:.4f}",
            'Pred Pos Rate @ Opt': f"{m_opt['predicted_positive_rate']*100:.1f}%",
            'Recall @ 0.50': f"{m_def['recall']*100:.2f}%",
            'Precision @ 0.50': f"{m_def['precision']*100:.2f}%",
            'F1 @ 0.50': f"{m_def['f1_score']:.4f}",
            'Pred Pos Rate @ 0.50': f"{m_def['predicted_positive_rate']*100:.1f}%"
        })

        threshold_calibration_rows.append({
            'Disease': res['disease_name'],
            'Selected Model': res['best_model_name'],
            'Val Brier Score': f"{res['brier_score_val']:.4f}",
            'Val Brier Baseline': f"{res['brier_baseline_val']:.4f}",
            'Brier Skill Score': f"{res['brier_skill_score_val']:.4f}",
            'Optimal Threshold (tau*)': f"{tau:.2f}",
            'Threshold Rationale': res['threshold_rationale']
        })

    # Save summary tables
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "="*85)
    print("REMEDIATED MULTI-DISEASE FINAL EVALUATION SUMMARY (HOLDOUT TEST SET)")
    print("="*85)
    print(summary_df.to_string(index=False))

    summary_csv_path = os.path.join(output_report_dir, "model_comparison_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved updated comparison summary table to: {summary_csv_path}")

    calib_csv_path = os.path.join(output_report_dir, "threshold_calibration_summary.csv")
    pd.DataFrame(threshold_calibration_rows).to_csv(calib_csv_path, index=False)
    print(f"Saved threshold and calibration summary to: {calib_csv_path}")

    # Save complete metadata JSON
    metadata_json_path = os.path.join(output_model_dir, "model_metadata.json")
    with open(metadata_json_path, 'w') as f:
        json.dump(all_disease_results, f, indent=4)
    print(f"Saved remediated metadata JSON to: {metadata_json_path}")

    return all_disease_results


if __name__ == "__main__":
    results = run_full_training_pipeline(sample_size=100000)
