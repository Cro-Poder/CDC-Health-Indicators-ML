"""
Model Evaluation Module for Multi-Disease Prediction
Author: Antigravity (ML Engineer & Technical Mentor)

This module provides:
1. Standardized calculation of clinical ML evaluation metrics:
   - Accuracy, Balanced Accuracy
   - Precision (Positive Predictive Value)
   - Recall / Sensitivity (True Positive Rate)
   - Specificity (True Negative Rate)
   - F1-Score (Harmonic Mean of Precision & Recall)
   - ROC-AUC (Area Under the Receiver Operating Characteristic)
   - PR-AUC (Average Precision Score)
   - Confusion Matrix counts: TP, TN, FP, FN
2. Brier Score & Probability Calibration metrics:
   - Brier Score vs Prevalence Baseline
   - Brier Skill Score
3. Threshold sweep analysis on validation data:
   - Trade-off between Threshold, Recall, Specificity, Precision, F1, and Predicted Positive Rate
   - Documented threshold selection for clinical screening
4. Visualization functions for:
   - Confusion Matrices
   - ROC Curves
   - Precision-Recall (PR) Curves
   - Reliability / Calibration Curves
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for binary disease classification.
    
    Why Accuracy alone is dangerous in healthcare:
    If a condition has 5% prevalence (e.g. Stroke), a naive model predicting 'No'
    for every patient achieves 95% accuracy while missing 100% of sick patients!
    Hence, Recall, Precision, and ROC-AUC are paramount.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = np.nan
    pr_auc = np.nan
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            pass

    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
        'pr_auc': float(pr_auc) if not np.isnan(pr_auc) else None,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'predicted_positive_rate': float((tp + fp) / len(y_true)),
        'prevalence': float(np.mean(y_true))
    }


def calculate_brier_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculates the Brier score, baseline prevalence Brier score, and Brier Skill Score.
    Brier Score: Mean squared error of probability predictions against actual outcomes.
    Baseline Brier: Score achieved by predicting constant population prevalence p.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    brier = float(brier_score_loss(y_true, y_prob))
    p = float(np.mean(y_true))
    brier_baseline = float(p * (1.0 - p))
    brier_skill = float(1.0 - (brier / brier_baseline)) if brier_baseline > 0 else 0.0

    return {
        'brier_score': brier,
        'brier_baseline': brier_baseline,
        'brier_skill_score': brier_skill,
        'sample_prevalence': p
    }


def evaluate_threshold_grid(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Evaluates classification performance across a grid of decision thresholds
    strictly on validation data.
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    y_val = np.asarray(y_val).astype(int)
    val_probs = np.asarray(val_probs).astype(float)

    records = []
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rec = recall_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_val, preds, zero_division=0)
        acc = accuracy_score(y_val, preds)
        bal_acc = balanced_accuracy_score(y_val, preds)
        pred_pos_rate = float(np.mean(preds))

        records.append({
            'threshold': float(t),
            'pred_pos_rate': pred_pos_rate,
            'recall': float(rec),
            'specificity': float(spec),
            'precision': float(prec),
            'f1_score': float(f1),
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        })

    return pd.DataFrame(records)


def select_optimal_threshold(
    threshold_df: pd.DataFrame,
    target_recall: float = 0.70,
    min_recall: float = 0.50
) -> Tuple[float, Dict[str, Any], str]:
    """
    Selects the optimal decision threshold based on a documented clinical screening objective:
    
    1. Primary Goal: High sensitivity (Recall >= target_recall, e.g. 70%) to avoid missing sick patients.
    2. Optimization: Among candidate thresholds meeting the sensitivity criterion, select
       the threshold that MAXIMIZES the F1-Score (reducing false positives and screening burden).
    3. Fallback: If no threshold achieves target_recall (e.g. Asthma), select the threshold
       that maximizes Youden's J Index (Recall + Specificity - 1) subject to Recall >= min_recall.
    """
    viable = threshold_df[threshold_df['recall'] >= target_recall]

    if not viable.empty:
        # Maximize F1-score to curb false positive rate while keeping high recall
        best_row = viable.sort_values(by=['f1_score', 'precision'], ascending=False).iloc[0]
        t_opt = float(best_row['threshold'])
        rationale = (
            f"Screening Objective: High Sensitivity (Recall >= {target_recall*100:.0f}%). "
            f"Threshold {t_opt:.2f} maximizes F1-Score ({best_row['f1_score']:.4f}) and Specificity "
            f"({best_row['specificity']*100:.1f}%) while maintaining {best_row['recall']*100:.1f}% Recall."
        )
    else:
        # Fallback: balance sensitivity and specificity via Youden's Index
        candidates = threshold_df[threshold_df['recall'] >= min_recall]
        if candidates.empty:
            candidates = threshold_df
        candidates = candidates.copy()
        candidates['youden_j'] = candidates['recall'] + candidates['specificity'] - 1.0
        best_row = candidates.sort_values(by=['youden_j', 'f1_score'], ascending=False).iloc[0]
        t_opt = float(best_row['threshold'])
        rationale = (
            f"Balanced Objective (Target Recall {target_recall*100:.0f}% unavailable): "
            f"Threshold {t_opt:.2f} maximizes Youden's J statistic ({best_row['youden_j']:.4f}) "
            f"with {best_row['recall']*100:.1f}% Recall and {best_row['specificity']*100:.1f}% Specificity."
        )

    return t_opt, best_row.to_dict(), rationale


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    disease_name: str,
    model_name: str,
    threshold: float = 0.50,
    save_path: Optional[str] = None
) -> None:
    """Plots and optionally saves a readable Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=',d',
        cmap='Blues',
        xticklabels=['Predicted: No', 'Predicted: Yes'],
        yticklabels=['Actual: No', 'Actual: Yes'],
        cbar=False,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title(f'Confusion Matrix: {disease_name}\n({model_name} | Threshold = {threshold:.2f})', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Actual Survey Ground Truth', fontsize=11)
    plt.xlabel('Model Prediction', fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_name: str,
    model_name: str,
    roc_auc: float,
    save_path: Optional[str] = None
) -> None:
    """Plots and optionally saves a Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Chance Line (AUC = 0.50)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title(f'ROC Curve: {disease_name}', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_name: str,
    model_name: str,
    pr_auc: float,
    save_path: Optional[str] = None
) -> None:
    """Plots and saves a Precision-Recall Curve with baseline prevalence reference."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    baseline_prev = float(np.mean(y_true))

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color='#2ca02c', lw=2.5, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    plt.axhline(y=baseline_prev, color='gray', linestyle='--', label=f'Prevalence Baseline ({baseline_prev*100:.1f}%)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=11)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=11)
    plt.title(f'Precision-Recall Curve: {disease_name}', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_name: str,
    model_name: str,
    brier_score: float,
    brier_baseline: float,
    save_path: Optional[str] = None
) -> None:
    """Plots and saves a Probability Calibration (Reliability) Curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', lw=2, color='#d62728', label=f'{model_name} (Brier = {brier_score:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.axhline(y=brier_baseline, color='#17becf', linestyle=':', label=f'Prevalence Base Brier ({brier_baseline:.4f})')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Mean Predicted Probability', fontsize=11)
    plt.ylabel('Observed Fraction of Positives', fontsize=11)
    plt.title(f'Calibration Curve (Reliability): {disease_name}', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='upper left', frameon=True, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
