from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def extract_positive_scores(estimator, features):
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        return np.asarray(probabilities[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(features), dtype=float)
        return scores
    return None


def compute_metrics(y_true, y_pred, y_score=None, class_names: tuple[str, str] = ("fake", "true")) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=list(class_names), zero_division=0),
        "roc_auc": None,
        "roc_curve": None,
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return metrics


def evaluate_split(estimator, features, labels, class_names: tuple[str, str] = ("fake", "true")) -> dict:
    predictions = estimator.predict(features)
    scores = extract_positive_scores(estimator, features)
    return compute_metrics(labels, predictions, scores, class_names=class_names)
