"""ROC curve calculation for binary research tasks."""

from sklearn.metrics import roc_auc_score, roc_curve


def binary_roc(y_true, scores):
    false_positive, true_positive, thresholds = roc_curve(y_true, scores)
    return {"fpr": false_positive, "tpr": true_positive, "thresholds": thresholds,
            "auc": float(roc_auc_score(y_true, scores))}
