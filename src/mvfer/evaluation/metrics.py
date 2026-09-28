"""Standard classification metrics for research evaluation."""

from sklearn.metrics import balanced_accuracy_score, f1_score


def classification_metrics(y_true, y_pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
