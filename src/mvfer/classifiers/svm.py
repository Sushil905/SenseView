"""SVM helpers for generic, exploratory session-feature classification.

This module intentionally assigns no autism or clinical meaning to a target.
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_estimator(**kwargs):
    return make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", **kwargs))


def fit_svm(features, labels, **kwargs):
    """Fit a scaled SVM to caller-supplied session features and target labels."""
    estimator = make_estimator(**kwargs)
    return estimator.fit(features, labels)
