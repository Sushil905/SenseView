"""Random-forest estimator factory for exploratory feature analysis."""

from sklearn.ensemble import RandomForestClassifier


def make_estimator(**kwargs):
    return RandomForestClassifier(class_weight="balanced", random_state=0, **kwargs)
