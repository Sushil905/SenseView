"""MLP estimator factory for exploratory feature analysis."""

from sklearn.neural_network import MLPClassifier


def make_estimator(**kwargs):
    return MLPClassifier(random_state=0, max_iter=500, **kwargs)
