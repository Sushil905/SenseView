"""Confusion-matrix helper."""

from sklearn.metrics import confusion_matrix


def confusion(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)
