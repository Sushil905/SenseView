"""Train a generic session-feature SVM with subject-disjoint holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .classifiers.svm import make_estimator
from .evaluation.metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Session feature CSV from mvfer.infer")
    parser.add_argument("--labels", required=True, help="Approved session target CSV with subject_id and session_id")
    parser.add_argument("--target-column", required=True, help="Name of the research target column")
    parser.add_argument("--output", default="runs/checkpoints/session_svm.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    required = {"subject_id", "session_id"}
    if required - set(features.columns) or required - set(labels.columns) or args.target_column not in labels:
        raise ValueError("features and labels must include subject_id/session_id, and labels need the target column")
    merged = features.merge(labels[["subject_id", "session_id", args.target_column]],
                            on=["subject_id", "session_id"], validate="one_to_one")
    numeric = merged.select_dtypes(include="number").columns.tolist()
    feature_columns = [name for name in numeric if name not in {args.target_column}]
    if len(merged) < 2 or merged.subject_id.nunique() < 2:
        raise ValueError("at least two subjects with session features and targets are required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_index, test_index = next(splitter.split(merged[feature_columns], merged[args.target_column], groups=merged.subject_id))
    estimator = make_estimator()
    estimator.fit(merged.iloc[train_index][feature_columns], merged.iloc[train_index][args.target_column])
    truth = merged.iloc[test_index][args.target_column]
    predictions = estimator.predict(merged.iloc[test_index][feature_columns])
    metrics = classification_metrics(truth, predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": estimator, "feature_columns": feature_columns,
                 "target_column": args.target_column}, output)
    log_path = Path("runs/logs/svm_metrics.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({**metrics, "n_train_sessions": len(train_index),
                                    "n_test_sessions": len(test_index),
                                    "n_test_subjects": int(merged.iloc[test_index].subject_id.nunique())}, indent=2) + "\n")
    print(metrics)


if __name__ == "__main__":
    main()
