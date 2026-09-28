from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .analytics import session_features
from .dataset.loader import MultiViewSequenceDataset
from .fer.vit_fer import QualityAwareMultiViewFER, ViTBaseLSTM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Export descriptive session-level FER features.")
    parser.add_argument("--manifest", required=True); parser.add_argument("--data-root", default=".")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--svm-checkpoint", help="Optional separately trained session-level SVM bundle")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = MultiViewSequenceDataset(args.manifest, args.data_root, checkpoint["views"])
    architecture = checkpoint.get("architecture", "legacy_tiny" if checkpoint.get("tiny") else "legacy_quality")
    if architecture == ViTBaseLSTM.architecture:
        model = ViTBaseLSTM(pretrained=False)
    else:
        model = QualityAwareMultiViewFER(pretrained=False, tiny=checkpoint.get("tiny", False))
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    rows = []
    for item in dataset:
        frames = item["frames"].unsqueeze(0)
        if isinstance(model, ViTBaseLSTM):
            output = model(frames, yaw_degrees=item["yaw_degrees"].unsqueeze(0))
        else:
            output = model(frames)
        probabilities = output["logits"].softmax(-1).squeeze(0)
        confidence, prediction = probabilities.max(-1)
        row = {"subject_id": item["subject_id"], "session_id": item["session_id"],
               **session_features(prediction.tolist(), confidence.tolist())}
        # Quality is a signal for review of technical validity, not a child attribute.
        row["mean_view_quality"] = output["view_weights"].max(-1).values.mean().item()
        if isinstance(model, ViTBaseLSTM):
            row["selected_views"] = ",".join(
                checkpoint["views"][index] for index in output["selected_view_indices"][0].tolist())
        rows.append(row)
    if args.svm_checkpoint:
        import joblib
        bundle = joblib.load(args.svm_checkpoint)
        for row in rows:
            values = pd.DataFrame([{name: row[name] for name in bundle["feature_columns"]}])
            row["svm_prediction"] = str(bundle["estimator"].predict(values)[0])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__": main()
