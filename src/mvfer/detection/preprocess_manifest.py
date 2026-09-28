"""RetinaFace-crop a synchronized multiview manifest and add yaw proxies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from .face_detector import RetinaFaceDetector


def preprocess_manifest(manifest: str | Path, data_root: str | Path = ".",
                        output_root: str | Path = "data/processed/faces",
                        output_manifest: str | Path = "data/processed/multiview_manifest.csv"):
    frame = pd.read_csv(manifest)
    required = {"subject_id", "session_id", "timestep", "view_id", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    root, crops = Path(data_root), Path(output_root)
    crops.mkdir(parents=True, exist_ok=True)
    detector = RetinaFaceDetector()
    paths, yaws = [], []
    for row in frame.itertuples(index=False):
        source = root / row.image_path
        with Image.open(source) as image:
            result = detector.detect_and_crop(image)
        relative = Path(str(row.session_id)) / f"{int(row.timestep):06d}_{row.view_id}.jpg"
        target = crops / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        result["image"].save(target, quality=95)
        paths.append(str(target))
        yaws.append(result["yaw_degrees"])
    frame["image_path"] = paths
    frame["yaw_degrees"] = yaws
    destination = Path(output_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-root", default="data/processed/faces")
    parser.add_argument("--output-manifest", default="data/processed/multiview_manifest.csv")
    args = parser.parse_args()
    print(preprocess_manifest(args.manifest, args.data_root, args.output_root, args.output_manifest))


if __name__ == "__main__":
    main()
