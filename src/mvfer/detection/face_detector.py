"""Optional RetinaFace face detection and crop adapter.

Install ``retina-face`` to enable this adapter. The yaw value is a geometric
proxy from five-point landmarks, intended only for selecting a relatively
frontal camera view. It is not a calibrated pose estimate.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def yaw_proxy_degrees(landmarks: dict) -> float:
    """Estimate relative yaw from nose displacement and eye spacing."""
    left = np.asarray(landmarks["left_eye"], dtype=float)
    right = np.asarray(landmarks["right_eye"], dtype=float)
    nose = np.asarray(landmarks["nose"], dtype=float)
    eye_midpoint = (left + right) / 2
    eye_distance = float(np.linalg.norm(left - right))
    if eye_distance <= 1e-6:
        raise ValueError("RetinaFace returned coincident eye landmarks")
    normalized_offset = (nose[0] - eye_midpoint[0]) / (eye_distance / 2)
    return math.degrees(math.atan(normalized_offset))


class RetinaFaceDetector:
    """Run RetinaFace and return the highest-confidence face crop plus yaw."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def detect_and_crop(self, image: Image.Image) -> dict:
        try:
            from retinaface import RetinaFace
        except ImportError as error:
            raise RuntimeError("Install requirements-vision.txt to use RetinaFace") from error

        rgb = np.asarray(image.convert("RGB"))
        detections = RetinaFace.detect_faces(rgb[:, :, ::-1].copy(), threshold=self.threshold)
        if not isinstance(detections, dict) or not detections:
            raise ValueError("RetinaFace found no face in this view")
        _, face = max(detections.items(), key=lambda item: float(item[1].get("score", 0)))
        x1, y1, x2, y2 = (int(v) for v in face["facial_area"])
        height, width = rgb.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("RetinaFace returned an invalid bounding box")
        crop = Image.fromarray(rgb[y1:y2, x1:x2])
        yaw = yaw_proxy_degrees(face["landmarks"])
        return {"image": crop, "yaw_degrees": yaw,
                "confidence": float(face.get("score", 0)),
                "bounding_box": [x1, y1, x2, y2]}


def detect_faces(image: Image.Image, threshold: float = 0.9):
    """Compatibility helper returning all RetinaFace detections."""
    try:
        from retinaface import RetinaFace
    except ImportError as error:
        raise RuntimeError("Install requirements-vision.txt to use RetinaFace") from error
    rgb = np.asarray(image.convert("RGB"))
    return RetinaFace.detect_faces(rgb[:, :, ::-1].copy(), threshold=threshold)
