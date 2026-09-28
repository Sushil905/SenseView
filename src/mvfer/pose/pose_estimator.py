"""Lightweight yaw proxy from RetinaFace eye and nose landmarks.

This is not a calibrated 3D head-pose estimator. It is used only to rank
synchronized camera views by relative frontalness.
"""

from ..detection.face_detector import yaw_proxy_degrees


def estimate_pose(landmarks):
    return {"yaw_degrees": yaw_proxy_degrees(landmarks), "pitch_degrees": None,
            "roll_degrees": None, "method": "retinaface_landmark_yaw_proxy"}
