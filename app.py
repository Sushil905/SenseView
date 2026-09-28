"""FastAPI dashboard for the local multi-view expression prototype."""
from __future__ import annotations

import os
import sys
import importlib.util
from itertools import cycle
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from mvfer import EXPRESSIONS  # noqa: E402
from mvfer.analytics import session_features  # noqa: E402
from mvfer.fer.vit_fer import QualityAwareMultiViewFER, ViTBaseLSTM  # noqa: E402

app = FastAPI(title="SenseView Research Dashboard", version="0.2.0")
MODELS = {}
DEMO_TIMESTEPS = cycle((0, 2, 4, 6))
CAMERA_NAMES = ("cam_01", "cam_02", "cam_03", "cam_04")
LEGACY_CHECKPOINT = ROOT / "runs" / "checkpoints" / "synthetic_fer.pt"


def load_model(checkpoint_path: Path):
    key = str(checkpoint_path.resolve())
    if key not in MODELS:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        architecture = checkpoint.get("architecture", "legacy_tiny" if checkpoint.get("tiny") else "legacy_quality")
        if architecture == ViTBaseLSTM.architecture:
            network = ViTBaseLSTM(pretrained=False)
        else:
            network = QualityAwareMultiViewFER(pretrained=False, tiny=checkpoint.get("tiny", False))
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        MODELS[key] = (network, architecture, checkpoint)
    return MODELS[key]


def configured_checkpoint() -> Path:
    configured = os.environ.get("MVFER_MODEL_CHECKPOINT")
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else ROOT / candidate
    candidate = ROOT / "runs" / "checkpoints" / "fer.pt"
    return candidate if candidate.exists() else LEGACY_CHECKPOINT


def decode_image(raw: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.width * image.height > 20_000_000:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the local demo limit.")
            image.load()
            return image.convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Upload a valid JPG, PNG, or WebP image.") from error


def predict_images(images, yaw_degrees=None, checkpoint_path: Path | None = None):
    checkpoint_path = checkpoint_path or configured_checkpoint()
    network, architecture, _ = load_model(checkpoint_path)
    image_size = 224 if architecture == ViTBaseLSTM.architecture else 64
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    frames_by_time = [images] if images and isinstance(images[0], Image.Image) else images
    if not frames_by_time or any(len(frame) != len(CAMERA_NAMES) for frame in frames_by_time):
        raise ValueError("Each synchronized timestep must include exactly four camera images")
    views = torch.stack([torch.stack([transform(image.convert("RGB")) for image in frame])
                         for frame in frames_by_time])
    selected_views = []
    if architecture == ViTBaseLSTM.architecture:
        yaw = yaw_degrees if yaw_degrees is not None else [[float("nan")] * len(CAMERA_NAMES) for _ in frames_by_time]
        with torch.inference_mode():
            result = network(views.unsqueeze(0), yaw_degrees=torch.tensor(yaw, dtype=torch.float32).unsqueeze(0))
            per_frame_probabilities = result["logits"].softmax(-1)[0]
            probabilities = per_frame_probabilities.mean(0)
            weights = result["view_weights"][0].mean(0)
        selected_views = [CAMERA_NAMES[index] for index in result["selected_view_indices"][0].tolist()]
    else:
        with torch.inference_mode():
            result = network(views.unsqueeze(0))
            per_frame_probabilities = result["logits"].softmax(-1)[0]
            probabilities = per_frame_probabilities.mean(0)
            weights = result["view_weights"][0].mean(0)
        selected_views = [CAMERA_NAMES[int(weights.argmax())]]
    confidence, prediction = probabilities.max(0)
    frame_confidence, frame_predictions = per_frame_probabilities.max(-1)
    session_summary = session_features(frame_predictions.tolist(), frame_confidence.tolist())
    session_summary["mean_view_quality"] = result["view_weights"].max(-1).values.mean().item()
    session_classifier = None
    svm_env_path = os.environ.get("MVFER_SVM_CHECKPOINT")
    svm_path = Path(svm_env_path) if svm_env_path else ROOT / "runs" / "checkpoints" / "session_svm.joblib"
    if not svm_path.is_absolute():
        svm_path = ROOT / svm_path
    if svm_path.is_file():
        import joblib
        import pandas as pd
        bundle = joblib.load(svm_path)
        feature_row = pd.DataFrame([{name: session_summary[name] for name in bundle["feature_columns"]}])
        svm_prediction = bundle["estimator"].predict(feature_row)[0]
        session_classifier = {"name": "SVM", "target_column": bundle.get("target_column", "target"),
                              "prediction": str(svm_prediction)}
    return {
        "expression": EXPRESSIONS[prediction.item()],
        "confidence": round(confidence.item(), 4),
        "expressions": list(EXPRESSIONS),
        "probabilities": [round(value, 4) for value in probabilities.tolist()],
        "view_quality": [round(value, 4) for value in weights.tolist()],
        "selected_views": selected_views,
        "yaw_degrees": [[round(float(value), 2) for value in row] for row in yaw_degrees] if yaw_degrees is not None else [],
        "fusion_method": "lowest_yaw" if architecture == ViTBaseLSTM.architecture else "learned_quality",
        "session_classifier": session_classifier,
        "notice": "Synthetic-demo output only unless separately trained and validated. It is not diagnostic.",
    }


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/styles.css")
def styles():
    return FileResponse(ROOT / "web" / "styles.css", media_type="text/css")


@app.get("/app.js")
def javascript():
    return FileResponse(ROOT / "web" / "app.js", media_type="text/javascript")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "mvfer-fastapi-dashboard"}


@app.get("/api/status")
def project_status():
    rafdb_images = ROOT / "data" / "raw" / "rafdb" / "aligned"
    rafdb_annotations = ROOT / "data" / "raw" / "rafdb" / "list_patition_label.txt"
    clip_manifest = ROOT / "data" / "processed" / "multiview_manifest.csv"
    temporal_checkpoint = ROOT / "runs" / "checkpoints" / "fer.pt"
    svm_checkpoint = ROOT / "runs" / "checkpoints" / "session_svm.joblib"
    return {
        "rafdb_ready": rafdb_images.is_dir() and rafdb_annotations.is_file(),
        "rafdb_images_present": rafdb_images.is_dir() and any(rafdb_images.iterdir()),
        "rafdb_annotations_present": rafdb_annotations.is_file(),
        "retinaface_available": importlib.util.find_spec("retinaface") is not None,
        "clip_manifest_ready": clip_manifest.is_file(),
        "temporal_checkpoint_ready": temporal_checkpoint.is_file(),
        "svm_checkpoint_ready": svm_checkpoint.is_file(),
    }


@app.get("/api/demo")
def demo():
    try:
        timestep = next(DEMO_TIMESTEPS)
        session = ROOT / "data" / "synthetic" / "crops" / "synthetic_012_00"
        images = [Image.open(session / f"{timestep:04d}_cam_{view:02d}.png").convert("RGB")
                  for view in range(1, 5)]
        response = predict_images(images, checkpoint_path=LEGACY_CHECKPOINT)
        response["demo_timestep"] = timestep
        return response
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Synthetic demo failed: {error}") from error


@app.post("/api/predict")
async def predict(cam_01: list[bytes] = File(...), cam_02: list[bytes] = File(...),
                  cam_03: list[bytes] = File(...), cam_04: list[bytes] = File(...)):
    try:
        camera_sequences = [cam_01, cam_02, cam_03, cam_04]
        lengths = {len(sequence) for sequence in camera_sequences}
        if len(lengths) != 1 or not lengths or 0 in lengths:
            raise HTTPException(status_code=422, detail="Upload the same number of synchronized frames for every camera.")
        images = [[decode_image(camera_sequences[view][timestep]) for view in range(4)]
                  for timestep in range(len(cam_01))]
        checkpoint = configured_checkpoint()
        _, architecture, _ = load_model(checkpoint)
        yaw_values = None
        if architecture == ViTBaseLSTM.architecture:
            from mvfer.detection.face_detector import RetinaFaceDetector
            detector = RetinaFaceDetector()
            detections = [[detector.detect_and_crop(image) for image in frame] for frame in images]
            images = [[item["image"] for item in frame] for frame in detections]
            yaw_values = [[item["yaw_degrees"] for item in frame] for frame in detections]
        return predict_images(images, yaw_degrees=yaw_values, checkpoint_path=checkpoint)
    except HTTPException:
        raise
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Unable to process the views: {error}") from error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
