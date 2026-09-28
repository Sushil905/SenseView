from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .. import EXPRESSIONS
from .transforms import make_image_transform

REQUIRED_COLUMNS = {"subject_id", "session_id", "timestep", "view_id", "image_path"}


def read_manifest(path: str | Path, views: Sequence[str]) -> pd.DataFrame:
    """Validate a de-identified synchronized-crop manifest."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if frame.duplicated(["session_id", "timestep", "view_id"]).any():
        raise ValueError("each session/timestep/view_id must be unique")
    observed = set(frame.view_id.unique())
    unknown = observed - set(views)
    if unknown:
        raise ValueError(f"unknown view IDs: {sorted(unknown)}")
    counts = frame.groupby(["session_id", "timestep"]).view_id.nunique()
    if not (counts == len(views)).all():
        raise ValueError("every timestep must include every expected view")
    if "expression" in frame and not frame.expression.dropna().isin(EXPRESSIONS).all():
        raise ValueError(f"expression must be one of {EXPRESSIONS}")
    return frame.sort_values(["session_id", "timestep", "view_id"]).reset_index(drop=True)


class MultiViewSequenceDataset(Dataset):
    """Returns entire synchronized sessions: views [T, V, C, H, W]."""
    def __init__(self, manifest: str | Path, data_root: str | Path, views: Sequence[str], image_size: int = 224):
        self.views = tuple(views)
        self.root = Path(data_root)
        self.frame = read_manifest(manifest, self.views)
        self.sessions = list(self.frame.session_id.unique())
        self.transform = make_image_transform(image_size)

    def __len__(self): return len(self.sessions)

    def __getitem__(self, index):
        session = self.sessions[index]
        group = self.frame[self.frame.session_id == session]
        tensors, labels, yaw_values = [], [], []
        for _, timestep in group.groupby("timestep", sort=True):
            ordered = timestep.set_index("view_id").loc[list(self.views)]
            images = [self.transform(Image.open(self.root / p).convert("RGB")) for p in ordered.image_path]
            tensors.append(torch.stack(images))
            labels.append(EXPRESSIONS.index(ordered.expression.iloc[0]) if "expression" in ordered and pd.notna(ordered.expression.iloc[0]) else -100)
            if "yaw_degrees" in ordered:
                yaw_values.append(torch.tensor(ordered.yaw_degrees.astype(float).tolist(), dtype=torch.float32))
            else:
                yaw_values.append(torch.full((len(self.views),), float("nan")))
        subject_id = str(group.subject_id.iloc[0]) if "subject_id" in group else ""
        return {"frames": torch.stack(tensors), "labels": torch.tensor(labels),
                "yaw_degrees": torch.stack(yaw_values), "session_id": session,
                "subject_id": subject_id}
