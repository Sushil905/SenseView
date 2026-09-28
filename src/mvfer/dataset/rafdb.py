"""Dataset adapter for RAF-DB basic-expression aligned images.

Point ``image_root`` at RAF-DB's aligned image folder and ``annotations`` at
its list_partition_label.txt file. RAF-DB is a single-image dataset: samples
from it are not turned into invented multi-camera or temporal sequences.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .. import EXPRESSIONS
from .transforms import make_image_transform

# RAF-DB basic labels: 1 surprise, 2 fear, 3 disgust, 4 happiness,
# 5 sadness, 6 anger, 7 neutral.
RAFDB_TO_EXPRESSION = {
    1: "surprise", 2: "fear", 3: "disgust", 4: "happiness",
    5: "sadness", 6: "anger", 7: "neutral",
}


def read_rafdb_annotations(annotations: str | Path) -> pd.DataFrame:
    """Read the whitespace-separated RAF-DB partition list into a table."""
    rows = []
    for line_number, line in enumerate(Path(annotations).read_text().splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"expected filename and integer label on line {line_number}")
        filename, label_text = fields
        try:
            label = int(label_text)
        except ValueError as error:
            raise ValueError(f"invalid RAF-DB label on line {line_number}: {label_text}") from error
        if label not in RAFDB_TO_EXPRESSION:
            raise ValueError(f"RAF-DB basic-expression label must be 1..7, got {label}")
        partition = filename.split("_", 1)[0].lower()
        rows.append({"filename": filename, "raf_label": label,
                     "expression": RAFDB_TO_EXPRESSION[label],
                     "label": EXPRESSIONS.index(RAFDB_TO_EXPRESSION[label]),
                     "partition": partition})
    if not rows:
        raise ValueError(f"no RAF-DB annotations found in {annotations}")
    return pd.DataFrame(rows)


class RAFDBDataset(Dataset):
    """Single-frame FER dataset backed by RAF-DB aligned image files."""

    def __init__(self, image_root: str | Path, annotations: str | Path,
                 split: str | None = None, image_size: int = 224):
        self.root = Path(image_root)
        frame = read_rafdb_annotations(annotations)
        if split is not None:
            split = split.lower()
            frame = frame[frame.partition == split].reset_index(drop=True)
            if frame.empty:
                available = sorted(read_rafdb_annotations(annotations).partition.unique())
                raise ValueError(f"split {split!r} is empty; available partitions: {available}")
        self.frame = frame
        self.transform = make_image_transform(image_size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.root / row.filename
        if not path.is_file():
            raise FileNotFoundError(f"RAF-DB image not found: {path}")
        with Image.open(path) as source:
            image = self.transform(source.convert("RGB"))
        return {"image": image, "label": int(row.label), "filename": row.filename}
