"""Generate an explicitly artificial line-art multi-view FER dataset for smoke tests.

The images are generated cartoons, not photographs or biometric data. They are
only useful for checking that the data/model pipeline can learn a known signal.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw

from . import EXPRESSIONS

VIEWS = ("cam_01", "cam_02", "cam_03", "cam_04")


def draw_face(expression: int, view: int, seed: int, size: int = 64) -> Image.Image:
    rng = random.Random(seed)
    # A deliberately obvious class cue lets this tiny demo converge quickly;
    # it is a test fixture, not an attempt to mimic real-world expression data.
    palettes = ((238, 242, 246), (222, 245, 226), (226, 235, 250), (251, 229, 225), (239, 226, 248), (247, 239, 212), (255, 235, 202))
    base = palettes[expression]
    image = Image.new("RGB", (size, size), tuple(max(0, min(255, x + rng.randrange(-3, 4))) for x in base))
    draw = ImageDraw.Draw(image)
    x_shift = (-5, -2, 2, 5)[view] + rng.randrange(-2, 3)
    y_shift = rng.randrange(-2, 3)
    skin = (238, 190 + rng.randrange(-10, 11), 155 + rng.randrange(-8, 9))
    draw.ellipse((14 + x_shift, 8 + y_shift, 50 + x_shift, 53 + y_shift), fill=skin, outline=(76, 55, 48), width=2)
    # Eyebrows / eyes vary with the abstract expression label.
    brow_y = 20 + y_shift
    slant = {0: 0, 1: -2, 2: 2, 3: 3, 4: -3, 5: 1, 6: -2}[expression]
    draw.line((21 + x_shift, brow_y, 28 + x_shift, brow_y + slant), fill=(48, 35, 35), width=2)
    draw.line((36 + x_shift, brow_y + slant, 43 + x_shift, brow_y), fill=(48, 35, 35), width=2)
    eye_height = 3 if expression in (4, 6) else 2
    for eye_x in (25, 39):
        draw.ellipse((eye_x + x_shift - 2, 26 + y_shift - eye_height, eye_x + x_shift + 2, 26 + y_shift + eye_height), fill=(42, 38, 38))
    # Mouth: neutral, smile, frown, open/teeth variants.
    box = (24 + x_shift, 32 + y_shift, 40 + x_shift, 45 + y_shift)
    if expression == 1:
        draw.arc(box, 10, 170, fill=(120, 42, 50), width=3)
    elif expression in (2, 3):
        draw.arc(box, 190, 350, fill=(120, 42, 50), width=3)
    elif expression in (4, 6):
        draw.ellipse((28 + x_shift, 34 + y_shift, 36 + x_shift, 43 + y_shift), fill=(92, 36, 42))
    elif expression == 5:
        draw.rectangle((27 + x_shift, 36 + y_shift, 37 + x_shift, 40 + y_shift), fill=(245, 245, 235), outline=(120, 42, 50))
    else:
        draw.line((27 + x_shift, 39 + y_shift, 37 + x_shift, 39 + y_shift), fill=(120, 42, 50), width=2)
    # Camera-specific partial occlusion models the reason for multi-view fusion.
    if (seed + view) % 11 == 0:
        draw.rectangle((8 if view % 2 else 44, 21, 18 if view % 2 else 56, 37), fill=(104, 135, 155))
    return image


def generate(output: Path, subjects: int, sessions_per_subject: int, timesteps: int, seed: int):
    crops = output / "crops"; crops.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in range(subjects):
        split = "train" if subject < int(subjects * 0.75) else "val"
        for session in range(sessions_per_subject):
            session_id = f"synthetic_{subject:03d}_{session:02d}"
            for timestep in range(timesteps):
                expression = (subject + session + timestep // 2) % len(EXPRESSIONS)
                for view_index, view in enumerate(VIEWS):
                    relative = Path("data") / "synthetic" / "crops" / session_id / f"{timestep:04d}_{view}.png"
                    absolute = output.parent.parent / relative
                    absolute.parent.mkdir(parents=True, exist_ok=True)
                    draw_face(expression, view_index, seed + subject * 10000 + session * 1000 + timestep * 10 + view_index).save(absolute)
                    rows.append({"subject_id": f"SYN{subject:03d}", "session_id": session_id, "timestep": timestep, "view_id": view, "image_path": str(relative), "expression": EXPRESSIONS[expression], "split": split})
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for split in ("train", "val"):
        with (output / f"{split}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows([r for r in rows if r["split"] == split])
    print(f"Created {len(rows)} artificial crops and {len(rows) // len(VIEWS)} synchronized timesteps at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--subjects", type=int, default=16); parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=8); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); generate(Path(args.output), args.subjects, args.sessions, args.timesteps, args.seed)
