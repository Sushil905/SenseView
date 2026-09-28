"""Train only the expression recognizer; use subject-level splits outside this script."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import EXPRESSIONS
from .dataset.loader import MultiViewSequenceDataset
from .fer.vit_fer import QualityAwareMultiViewFER, ViTBaseLSTM


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        if isinstance(model, ViTBaseLSTM):
            output = model(frames, yaw_degrees=batch["yaw_degrees"].to(device))
        else:
            output = model(frames)
        logits = output["logits"].cpu()
        labels = batch["labels"]
        valid = labels != -100
        correct += (logits.argmax(-1)[valid] == labels[valid]).sum().item()
        total += valid.sum().item()
    model.train()
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True); parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="runs/checkpoints/fer.pt"); parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-manifest", help="Subject-disjoint validation manifest")
    parser.add_argument("--image-size", type=int, default=224); parser.add_argument("--tiny", action="store_true", help="Use legacy small CNN for synthetic smoke tests")
    parser.add_argument("--fer-checkpoint", help="Optional ViT-Base checkpoint fine-tuned on RAF-DB")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet ViT-Base weights")
    parser.add_argument("--progress", action="store_true", help="Show per-batch progress")
    parser.add_argument("--views", nargs="+", default=["cam_01", "cam_02", "cam_03", "cam_04"])
    args = parser.parse_args()
    dataset = MultiViewSequenceDataset(args.manifest, args.data_root, args.views, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  # sessions must have equal T; window longer sessions beforehand
    val_loader = DataLoader(MultiViewSequenceDataset(args.val_manifest, args.data_root, args.views, args.image_size), batch_size=args.batch_size) if args.val_manifest else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.tiny:
        model = QualityAwareMultiViewFER(pretrained=False, tiny=True)
    else:
        model = ViTBaseLSTM(pretrained=not (args.no_pretrained or args.fer_checkpoint))
        if args.fer_checkpoint:
            frame_checkpoint = torch.load(args.fer_checkpoint, map_location="cpu", weights_only=True)
            state = frame_checkpoint.get("state_dict", frame_checkpoint)
            backbone_state = {key.removeprefix("backbone."): value for key, value in state.items()
                              if key.startswith("backbone.")}
            if not backbone_state:
                raise ValueError("RAF-DB checkpoint does not contain ViT backbone weights")
            model.fer.backbone.load_state_dict(backbone_state)
    model = model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    history = []
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc="training", disable=not args.progress):
            frames = batch["frames"].to(device)
            if isinstance(model, ViTBaseLSTM):
                output = model(frames, yaw_degrees=batch["yaw_degrees"].to(device))["logits"]
            else:
                output = model(frames)["logits"]
            loss = loss_fn(output.flatten(0, 1), batch["labels"].to(device).flatten())
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item()
        validation_accuracy = evaluate(model, val_loader, device) if val_loader else None
        record = {"epoch": epoch + 1, "train_loss": total_loss / len(loader), "val_frame_accuracy": validation_accuracy}
        history.append(record); print(record)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "legacy_tiny" if args.tiny else ViTBaseLSTM.architecture,
                "state_dict": model.state_dict(), "views": args.views, "tiny": args.tiny,
                "expressions": list(EXPRESSIONS)}, args.output)
    log_path = Path("runs/logs") / Path(args.output).with_suffix(".metrics.json").name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history, indent=2))

if __name__ == "__main__": main()
