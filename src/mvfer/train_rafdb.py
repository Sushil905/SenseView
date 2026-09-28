"""Fine-tune a ViT-Base/16 expression model on a licensed RAF-DB copy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset.rafdb import RAFDBDataset
from .fer.vit_fer import ViTBaseFER


@torch.inference_mode()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        logits = model(batch["image"].to(device))
        correct += (logits.argmax(-1).cpu() == batch["label"]).sum().item()
        total += batch["label"].numel()
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, help="RAF-DB aligned image directory")
    parser.add_argument("--annotations", required=True, help="list_partition_label.txt")
    parser.add_argument("--output", default="runs/checkpoints/rafdb_vit_base.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-imagenet-weights", action="store_true")
    args = parser.parse_args()

    train_data = RAFDBDataset(args.image_root, args.annotations, "train", args.image_size)
    test_data = RAFDBDataset(args.image_root, args.annotations, "test", args.image_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTBaseFER(pretrained=not args.no_imagenet_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            logits = model(batch["image"].to(device))
            loss = loss_fn(logits, batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print({"epoch": epoch + 1, "train_loss": total_loss / max(1, len(train_loader)),
               "status": "trained"})

    print({"rafdb_test_accuracy": accuracy(model, test_loader, device)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "vit_base_fer", "state_dict": model.state_dict(),
                "expressions": ["neutral", "happiness", "sadness", "anger", "fear", "disgust", "surprise"],
                "image_size": args.image_size}, output)


if __name__ == "__main__":
    main()
