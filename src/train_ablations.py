"""
Reproduces the four ablation checkpoints reported in the deck's comparison table.
The final submitted model is trained separately by train.py in the repo root.

Trains, in order:
    baseline_model.pth          U-Net baseline
    adaptive_model.pth          U-Net + FiLM
    nafnet_baseline_model.pth   NAFNet-lite baseline
    nafnet_adaptive_model.pth   NAFNet-lite + FiLM

All four use the same loss (Charbonnier + Sobel edge, no perceptual term), the
same seed-42 split, and the same optimiser settings, so the only variable across
them is the architecture change under test.

Usage (from the repository root):
    python src/train_ablations.py --gt_dir <path/to/GT> --noisy_dir <path/to/NoisyLR>
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import RestorationUNet, RestorationUNetAdaptive
from src.model_nafnet import NAFNetLite, NAFNetLiteAdaptive
from src.dataset_and_losses import RestorationDataset, combined_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train the four ablation models")
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--noisy_dir", required=True)
    p.add_argument("--weights_dir", default="weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_one(model, name, args, train_loader, val_loader, device):
    print(f"\n=== Training {name} ===")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    out_path = os.path.join(args.weights_dir, f"{name}.pth")

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(noisy), gt)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        val = 0.0
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                val += combined_loss(model(noisy), gt).item()
        val /= len(val_loader)

        print(f"[{name}] Epoch {epoch+1}/{args.epochs} - "
              f"train_loss: {total/len(train_loader):.5f} - val_loss: {val:.5f}", flush=True)

        if val < best:
            best = val
            torch.save(model.state_dict(), out_path)

    print(f"[{name}] Best val_loss: {best:.5f} -> {out_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.weights_dir, exist_ok=True)

    dataset = RestorationDataset(args.gt_dir, args.noisy_dir)
    val_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    train_one(RestorationUNet(),         "baseline_model",        args, train_loader, val_loader, device)
    train_one(RestorationUNetAdaptive(), "adaptive_model",        args, train_loader, val_loader, device)
    train_one(NAFNetLite(),              "nafnet_baseline_model", args, train_loader, val_loader, device)
    train_one(NAFNetLiteAdaptive(),      "nafnet_adaptive_model", args, train_loader, val_loader, device)

    print("\nAll four ablation checkpoints written to", args.weights_dir)


if __name__ == "__main__":
    main()
