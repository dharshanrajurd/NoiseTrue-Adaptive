"""
Reproduces the submitted checkpoint: weights/final_model.pth

Final model = wide NAFNet-lite (base_ch=48) + degradation-aware FiLM conditioning,
trained with Charbonnier + Sobel edge + VGG16 perceptual loss, flip augmentation,
and a cosine learning-rate schedule.

Usage:
    python train.py --gt_dir <path/to/GT> --noisy_dir <path/to/NoisyLR>

All hyperparameters below match configs/final_model_config.yaml.
Random seed is fixed at 42 so the train/validation split is reproducible.
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_nafnet import NAFNetLiteAdaptive
from src.advanced_loss import VGGPerceptualLoss, combined_loss_v2
from src.dataset_augmented import RestorationDatasetAugmented


def parse_args():
    p = argparse.ArgumentParser(description="Train NoiseTrue-Adaptive final model")
    p.add_argument("--gt_dir", required=True, help="Folder of clean ground-truth .npy images")
    p.add_argument("--noisy_dir", required=True, help="Folder of degraded .npy images")
    p.add_argument("--out", default="weights/final_model.pth", help="Where to save the checkpoint")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base_ch", type=int, default=48, help="Model width")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)

    dataset = RestorationDatasetAugmented(args.gt_dir, args.noisy_dir, augment=True)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    model = NAFNetLiteAdaptive(base_ch=args.base_ch).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    vgg_loss_fn = VGGPerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        dataset.augment = True          # flips on for training
        total = 0.0
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad()
            loss = combined_loss_v2(model(noisy), gt, vgg_loss_fn)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()

        model.eval()
        dataset.augment = False         # flips off for clean, comparable validation
        val = 0.0
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                val += combined_loss_v2(model(noisy), gt, vgg_loss_fn).item()
        val /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} - train_loss: {total/len(train_loader):.5f} "
              f"- val_loss: {val:.5f} - lr: {scheduler.get_last_lr()[0]:.6f}", flush=True)

        if val < best_val:                      # keep the best checkpoint, not the last
            best_val = val
            torch.save(model.state_dict(), args.out)

    print(f"\nBest val_loss: {best_val:.5f}")
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
