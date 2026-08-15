"""
Reproduces the ablation table reported in the solution deck and README.

Evaluates every checkpoint present in weights/ on the held-out validation split
(the same split train.py uses, fixed by seed 42) and reports PSNR, SSIM, LPIPS,
per-image model time and parameter count for each.

Checkpoints that are not present are skipped rather than causing a failure, so
this runs with just the final model if that is all you have.

Usage:
    python evaluate.py --gt_dir <path/to/GT> --noisy_dir <path/to/NoisyLR>
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import lpips

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.model import RestorationUNet, RestorationUNetAdaptive
from src.model_nafnet import NAFNetLite, NAFNetLiteAdaptive
from src.dataset_and_losses import RestorationDataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate NoiseTrue-Adaptive checkpoints")
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--noisy_dir", required=True)
    p.add_argument("--weights_dir", default=os.path.join(HERE, "weights"),
                   help="Folder containing final_model.pth")
    p.add_argument("--ablation_dir", default=os.path.join(HERE, "weights", "baseline_models"),
                   help="Folder containing the four ablation checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(model, name, val_loader, lpips_fn, device):
    model.eval()
    psnrs, ssims, lps, times = [], [], [], []

    with torch.no_grad():
        for noisy, gt in val_loader:
            noisy, gt = noisy.to(device), gt.to(device)

            t0 = time.time()
            pred = model(noisy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)

            pred_np = pred.clamp(0, 1).cpu().numpy()[0, 0]
            gt_np = gt.cpu().numpy()[0, 0]

            psnrs.append(psnr_fn(gt_np, pred_np, data_range=1.0))
            ssims.append(ssim_fn(gt_np, pred_np, data_range=1.0))

            p3 = (pred.clamp(0, 1) * 2 - 1).repeat(1, 3, 1, 1)
            g3 = (gt * 2 - 1).repeat(1, 3, 1, 1)
            lps.append(lpips_fn(p3, g3).item())

    r = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
         "lpips": float(np.mean(lps)), "time_ms": float(np.mean(times) * 1000),
         "params": sum(p.numel() for p in model.parameters())}
    print(f"=== {name} ===")
    print(f"PSNR:  {r['psnr']:.3f} dB")
    print(f"SSIM:  {r['ssim']:.4f}")
    print(f"LPIPS: {r['lpips']:.4f}")
    print(f"Time:  {r['time_ms']:.2f} ms/image")
    print(f"Params: {r['params']:,}\n")
    return r


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RestorationDataset(args.gt_dir, args.noisy_dir)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_ds = random_split(dataset, [train_size, val_size],
                             generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=1)   # batch 1 for honest per-image timing
    print(f"Evaluating on {len(val_ds)} validation pairs\n")

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    # final submitted model lives directly in weights/; the four ablation
    # checkpoints live in weights/baseline_models/ to keep the submitted
    # model unambiguous for a reviewer skimming the folder
    candidates = [
        ("U-Net baseline",              lambda: RestorationUNet(),                  args.ablation_dir, "baseline_model.pth"),
        ("U-Net + FiLM",                lambda: RestorationUNetAdaptive(),          args.ablation_dir, "adaptive_model.pth"),
        ("NAFNet-lite baseline",        lambda: NAFNetLite(),                       args.ablation_dir, "nafnet_baseline_model.pth"),
        ("NAFNet-lite + FiLM",          lambda: NAFNetLiteAdaptive(),               args.ablation_dir, "nafnet_adaptive_model.pth"),
        ("FINAL wide NAFNet+FiLM+VGG",  lambda: NAFNetLiteAdaptive(base_ch=48),     args.weights_dir,   "final_model.pth"),
    ]

    results = {}
    for name, ctor, folder, fname in candidates:
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            print(f"Skipping {name} -- checkpoint not found: {path}\n")
            continue
        model = ctor()
        model.load_state_dict(torch.load(path, map_location=device))
        results[name] = evaluate(model.to(device), name, val_loader, lpips_fn, device)

    if not results:
        print("No checkpoints found in", args.weights_dir)
        return

    print("=" * 88)
    print(f"{'Model':<30} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10} {'ms/img':<10} {'Params':<12}")
    print("=" * 88)
    for name, r in results.items():
        print(f"{name:<30} {r['psnr']:<10.3f} {r['ssim']:<10.4f} {r['lpips']:<10.4f} "
              f"{r['time_ms']:<10.2f} {r['params']:<12,}")


if __name__ == "__main__":
    main()
