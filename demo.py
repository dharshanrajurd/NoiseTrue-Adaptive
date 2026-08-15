"""
NoiseTrue-Adaptive -- live demo script for screen recording.

Runs the final model end-to-end in front of the camera:
  1. Shows a degraded input and its ground truth
  2. Restores it live, with timing
  3. Shows noisy / restored / ground truth side by side with PSNR + SSIM
  4. Repeats for a few images
  5. Runs a batch throughput benchmark and prints the headline speed number

Everything prints slowly enough to narrate over.

REQUIREMENTS -- all in the same folder as this script:
    model_nafnet.py
    final_model.pth

Usage:
    python demo.py
"""

import os
import time
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from model_nafnet import NAFNetLiteAdaptive

# ---- EDIT THESE PATHS ----
GT_FOLDER = r"C:\Users\dhars\Desktop\train-semicon\train\GT"
NOISY_FOLDER = r"C:\Users\dhars\Desktop\train-semicon\train\NoisyLR"
MODEL_PATH = r"final_model.pth"

NUM_DEMO_IMAGES = 3      # how many images to show one-by-one
BENCH_IMAGES = 200       # how many to run for the speed benchmark
PAUSE = 1.2              # seconds between printed lines, so you can narrate
# --------------------------


def say(msg, pause=None):
    print(msg, flush=True)
    time.sleep(PAUSE if pause is None else pause)


def banner(msg):
    print()
    print("=" * 72, flush=True)
    print(f"  {msg}", flush=True)
    print("=" * 72, flush=True)
    time.sleep(PAUSE)


# =====================================================================
banner("NoiseTrue-Adaptive  |  live restoration demo")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
say(f"Device: {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

say("Loading final model: wide NAFNet-lite + degradation-aware FiLM conditioning...")
model = NAFNetLiteAdaptive(base_ch=48).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

n_params = sum(p.numel() for p in model.parameters())
say(f"Model loaded.  {n_params:,} parameters  ({n_params/1e6:.2f} M)")

files = sorted(f for f in os.listdir(NOISY_FOLDER) if f.endswith(".npy"))
say(f"Found {len(files)} degraded images in the input folder.")


# =====================================================================
banner("PART 1  |  Restoring individual images")

random.seed(7)
demo_files = random.sample(files, NUM_DEMO_IMAGES)

for k, fname in enumerate(demo_files, 1):
    print()
    say(f"--- Image {k} of {NUM_DEMO_IMAGES}:  {fname} ---")

    noisy = np.load(os.path.join(NOISY_FOLDER, fname)).astype(np.float32)
    gt = np.load(os.path.join(GT_FOLDER, fname)).astype(np.float32)

    say(f"  Degraded input : {noisy.shape[0]}x{noisy.shape[1]}   "
        f"value range [{noisy.min():.3f}, {noisy.max():.3f}]")
    say(f"  Ground truth   : {gt.shape[0]}x{gt.shape[1]}   "
        f"value range [{gt.min():.3f}, {gt.max():.3f}]")
    if noisy.max() > 1.0 or noisy.min() < 0.0:
        say("  Note: input exceeds [0,1] -- that is the speckle noise, and it is preserved, not clipped.")

    x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)

    # warm up once so the timing reflects steady state, not CUDA init
    with torch.no_grad():
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    say("  Restoring...", pause=0.4)
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000

    restored = out.clamp(0, 1).cpu().numpy()[0, 0]

    p = psnr_fn(gt, restored, data_range=1.0)
    s = ssim_fn(gt, restored, data_range=1.0)

    say(f"  Done in {dt:.2f} ms   ->   output {restored.shape[0]}x{restored.shape[1]}")
    say(f"  PSNR: {p:.2f} dB     SSIM: {s:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    fig.suptitle(f"{fname}    |    restored in {dt:.2f} ms    |    "
                 f"PSNR {p:.2f} dB    SSIM {s:.4f}",
                 fontsize=13, fontweight="bold")
    ax[0].imshow(np.clip(noisy, 0, 1), cmap="gray")
    ax[0].set_title(f"Degraded input\n{noisy.shape[0]}x{noisy.shape[1]}", fontsize=11)
    ax[1].imshow(restored, cmap="gray")
    ax[1].set_title(f"Restored by our model\n{restored.shape[0]}x{restored.shape[1]}",
                    fontsize=11, color="darkgreen", fontweight="bold")
    ax[2].imshow(gt, cmap="gray")
    ax[2].set_title(f"Ground truth\n{gt.shape[0]}x{gt.shape[1]}", fontsize=11)
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    say("  Showing result -- close the image window to continue.", pause=0.3)
    plt.show()


# =====================================================================
banner(f"PART 2  |  Throughput benchmark on {BENCH_IMAGES} images")

say("Timing the full pipeline: disk read -> preprocess -> model -> clip -> back to CPU.")
say("This is the end-to-end number, not just the forward pass.", pause=1.5)

bench_files = files[:BENCH_IMAGES]

# warm up
_warm = np.load(os.path.join(NOISY_FOLDER, bench_files[0])).astype(np.float32)
with torch.no_grad():
    _ = model(torch.from_numpy(_warm).unsqueeze(0).unsqueeze(0).to(device))
if device.type == "cuda":
    torch.cuda.synchronize()

print()
t_start = time.time()
for i, fname in enumerate(bench_files, 1):
    arr = np.load(os.path.join(NOISY_FOLDER, fname)).astype(np.float32)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    _ = out.clamp(0, 1).cpu().numpy()[0, 0]
    if i % 25 == 0:
        elapsed = time.time() - t_start
        print(f"  {i:>4}/{len(bench_files)} images   "
              f"{elapsed:.2f}s elapsed   "
              f"{elapsed/i*1000:.2f} ms/image", flush=True)

if device.type == "cuda":
    torch.cuda.synchronize()
total = time.time() - t_start

print()
say(f"Processed {len(bench_files)} images in {total:.2f} seconds.")
say(f"Average: {total/len(bench_files)*1000:.2f} ms per image "
    f"({len(bench_files)/total:.1f} images/second).")

banner("Demo complete")
say("Final model: wide NAFNet-lite + FiLM, 2.73 M parameters.")
say("Validation set results: PSNR 28.546 dB | SSIM 0.7682 | LPIPS 0.1938 -- best of all five models tested.")
print()
