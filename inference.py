"""
Standalone inference script -- the mandatory submission deliverable.

Loads every degraded .npy image from --input_dir, restores it with the final
model, and writes each restored image to --output_dir under the same filename.

Runs on GPU automatically when one is available. Images that share a shape are
batched together for throughput. Outputs are clipped to [0,1] before saving,
since the evaluator scores files exactly as written.

No source-code edits are required -- everything is set via command line.

Usage:
    python inference.py --input_dir <degraded_npy_folder> --output_dir <restored_folder>
"""

import os
import sys
import time
import argparse
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.model_nafnet import NAFNetLiteAdaptive


def parse_args():
    p = argparse.ArgumentParser(description="NoiseTrue-Adaptive restoration inference")
    p.add_argument("--input_dir", required=True,
                   help="Directory containing degraded .npy images")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write restored .npy images")
    p.add_argument("--model_path", default=os.path.join(HERE, "weights", "final_model.pth"),
                   help="Path to the trained checkpoint")
    p.add_argument("--base_ch", type=int, default=48,
                   help="Model width used at training time (48 for the submitted checkpoint)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Maximum images per batch when their shapes match")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    model = NAFNetLiteAdaptive(base_ch=args.base_ch)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    input_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    if not input_files:
        print(f"No .npy files found in {args.input_dir}. Nothing to do.")
        return
    print(f"Found {len(input_files)} degraded images to restore.")

    # group by shape so same-size images can be batched (the set mixes 128x128 and 256x256)
    groups = defaultdict(list)
    for fname in input_files:
        arr = np.load(os.path.join(args.input_dir, fname))
        groups[arr.shape].append(fname)

    t0 = time.time()
    processed = 0

    with torch.no_grad():
        for shape, filenames in groups.items():
            for i in range(0, len(filenames), args.batch_size):
                batch_files = filenames[i:i + args.batch_size]

                arrays = [np.load(os.path.join(args.input_dir, f)).astype(np.float32)
                          for f in batch_files]
                batch = torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)

                out = model(batch).clamp(0, 1).cpu().numpy()

                for j, fname in enumerate(batch_files):
                    np.save(os.path.join(args.output_dir, fname),
                            out[j, 0].astype(np.float32))

                processed += len(batch_files)
                print(f"  Processed {processed}/{len(input_files)} images...", flush=True)

    total = time.time() - t0
    print(f"\nDone. Restored {len(input_files)} images in {total:.2f}s "
          f"({total / len(input_files) * 1000:.2f} ms/image average, including I/O).")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
