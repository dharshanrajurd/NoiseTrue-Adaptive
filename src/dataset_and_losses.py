import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class RestorationDataset(Dataset):
    """Loads matching GT/NoisyLR .npy pairs by filename."""

    def __init__(self, gt_folder, noisy_folder):
        self.gt_folder = gt_folder
        self.noisy_folder = noisy_folder
        self.files = sorted(os.listdir(gt_folder))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        gt = np.load(os.path.join(self.gt_folder, fname)).astype(np.float32)
        noisy = np.load(os.path.join(self.noisy_folder, fname)).astype(np.float32)

        gt = torch.from_numpy(gt).unsqueeze(0)       # (1, H, W)
        noisy = torch.from_numpy(noisy).unsqueeze(0)  # (1, H/2, W/2)

        return noisy, gt


def charbonnier_loss(pred, target, eps=1e-6):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def sobel_edges(img):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def edge_loss(pred, target):
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))


def combined_loss(pred, target):
    return charbonnier_loss(pred, target) + 0.1 * edge_loss(pred, target)
