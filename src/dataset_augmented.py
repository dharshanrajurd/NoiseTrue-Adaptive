import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class RestorationDatasetAugmented(Dataset):
    """Same as RestorationDataset, but randomly flips each pair horizontally
    and/or vertically during training. Since chip/inspection structures don't
    have a 'correct' orientation, this effectively multiplies your training
    diversity for free -- helps generalization, which matters for the OOD
    test set specifically."""

    def __init__(self, gt_folder, noisy_folder, augment=True):
        self.gt_folder = gt_folder
        self.noisy_folder = noisy_folder
        self.files = sorted(os.listdir(gt_folder))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        gt = np.load(os.path.join(self.gt_folder, fname)).astype(np.float32)
        noisy = np.load(os.path.join(self.noisy_folder, fname)).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                gt = np.ascontiguousarray(gt[:, ::-1])       # horizontal flip
                noisy = np.ascontiguousarray(noisy[:, ::-1])
            if random.random() < 0.5:
                gt = np.ascontiguousarray(gt[::-1, :])       # vertical flip
                noisy = np.ascontiguousarray(noisy[::-1, :])

        gt = torch.from_numpy(gt).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        return noisy, gt
