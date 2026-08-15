import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RestorationUNet(nn.Module):
    """Baseline model: plain U-Net with a 2x pixel-shuffle upsample head.
    Input: noisy image at half resolution. Output: restored image at full resolution."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # final 2x upsample: e.g. 128x128 -> 256x256, or 256x256 -> 512x512
        self.final_up = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final_up(d1)


class DegradationEstimator(nn.Module):
    """Small network that looks at the noisy image and estimates how
    degraded it is. Produces a feature vector used to condition the main network."""

    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    """Turns the degradation vector into a per-channel scale+shift that
    modulates the bottleneck features -- lets the model adapt per image.
    This is the project's novelty: degradation-aware adaptivity."""

    def __init__(self, cond_dim, feat_ch):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, feat_ch)
        self.to_shift = nn.Linear(cond_dim, feat_ch)
        # zero-init: at the start of training FiLM does nothing (scale=0, shift=0),
        # so the model starts equivalent to the plain baseline and only learns to
        # deviate from it if doing so genuinely helps -- gives FiLM a fair chance
        # instead of immediately distorting features with random init
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, feat, cond):
        scale = self.to_scale(cond).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(cond).unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + scale) + shift


class RestorationUNetAdaptive(RestorationUNet):
    """Final model: same U-Net, but the bottleneck features are modulated
    based on an estimate of how degraded the specific input image is."""

    def __init__(self):
        super().__init__()
        self.estimator = DegradationEstimator(out_dim=32)
        self.film = FiLM(cond_dim=32, feat_ch=256)

    def forward(self, x):
        cond = self.estimator(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        b = self.film(b, cond)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final_up(d1)
