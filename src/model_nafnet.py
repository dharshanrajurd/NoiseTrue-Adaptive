import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# NAFNet building blocks
# ============================================================

class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension, for (B, C, H, W) tensors.
    NAFNet uses LayerNorm instead of BatchNorm -- more stable for restoration."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """NAFNet's replacement for nonlinear activations (ReLU/GELU etc).
    Splits channels into two halves and multiplies them elementwise.
    Cheaper than an activation function, works as well or better in practice."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """One NAFNet block: spatial mixing (depthwise conv + simplified channel
    attention) followed by channel mixing (feed-forward), both using
    SimpleGate instead of activations, both wrapped in residual connections
    with a learnable scale (starts at 0 -- block starts as identity, like our
    FiLM zero-init trick, which makes deep stacks of these blocks easy to train)."""

    def __init__(self, ch, expand=2):
        super().__init__()
        dw_ch = ch * expand

        # spatial mixing sub-block
        self.norm1 = LayerNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)  # depthwise
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(          # simplified channel attention
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, ch, 1)

        # channel mixing / feed-forward sub-block
        ffn_ch = ch * expand
        self.norm2 = LayerNorm2d(ch)
        self.conv4 = nn.Conv2d(ch, ffn_ch, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, ch, 1)

        # learnable residual scales, zero-init so each block starts as a
        # pure identity pass-through -- makes stacking many blocks stable
        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        x = x + y * self.gamma

        return x


# ============================================================
# NAFNet-lite restoration model (same skeleton as RestorationUNet)
# ============================================================

class NAFNetLite(nn.Module):
    """Baseline NAFNet-lite: same encoder-bottleneck-decoder shape as your
    original U-Net, but each stage uses a NAFBlock instead of plain conv+ReLU."""

    def __init__(self, base_ch=32):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8  # 32,64,128,256

        self.stem = nn.Conv2d(1, c1, 3, padding=1)

        self.enc1 = NAFBlock(c1)
        self.down1 = nn.Conv2d(c1, c2, 2, stride=2)
        self.enc2 = NAFBlock(c2)
        self.down2 = nn.Conv2d(c2, c3, 2, stride=2)
        self.enc3 = NAFBlock(c3)
        self.down3 = nn.Conv2d(c3, c4, 2, stride=2)

        self.bottleneck = NAFBlock(c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.reduce3 = nn.Conv2d(c3 * 2, c3, 1)   # after concat with skip
        self.dec3 = NAFBlock(c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.reduce2 = nn.Conv2d(c2 * 2, c2, 1)
        self.dec2 = NAFBlock(c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.reduce1 = nn.Conv2d(c1 * 2, c1, 1)
        self.dec1 = NAFBlock(c1)

        # final 2x upsample head -- same pixel-shuffle approach as before
        self.final_up = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, 3, padding=1),
        )

    def _forward_encoder(self, x):
        x0 = self.stem(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.down3(e3)
        return e1, e2, e3, b

    def _forward_decoder(self, e1, e2, e3, b):
        d3 = self.up3(b)
        d3 = self.reduce3(torch.cat([d3, e3], dim=1))
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.reduce2(torch.cat([d2, e2], dim=1))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.reduce1(torch.cat([d1, e1], dim=1))
        d1 = self.dec1(d1)

        return self.final_up(d1)

    def forward(self, x):
        e1, e2, e3, b = self._forward_encoder(x)
        b = self.bottleneck(b)
        return self._forward_decoder(e1, e2, e3, b)


# ============================================================
# FiLM conditioning (same idea as before, zero-init from the start)
# ============================================================

class DegradationEstimator(nn.Module):
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
    def __init__(self, cond_dim, feat_ch):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, feat_ch)
        self.to_shift = nn.Linear(cond_dim, feat_ch)
        # zero-init: FiLM starts as identity, only learns to deviate if it helps
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, feat, cond):
        scale = self.to_scale(cond).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(cond).unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + scale) + shift


class NAFNetLiteAdaptive(NAFNetLite):
    """NAFNet-lite + degradation-aware FiLM conditioning at the bottleneck.
    Same novelty idea as before, now on the stronger backbone."""

    def __init__(self, base_ch=32):
        super().__init__(base_ch=base_ch)
        c4 = base_ch * 8
        self.estimator = DegradationEstimator(out_dim=32)
        self.film = FiLM(cond_dim=32, feat_ch=c4)

    def forward(self, x):
        cond = self.estimator(x)
        e1, e2, e3, b = self._forward_encoder(x)
        b = self.bottleneck(b)
        b = self.film(b, cond)
        return self._forward_decoder(e1, e2, e3, b)
