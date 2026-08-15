import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.dataset_and_losses import charbonnier_loss, edge_loss  # reuse what already works


class VGGPerceptualLoss(nn.Module):
    """Compares images in VGG feature space instead of raw pixels -- this is
    what actually targets 'does it look right', which is what LPIPS measures.
    Your training loss never included this before -- it's the most direct fix
    for the LPIPS gap you were trying to close."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights='DEFAULT').features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # grayscale (1 channel) -> fake RGB (3 channel), VGG expects 3-channel input
        pred3 = pred.repeat(1, 3, 1, 1)
        target3 = target.repeat(1, 3, 1, 1)

        pred_norm = (pred3 - self.mean) / self.std
        target_norm = (target3 - self.mean) / self.std

        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)

        return F.l1_loss(pred_feat, target_feat)


def combined_loss_v2(pred, target, vgg_loss_fn, vgg_weight=0.05):
    """Charbonnier (pixel accuracy) + edge (structure) + VGG (perceptual quality).
    vgg_weight kept small since VGG loss values are naturally larger in scale
    than pixel losses -- 0.05 is a reasonable starting point, tunable if needed."""
    pixel = charbonnier_loss(pred, target)
    edge = edge_loss(pred, target)
    perceptual = vgg_loss_fn(pred, target)
    return pixel + 0.1 * edge + vgg_weight * perceptual
