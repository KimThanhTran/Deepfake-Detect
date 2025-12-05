import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveNPR(nn.Module):
    """Adaptive Neighboring Pixel Relationship module.

    Computes a residual map capturing inconsistencies introduced by generative models.
    Extends fixed NPR (downsample->upsample difference) with learnable multi-scale kernels.
    """

    def __init__(self, in_channels=3, scales=(2,4), kernel_size=3):
        super().__init__()
        self.scales = scales
        padding = kernel_size // 2
        # Per-scale depthwise conv to learn local pixel affinity adjustments
        self.local_filters = nn.ModuleDict({
            f's{s}': nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
            for s in scales
        })
        # Gating to weight scale contributions
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.reduction = nn.Conv2d(in_channels* (len(scales)+1), in_channels, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.local_filters.values():
            nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in')
        for m in self.gate:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.reduction.weight)

    def forward(self, x):
        # Base NPR residual (fixed 2x down-up)
        base = x - F.interpolate(F.interpolate(x, scale_factor=0.5, mode='nearest'), scale_factor=2, mode='nearest')
        feats = [base]
        gate = self.gate(x)
        for s in self.scales:
            # Multi-scale smoothing then difference reconstruction
            down = F.interpolate(x, scale_factor=1.0/ s, mode='nearest')
            up = F.interpolate(down, size=x.shape[-2:], mode='nearest')
            raw = x - up
            adj = self.local_filters[f's{s}'](raw) * gate
            feats.append(adj)
        cat = torch.cat(feats, dim=1)
        out = self.reduction(cat)
        return out
