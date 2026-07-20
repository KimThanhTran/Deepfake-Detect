import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DCT2D(nn.Module):
    """2D Discrete Cosine Transform using FFT"""
    def __init__(self):
        super(DCT2D, self).__init__()
    
    def forward(self, x):
        # x: (B, 3, H, W)
        # Convert to grayscale
        if x.size(1) == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Apply FFT
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shift)
        
        # Log scale for better visualization
        magnitude = torch.log(magnitude + 1e-8)
        
        return magnitude

class FrequencyBranch(nn.Module):
    """Lightweight CNN for frequency domain features"""
    def __init__(self, feature_dim=256):
        super(FrequencyBranch, self).__init__()
        
        self.dct = DCT2D()
        
        # Lightweight conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(256, feature_dim)
    
    def forward(self, x):
        # Apply DCT
        freq = self.dct(x)
        
        # Conv layers
        x = self.conv1(freq)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class HybridNPRDetector(nn.Module):
    """Hybrid detector: frozen NPR spatial branch + trainable frequency branch.

    The spatial branch is the actual NPR network (truncated ResNet-50 operating
    on the NPR residual), loaded strictly from a trained checkpoint and frozen.
    Only the frequency branch and the fusion head are trained.
    """
    def __init__(self, spatial_model_path, feature_dim=256):
        super(HybridNPRDetector, self).__init__()

        print(f"Loading spatial branch (NPR) from: {spatial_model_path}")
        from util import build_npr_model
        self.spatial_branch, adaptive = build_npr_model(spatial_model_path)
        if adaptive:
            print("AdaptiveNPR checkpoint detected")

        # Freeze spatial branch: no gradients, and BatchNorm stays in eval mode
        for p in self.spatial_branch.parameters():
            p.requires_grad = False
        self.spatial_branch.eval()

        spatial_feat_dim = self.spatial_branch.fc1.in_features  # 512
        print(f"Spatial feature dimension: {spatial_feat_dim}")

        # Frequency branch
        self.frequency_branch = FrequencyBranch(feature_dim=feature_dim)

        self.fusion = nn.Sequential(
            nn.Linear(spatial_feat_dim + feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        info = self.get_trainable_params()
        print(f"Total params: {info['total']:,} | trainable: {info['trainable']:,} | frozen: {info['frozen']:,}")

    def get_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}

    def train(self, mode=True):
        # Keep the frozen spatial branch in eval mode (fixed BatchNorm statistics)
        super().train(mode)
        self.spatial_branch.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            spatial_feat = self.spatial_branch.forward_features(x)

        freq_feat = self.frequency_branch(x)

        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        output = self.fusion(combined)

        return output
