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
    """Hybrid detector combining spatial and frequency branches"""
    def __init__(self, spatial_model_path, feature_dim=256):
        super(HybridNPRDetector, self).__init__()
        
        # Load spatial branch
        print(f"Loading spatial branch from: {spatial_model_path}")
        checkpoint = torch.load(spatial_model_path, map_location='cpu')
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Check conv1 shape from checkpoint
        conv1_weight = state_dict.get('conv1.weight')
        if conv1_weight is not None:
            conv1_shape = conv1_weight.shape
            print(f"Checkpoint conv1 shape: {conv1_shape}")
            is_custom_conv1 = (conv1_shape[2:] == torch.Size([3, 3]))
        else:
            is_custom_conv1 = False
        
        # Build spatial feature extractor
        import torchvision.models as models
        base_model = models.resnet50(weights=None)
        
        # Modify conv1 if needed
        if is_custom_conv1:
            print("Detected custom 3x3 conv1, modifying architecture...")
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Load state dict - only keep layers that match, ignore fc layers
        model_dict = base_model.state_dict()
        filtered_state_dict = {}
        
        for k, v in state_dict.items():
            # Skip fc layers
            if k.startswith('fc') or k.startswith('classifier'):
                continue
            # Only load if key exists in model and shapes match
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
        
        print(f"Loading {len(filtered_state_dict)}/{len(model_dict)} matching layers")
        base_model.load_state_dict(filtered_state_dict, strict=False)
        
        # Remove fc layer and create feature extractor
        self.spatial_branch = nn.Sequential(*list(base_model.children())[:-1])
        
        # CRITICAL FIX: Detect spatial feature dimension by running a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            spatial_feat_dim = self.spatial_branch(dummy_input).view(1, -1).shape[1]
        print(f"Detected spatial feature dimension: {spatial_feat_dim}")
        
        # Frequency branch
        self.frequency_branch = FrequencyBranch(feature_dim=feature_dim)
        
        # Fusion layers - use detected spatial_feat_dim instead of hardcoded 2048
        self.fusion = nn.Sequential(
            nn.Linear(spatial_feat_dim + feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # Count parameters
        spatial_params = sum(p.numel() for p in self.spatial_branch.parameters())
        freq_params = sum(p.numel() for p in self.frequency_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Spatial branch params: {spatial_params:,}")
        print(f"Frequency branch params: {freq_params:,}")
        print(f"Fusion params: {fusion_params:,}")
        print(f"Trainable params: {trainable_params:,}")
    
    def forward(self, x):
        # Spatial features
        spatial_feat = self.spatial_branch(x)
        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1)
        
        # Frequency features
        freq_feat = self.frequency_branch(x)
        
        # Concatenate and fuse
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        output = self.fusion(combined)
        
        return output
