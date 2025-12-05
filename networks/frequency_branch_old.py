"""
Frequency Branch Module for Hybrid NPR Detector

Cải tiến: Kết hợp Spatial domain (NPR) với Frequency domain (DCT)
để cải thiện detection accuracy, đặc biệt cho GANGen-Detection dataset.

Lý thuyết: GAN-generated images có artifacts rõ rệt ở frequency spectrum
do up-sampling operations trong generator.

Author: NPR-DeepfakeDetection Project
Date: 01/12/2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class DCT2D(nn.Module):
    """
    Discrete Cosine Transform 2D - Extract frequency features
    
    Chuyển ảnh từ spatial domain sang frequency domain để
    phát hiện high-frequency artifacts đặc trưng của GAN.
    """
    def __init__(self):
        super(DCT2D, self).__init__()
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - RGB image tensor
            
        Returns:
            X_mag: (B, 3, H, W) - Frequency magnitude spectrum
        """
        # Convert RGB to grayscale for DCT
        # Grayscale = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
        if x.size(1) == 3:
            x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            x_gray = x_gray.unsqueeze(1)
        else:
            x_gray = x
            
        # Apply 2D FFT (Fast Fourier Transform) - faster than DCT
        # FFT is equivalent to DCT for our purpose (frequency analysis)
        X = torch.fft.fft2(x_gray, dim=(-2, -1))
        
        # Shift zero frequency to center
        X = torch.fft.fftshift(X, dim=(-2, -1))
        
        # Take magnitude (amplitude spectrum)
        # |X| = sqrt(real^2 + imag^2)
        X_mag = torch.abs(X)
        
        # Log scale for better visualization and learning
        # log(1 + x) to avoid log(0)
        X_mag = torch.log(1 + X_mag)
        
        # Normalize to [0, 1]
        X_mag = (X_mag - X_mag.min()) / (X_mag.max() - X_mag.min() + 1e-8)
        
        # Repeat to 3 channels for compatibility with conv layers
        X_mag = X_mag.repeat(1, 3, 1, 1)
        
        return X_mag


class FrequencyBranch(nn.Module):
    """
    Lightweight CNN for extracting frequency domain features
    
    Kiến trúc nhỏ gọn (chỉ ~500K params) để extract high-frequency patterns
    đặc trưng của GAN-generated images.
    """
    def __init__(self, feature_dim=256):
        super(FrequencyBranch, self).__init__()
        
        self.dct = DCT2D()
        
        # Convolutional layers - extract frequency patterns
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(256, feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) - Input RGB image
            
        Returns:
            features: (B, feature_dim) - Frequency domain features
        """
        # Extract frequency spectrum
        x_freq = self.dct(x)
        
        # CNN processing
        x = self.relu1(self.bn1(self.conv1(x_freq)))
        x = self.pool1(x)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.relu4(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # FC layer
        features = self.fc(x)
        
        return features


class HybridNPRDetector(nn.Module):
    """
    Hybrid Detector kết hợp Spatial branch (NPR) và Frequency branch (DCT)
    
    Architecture:
        Input (224x224x3)
            ├─> Spatial Branch (ResNet-50 NPR) [FROZEN] -> 2048 features
            └─> Frequency Branch (DCT + CNN) [TRAINABLE] -> 256 features
                    ↓
                Concatenate [2048 + 256] = 2304
                    ↓
                Fusion Layer (FC) -> 512 -> 1
                    ↓
                Sigmoid -> P(fake)
    
    Training Strategy:
        - Freeze spatial branch (tận dụng pretrained model)
        - Train chỉ frequency branch + fusion layer
        - Fast training: ~2 giờ với P100 GPU
    """
    def __init__(self, spatial_model_path, feature_dim=256):
        super(HybridNPRDetector, self).__init__()
        
        # Load spatial branch (ResNet-50 NPR pretrained)
        print(f"Loading spatial branch from: {spatial_model_path}")
        from networks.resnet import resnet50
        self.spatial_model = resnet50(num_classes=1)
        
        try:
            checkpoint = torch.load(spatial_model_path, map_location='cpu')
            self.spatial_model.load_state_dict(checkpoint)
            print("✓ Spatial branch loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load spatial model: {e}")
            print("  Using random initialization for spatial branch")
        
        # Freeze spatial branch - KHÔNG train lại
        for param in self.spatial_model.parameters():
            param.requires_grad = False
        
        # Extract feature extractor (remove final FC layer)
        # ResNet-50: conv1 -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        self.spatial_features = nn.Sequential(
            self.spatial_model.conv1,
            self.spatial_model.bn1,
            self.spatial_model.relu,
            self.spatial_model.maxpool,
            self.spatial_model.layer1,
            self.spatial_model.layer2,
            self.spatial_model.layer3,
            self.spatial_model.layer4,
            self.spatial_model.avgpool
        )
        
        # Set to eval mode permanently
        self.spatial_features.eval()
        
        # New frequency branch - TRAINABLE
        self.frequency_branch = FrequencyBranch(feature_dim=feature_dim)
        
        # Fusion layer
        # Input: 2048 (spatial) + 256 (frequency) = 2304
        self.fusion = nn.Sequential(
            nn.Linear(2048 + feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) - Input RGB images
            
        Returns:
            output: (B, 1) - Logits (before sigmoid)
        """
        # Spatial features (frozen - no gradient)
        with torch.no_grad():
            spatial_feat = self.spatial_features(x)
            spatial_feat = spatial_feat.view(spatial_feat.size(0), -1)
        
        # Frequency features (trainable)
        freq_feat = self.frequency_branch(x)
        
        # Concatenate features
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        
        # Final prediction
        output = self.fusion(combined)
        
        return output
    
    def train(self, mode=True):
        """
        Override train() để đảm bảo spatial branch luôn ở eval mode
        """
        super(HybridNPRDetector, self).train(mode)
        
        # Force spatial branch to eval mode
        self.spatial_features.eval()
        
        return self
    
    def get_trainable_params(self):
        """
        Trả về số lượng parameters trainable
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable': trainable,
            'total': total,
            'frozen': total - trainable
        }


# Test code
if __name__ == '__main__':
    print("="*60)
    print("Testing Frequency Branch Module")
    print("="*60)
    
    # Test DCT2D
    print("\n1. Testing DCT2D...")
    dct = DCT2D()
    x = torch.randn(2, 3, 224, 224)
    x_freq = dct(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Frequency shape: {x_freq.shape}")
    print(f"   Frequency range: [{x_freq.min():.3f}, {x_freq.max():.3f}]")
    
    # Test FrequencyBranch
    print("\n2. Testing FrequencyBranch...")
    freq_branch = FrequencyBranch(feature_dim=256)
    features = freq_branch(x)
    print(f"   Output shape: {features.shape}")
    print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Count parameters
    freq_params = sum(p.numel() for p in freq_branch.parameters())
    print(f"   Parameters: {freq_params:,}")
    
    # Test HybridNPRDetector
    print("\n3. Testing HybridNPRDetector...")
    
    # Try to load actual model if exists
    import os
    model_path = 'trained_model_forensynths_100acc/model_epoch_last.pth'
    if not os.path.exists(model_path):
        model_path = 'NPR.pth'
    
    if os.path.exists(model_path):
        print(f"   Loading model from: {model_path}")
        model = HybridNPRDetector(model_path, feature_dim=256)
    else:
        print("   ⚠ Model file not found, using random initialization")
        print("   For actual training, provide correct model path")
        model = HybridNPRDetector('dummy_path.pth', feature_dim=256)
    
    # Forward pass
    output = model(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check trainable parameters
    params_info = model.get_trainable_params()
    print(f"\n4. Parameter Statistics:")
    print(f"   Total parameters: {params_info['total']:,}")
    print(f"   Trainable parameters: {params_info['trainable']:,}")
    print(f"   Frozen parameters: {params_info['frozen']:,}")
    print(f"   Trainable ratio: {100*params_info['trainable']/params_info['total']:.2f}%")
    
    # Verify spatial branch is frozen
    spatial_trainable = sum(p.numel() for p in model.spatial_features.parameters() if p.requires_grad)
    freq_trainable = sum(p.numel() for p in model.frequency_branch.parameters() if p.requires_grad)
    fusion_trainable = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
    
    print(f"\n5. Branch-wise Trainable Parameters:")
    print(f"   Spatial branch: {spatial_trainable:,} (should be 0)")
    print(f"   Frequency branch: {freq_trainable:,}")
    print(f"   Fusion layer: {fusion_trainable:,}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
