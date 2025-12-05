"""
Visualization Script - Frequency Spectrum Analysis

So sánh frequency spectrum giữa real vs fake images
để chứng minh GAN artifacts ở frequency domain

Author: NPR-DeepfakeDetection Project
Date: 01/12/2025
"""

import sys
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Add project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from networks.frequency_branch import DCT2D, HybridNPRDetector


def load_image(image_path, size=224):
    """
    Load and preprocess image
    """
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    # Also load original for display
    img_display = Image.open(image_path).convert('RGB')
    img_display = img_display.resize((size, size))
    
    return img_tensor, np.array(img_display)


def visualize_frequency_comparison(real_img_path, fake_img_path, output_path='frequency_comparison.png'):
    """
    Visualize frequency spectrum comparison between real and fake images
    """
    print(f"Loading images...")
    print(f"  Real: {real_img_path}")
    print(f"  Fake: {fake_img_path}")
    
    # Load images
    real_tensor, real_display = load_image(real_img_path)
    fake_tensor, fake_display = load_image(fake_img_path)
    
    # Extract frequency spectra
    print(f"Extracting frequency spectra...")
    dct = DCT2D()
    
    with torch.no_grad():
        real_freq = dct(real_tensor)[0, 0].numpy()  # Take first channel
        fake_freq = dct(fake_tensor)[0, 0].numpy()
    
    # Create visualization
    print(f"Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Real image
    axes[0, 0].imshow(real_display)
    axes[0, 0].set_title('Real Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Real frequency spectrum
    im1 = axes[0, 1].imshow(real_freq, cmap='jet')
    axes[0, 1].set_title('Real Frequency Spectrum', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Real frequency profile (center line)
    center = real_freq.shape[0] // 2
    axes[0, 2].plot(real_freq[center, :], 'b-', linewidth=2, label='Real')
    axes[0, 2].set_title('Frequency Profile (Center Line)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Frequency')
    axes[0, 2].set_ylabel('Magnitude')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Row 2: Fake image
    axes[1, 0].imshow(fake_display)
    axes[1, 0].set_title('Fake Image (GAN)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Fake frequency spectrum
    im2 = axes[1, 1].imshow(fake_freq, cmap='jet')
    axes[1, 1].set_title('Fake Frequency Spectrum', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    # Comparison plot
    axes[1, 2].plot(real_freq[center, :], 'b-', linewidth=2, label='Real', alpha=0.7)
    axes[1, 2].plot(fake_freq[center, :], 'r-', linewidth=2, label='Fake', alpha=0.7)
    axes[1, 2].set_title('Frequency Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Frequency')
    axes[1, 2].set_ylabel('Magnitude')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    # Calculate difference metrics
    diff = np.abs(real_freq - fake_freq)
    mse = np.mean(diff ** 2)
    mae = np.mean(diff)
    
    print(f"\nFrequency Spectrum Difference:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return mse, mae


def visualize_multiple_samples(real_paths, fake_paths, output_path='frequency_gallery.png'):
    """
    Visualize multiple real vs fake pairs
    """
    n_samples = len(real_paths)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    dct = DCT2D()
    
    for i, (real_path, fake_path) in enumerate(zip(real_paths, fake_paths)):
        # Load images
        real_tensor, real_display = load_image(real_path)
        fake_tensor, fake_display = load_image(fake_path)
        
        # Frequency spectra
        with torch.no_grad():
            real_freq = dct(real_tensor)[0, 0].numpy()
            fake_freq = dct(fake_tensor)[0, 0].numpy()
        
        # Plot
        if n_samples == 1:
            ax = axes
        else:
            ax = axes[i]
        
        ax[0].imshow(real_display)
        ax[0].set_title(f'Real {i+1}')
        ax[0].axis('off')
        
        ax[1].imshow(real_freq, cmap='jet')
        ax[1].set_title(f'Real Freq {i+1}')
        ax[1].axis('off')
        
        ax[2].imshow(fake_display)
        ax[2].set_title(f'Fake {i+1}')
        ax[2].axis('off')
        
        ax[3].imshow(fake_freq, cmap='jet')
        ax[3].set_title(f'Fake Freq {i+1}')
        ax[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gallery saved to: {output_path}")


def main(args):
    print(f"\n{'='*70}")
    print(f"Frequency Spectrum Visualization")
    print(f"{'='*70}\n")
    
    if args.mode == 'single':
        # Single pair comparison
        if not args.real_image or not args.fake_image:
            print("Error: Please provide --real_image and --fake_image paths")
            return
        
        visualize_frequency_comparison(
            args.real_image,
            args.fake_image,
            args.output
        )
    
    elif args.mode == 'auto':
        # Auto-select samples from dataset
        print("Auto-selecting samples from dataset...")
        
        import glob
        import random
        
        # Find real images
        real_pattern = os.path.join(args.dataroot, 'val', '*', '0_real', '*.png')
        real_images = glob.glob(real_pattern)
        
        # Find fake images
        fake_pattern = os.path.join(args.dataroot, 'val', '*', '1_fake', '*.png')
        fake_images = glob.glob(fake_pattern)
        
        if len(real_images) == 0 or len(fake_images) == 0:
            print(f"Error: No images found in {args.dataroot}")
            return
        
        print(f"Found {len(real_images)} real and {len(fake_images)} fake images")
        
        # Random sample
        n_samples = min(args.num_samples, len(real_images), len(fake_images))
        real_samples = random.sample(real_images, n_samples)
        fake_samples = random.sample(fake_images, n_samples)
        
        print(f"Visualizing {n_samples} sample(s)...")
        
        if n_samples == 1:
            visualize_frequency_comparison(
                real_samples[0],
                fake_samples[0],
                args.output
            )
        else:
            visualize_multiple_samples(
                real_samples,
                fake_samples,
                args.output
            )
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Frequency Spectrum')
    
    # Mode
    parser.add_argument('--mode', type=str, choices=['single', 'auto'], default='auto',
                        help='Visualization mode: single pair or auto-select from dataset')
    
    # Single mode
    parser.add_argument('--real_image', type=str,
                        help='Path to real image')
    parser.add_argument('--fake_image', type=str,
                        help='Path to fake image')
    
    # Auto mode
    parser.add_argument('--dataroot', type=str, default='dataset/ForenSynths',
                        help='Path to dataset root')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to visualize')
    
    # Output
    parser.add_argument('--output', type=str, default='frequency_visualization.png',
                        help='Output image path')
    
    args = parser.parse_args()
    
    main(args)
