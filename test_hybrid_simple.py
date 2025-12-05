"""
Simple evaluation script for Hybrid NPR Detector
Test on GANGen-Detection dataset
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import model
from networks.frequency_branch import HybridNPRDetector

# Simple dataset class
class SimpleImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        
        # GANGen-Detection structure: root/[GAN_type]/0_real or 1_fake/*.png
        # Scan all subdirectories
        if os.path.exists(root):
            for gan_type in os.listdir(root):
                gan_dir = os.path.join(root, gan_type)
                if not os.path.isdir(gan_dir):
                    continue
                
                real_dir = os.path.join(gan_dir, '0_real')
                fake_dir = os.path.join(gan_dir, '1_fake')
                
                if os.path.exists(real_dir):
                    for fname in os.listdir(real_dir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(real_dir, fname), 0))
                
                if os.path.exists(fake_dir):
                    for fname in os.listdir(fake_dir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(fake_dir, fname), 1))
        
        print(f"Found {len(self.samples)} images ({len([s for s in self.samples if s[1]==0])} real, {len([s for s in self.samples if s[1]==1])} fake)")

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def evaluate(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i % 10 == 0:
                print(f"  Batch {i}/{len(dataloader)}")
            
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    
    # Real/Fake accuracy
    real_mask = (all_labels == 0)
    fake_mask = (all_labels == 1)
    
    real_acc = 100.0 * (all_preds[real_mask] == all_labels[real_mask]).sum() / real_mask.sum() if real_mask.sum() > 0 else 0
    fake_acc = 100.0 * (all_preds[fake_mask] == all_labels[fake_mask]).sum() / fake_mask.sum() if fake_mask.sum() > 0 else 0
    
    return accuracy, real_acc, fake_acc

def main():
    print("="*70)
    print("Hybrid NPR Detector Evaluation - GANGen-Detection")
    print("="*70)
    
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model_path = "hybrid_model_best.pth"
    spatial_checkpoint = "model_epoch_last_3090.pth"
    dataset_path = "dataset/GANGen-Detection"
    
    # Check files
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return
    
    if not os.path.exists(spatial_checkpoint):
        print(f"ERROR: Spatial checkpoint not found: {spatial_checkpoint}")
        return
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        return
    
    # Load model
    print("\nLoading model...")
    model = HybridNPRDetector(
        spatial_model_path=spatial_checkpoint,
        feature_dim=256
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"Val Acc: {checkpoint.get('val_acc', '?'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"\nLoading dataset: {dataset_path}")
    dataset = SimpleImageDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Evaluate
    accuracy, real_acc, fake_acc = evaluate(model, dataloader, device)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS - GANGen-Detection")
    print("="*70)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Real Accuracy:    {real_acc:.2f}%")
    print(f"Fake Accuracy:    {fake_acc:.2f}%")
    print("="*70)
    
    # Comparison with baseline
    print("\nBaseline NPR:")
    print("  Overall: 64.37%")
    print("  Real:    32.96%")
    print("  Fake:    95.78%")
    print(f"\nImprovement:")
    print(f"  Overall: {accuracy - 64.37:+.2f}%")
    print(f"  Real:    {real_acc - 32.96:+.2f}%")
    print(f"  Fake:    {fake_acc - 95.78:+.2f}%")

if __name__ == "__main__":
    main()
