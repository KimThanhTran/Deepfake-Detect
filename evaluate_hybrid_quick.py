"""
Quick Evaluation Script for Hybrid Model
Evaluate on ForenSynths, UniversalFakeDetect, GANGen-Detection
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

# Import from networks
from networks.frequency_branch import HybridNPRDetector
from data.datasets import ForenSynthsDataset

def evaluate_dataset(model, dataroot, batch_size=32, device='cpu'):
    """Evaluate model on a dataset"""
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = ForenSynthsDataset(dataroot, split='test', transform=transform)
    if len(test_dataset) == 0:
        print(f"⚠️ No test samples found in {dataroot}")
        return None
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate metrics
    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    total = len(all_labels)
    accuracy = 100. * correct / total
    
    # Real/Fake accuracy
    real_mask = [l == 0 for l in all_labels]
    fake_mask = [l == 1 for l in all_labels]
    
    real_correct = sum([p == l for p, l, m in zip(all_preds, all_labels, real_mask) if m])
    fake_correct = sum([p == l for p, l, m in zip(all_preds, all_labels, fake_mask) if m])
    
    real_total = sum(real_mask)
    fake_total = sum(fake_mask)
    
    real_acc = 100. * real_correct / real_total if real_total > 0 else 0.0
    fake_acc = 100. * fake_correct / fake_total if fake_total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'real_acc': real_acc,
        'fake_acc': fake_acc,
        'total': total,
        'real_total': real_total,
        'fake_total': fake_total
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to hybrid_model_best.pth')
    parser.add_argument('--spatial_checkpoint', type=str, default='model_epoch_last_3090.pth', help='Original spatial checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading hybrid model from: {args.model_path}")
    
    # First create model architecture
    model = HybridNPRDetector(
        spatial_model_path=args.spatial_checkpoint,
        feature_dim=256,
        freeze_spatial=False
    )
    
    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded! Trained epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Val accuracy: {checkpoint.get('val_acc', 0):.2f}%\n")
    
    # Dataset paths
    datasets = {
        'ForenSynths': 'dataset/ForenSynths',
        'UniversalFakeDetect': 'dataset/UniversalFakeDetect',
        'GANGen-Detection': 'dataset/GANGen-Detection'
    }
    
    results = {}
    
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    for name, path in datasets.items():
        if not Path(path).exists():
            print(f"\n⚠️ {name}: Dataset not found at {path}")
            continue
        
        print(f"\n📊 Testing on {name}...")
        print("-"*70)
        
        result = evaluate_dataset(model, path, args.batch_size, device)
        
        if result:
            results[name] = result
            print(f"Overall Accuracy: {result['accuracy']:.2f}%")
            print(f"Real Accuracy:    {result['real_acc']:.2f}% ({result['real_total']} samples)")
            print(f"Fake Accuracy:    {result['fake_acc']:.2f}% ({result['fake_total']} samples)")
            print(f"Total samples:    {result['total']}")
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        print(f"\n{'Dataset':<25} {'Accuracy':>10} {'Real Acc':>10} {'Fake Acc':>10}")
        print("-"*70)
        
        for name, res in results.items():
            print(f"{name:<25} {res['accuracy']:>9.2f}% {res['real_acc']:>9.2f}% {res['fake_acc']:>9.2f}%")
        
        # Calculate mean
        mean_acc = sum(r['accuracy'] for r in results.values()) / len(results)
        print("-"*70)
        print(f"{'Overall Mean':<25} {mean_acc:>9.2f}%")
        
        # Highlight GANGen results
        if 'GANGen-Detection' in results:
            gan_result = results['GANGen-Detection']
            print("\n" + "="*70)
            print("🎯 GANGen-Detection (CRITICAL METRIC)")
            print("="*70)
            print(f"Overall Accuracy:  {gan_result['accuracy']:.2f}%")
            print(f"Real Accuracy:     {gan_result['real_acc']:.2f}% ← KEY METRIC")
            print(f"Fake Accuracy:     {gan_result['fake_acc']:.2f}%")

if __name__ == '__main__':
    main()
