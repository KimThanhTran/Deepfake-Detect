"""
Evaluation Script cho Hybrid NPR Detector

Evaluate hybrid model (Spatial + Frequency) trên:
- ForenSynths validation set
- UniversalFakeDetect
- GANGen-Detection
- 8-GAN benchmark (optional)

So sánh với baseline spatial-only model

Author: NPR-DeepfakeDetection Project
Date: 01/12/2025
"""

import sys
import os
import argparse
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from networks.frequency_branch import HybridNPRDetector
from networks.resnet import resnet50
from data import create_dataloader


def evaluate_model(model, dataloader, device, dataset_name="Test"):
    """
    Evaluate model on a dataset
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    print(f"\nEvaluating on {dataset_name}...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f'Evaluating {dataset_name}'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds) * 100
    ap = average_precision_score(all_labels, all_scores) * 100
    
    try:
        auc = roc_auc_score(all_labels, all_scores) * 100
    except:
        auc = 0.0
    
    # Real/Fake accuracy
    real_mask = (all_labels == 0)
    fake_mask = (all_labels == 1)
    
    real_acc = accuracy_score(all_labels[real_mask], all_preds[real_mask]) * 100 if real_mask.sum() > 0 else 0.0
    fake_acc = accuracy_score(all_labels[fake_mask], all_preds[fake_mask]) * 100 if fake_mask.sum() > 0 else 0.0
    
    return {
        'dataset': dataset_name,
        'accuracy': acc,
        'ap': ap,
        'auc': auc,
        'real_acc': real_acc,
        'fake_acc': fake_acc,
        'num_samples': len(all_labels)
    }


def load_hybrid_model(model_path, spatial_model_path, device):
    """
    Load hybrid model
    """
    print(f"Loading Hybrid Model...")
    print(f"  Hybrid weights: {model_path}")
    print(f"  Spatial branch: {spatial_model_path}")
    
    model = HybridNPRDetector(spatial_model_path=spatial_model_path, feature_dim=256)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded successfully\n")
    
    return model


def load_spatial_only_model(model_path, device):
    """
    Load spatial-only baseline model
    """
    print(f"Loading Spatial-Only Model...")
    print(f"  Model path: {model_path}")
    
    model = resnet50(num_classes=1)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded successfully\n")
    
    return model


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"Evaluating Hybrid NPR Detector")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Hybrid model: {args.hybrid_model_path}")
    print(f"Baseline model: {args.baseline_model_path}")
    print(f"{'='*70}\n")
    
    # Load models
    hybrid_model = load_hybrid_model(
        args.hybrid_model_path,
        args.baseline_model_path,
        device
    )
    
    if args.compare_baseline:
        baseline_model = load_spatial_only_model(args.baseline_model_path, device)
    
    # Datasets to evaluate
    datasets = []
    
    # 1. ForenSynths validation
    if os.path.exists(os.path.join(args.dataroot, 'val')):
        print(f"Found ForenSynths validation set")
        val_loader = create_dataloader(
            dataroot=args.dataroot,
            classes=args.classes,
            mode='val',
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        datasets.append(('ForenSynths-Val', val_loader))
    
    # 2. UniversalFakeDetect
    ufd_path = 'dataset/UniversalFakeDetect'
    if os.path.exists(ufd_path):
        print(f"Found UniversalFakeDetect dataset")
        # Implementation depends on dataset structure
        # For now, skip if complex structure
        pass
    
    # 3. GANGen-Detection
    gangen_path = 'dataset/GANGen-Detection'
    if os.path.exists(gangen_path):
        print(f"Found GANGen-Detection dataset")
        # Implementation depends on dataset structure
        pass
    
    print(f"\nFound {len(datasets)} dataset(s) to evaluate\n")
    
    # Results storage
    results_hybrid = []
    results_baseline = []
    
    # Evaluate each dataset
    for dataset_name, dataloader in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        # Evaluate hybrid model
        print(f"\n[1] Hybrid Model (Spatial + Frequency)")
        metrics_hybrid = evaluate_model(hybrid_model, dataloader, device, dataset_name)
        results_hybrid.append(metrics_hybrid)
        
        print(f"\nResults:")
        print(f"  Accuracy: {metrics_hybrid['accuracy']:.2f}%")
        print(f"  AP: {metrics_hybrid['ap']:.2f}%")
        print(f"  AUC: {metrics_hybrid['auc']:.2f}%")
        print(f"  Real Acc: {metrics_hybrid['real_acc']:.2f}%")
        print(f"  Fake Acc: {metrics_hybrid['fake_acc']:.2f}%")
        print(f"  Samples: {metrics_hybrid['num_samples']:,}")
        
        # Evaluate baseline if requested
        if args.compare_baseline:
            print(f"\n[2] Baseline Model (Spatial Only)")
            metrics_baseline = evaluate_model(baseline_model, dataloader, device, dataset_name)
            results_baseline.append(metrics_baseline)
            
            print(f"\nResults:")
            print(f"  Accuracy: {metrics_baseline['accuracy']:.2f}%")
            print(f"  AP: {metrics_baseline['ap']:.2f}%")
            print(f"  AUC: {metrics_baseline['auc']:.2f}%")
            print(f"  Real Acc: {metrics_baseline['real_acc']:.2f}%")
            print(f"  Fake Acc: {metrics_baseline['fake_acc']:.2f}%")
            
            # Comparison
            print(f"\n[3] Improvement (Hybrid vs Baseline)")
            acc_diff = metrics_hybrid['accuracy'] - metrics_baseline['accuracy']
            ap_diff = metrics_hybrid['ap'] - metrics_baseline['ap']
            real_diff = metrics_hybrid['real_acc'] - metrics_baseline['real_acc']
            fake_diff = metrics_hybrid['fake_acc'] - metrics_baseline['fake_acc']
            
            print(f"  Accuracy: {acc_diff:+.2f}%")
            print(f"  AP: {ap_diff:+.2f}%")
            print(f"  Real Acc: {real_diff:+.2f}%")
            print(f"  Fake Acc: {fake_diff:+.2f}%")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Evaluation Summary")
    print(f"{'='*70}")
    
    if len(results_hybrid) > 0:
        avg_acc_hybrid = np.mean([r['accuracy'] for r in results_hybrid])
        avg_ap_hybrid = np.mean([r['ap'] for r in results_hybrid])
        
        print(f"\nHybrid Model:")
        print(f"  Average Accuracy: {avg_acc_hybrid:.2f}%")
        print(f"  Average AP: {avg_ap_hybrid:.2f}%")
        
        if args.compare_baseline and len(results_baseline) > 0:
            avg_acc_baseline = np.mean([r['accuracy'] for r in results_baseline])
            avg_ap_baseline = np.mean([r['ap'] for r in results_baseline])
            
            print(f"\nBaseline Model:")
            print(f"  Average Accuracy: {avg_acc_baseline:.2f}%")
            print(f"  Average AP: {avg_ap_baseline:.2f}%")
            
            print(f"\nOverall Improvement:")
            print(f"  Accuracy: {avg_acc_hybrid - avg_acc_baseline:+.2f}%")
            print(f"  AP: {avg_ap_hybrid - avg_ap_baseline:+.2f}%")
    
    # Save results
    if args.output_csv:
        # Save hybrid results
        with open(args.output_csv, 'w', newline='') as f:
            if len(results_hybrid) > 0:
                writer = csv.DictWriter(f, fieldnames=results_hybrid[0].keys())
                writer.writeheader()
                writer.writerows(results_hybrid)
        
        print(f"\n✓ Results saved to: {args.output_csv}")
        
        # Save comparison if available
        if args.compare_baseline and len(results_baseline) > 0:
            comparison_path = args.output_csv.replace('.csv', '_comparison.csv')
            with open(comparison_path, 'w', newline='') as f:
                fieldnames = ['dataset', 'metric', 'hybrid', 'baseline', 'improvement']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, dataset_name in enumerate([r['dataset'] for r in results_hybrid]):
                    for metric in ['accuracy', 'ap', 'auc', 'real_acc', 'fake_acc']:
                        writer.writerow({
                            'dataset': dataset_name,
                            'metric': metric,
                            'hybrid': f"{results_hybrid[i][metric]:.2f}",
                            'baseline': f"{results_baseline[i][metric]:.2f}",
                            'improvement': f"{results_hybrid[i][metric] - results_baseline[i][metric]:+.2f}"
                        })
            
            print(f"✓ Comparison saved to: {comparison_path}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Hybrid NPR Detector')
    
    # Models
    parser.add_argument('--hybrid_model_path', type=str, required=True,
                        help='Path to hybrid model checkpoint')
    parser.add_argument('--baseline_model_path', type=str,
                        default='trained_model_forensynths_100acc/model_epoch_last.pth',
                        help='Path to baseline spatial-only model')
    parser.add_argument('--compare_baseline', action='store_true',
                        help='Compare with baseline model')
    
    # Data
    parser.add_argument('--dataroot', type=str, default='dataset/ForenSynths',
                        help='Path to ForenSynths dataset')
    parser.add_argument('--classes', nargs='+', default=['car', 'cat', 'chair', 'horse'],
                        help='Classes to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output_csv', type=str, default='hybrid_evaluation_results.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    main(args)
