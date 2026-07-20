"""
Training Script cho Frequency Branch của Hybrid NPR Detector

Freeze spatial branch (ResNet-50 NPR đã train), chỉ train:
- Frequency branch (DCT + CNN)
- Fusion layer

Training time: ~2-3 giờ với P100 GPU
Expected improvement: +4-6% overall accuracy

Author: NPR-DeepfakeDetection Project
Date: 01/12/2025
"""

import sys
import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from networks.frequency_branch import HybridNPRDetector
from data import create_dataloader
from util import Logger


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """
    Train model for one epoch
    """
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).float()
        
        # Mixed precision training
        with autocast(enabled=scaler.is_enabled()):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Calculate metrics
    train_loss = running_loss / len(dataloader)
    train_acc = accuracy_score(all_labels, all_preds) * 100
    
    return train_loss, train_acc


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate model
    """
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Calculate metrics
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds) * 100
    val_ap = average_precision_score(all_labels, all_scores) * 100
    
    try:
        val_auc = roc_auc_score(all_labels, all_scores) * 100
    except:
        val_auc = 0.0
    
    # Calculate real/fake accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    real_mask = (all_labels == 0)
    fake_mask = (all_labels == 1)
    
    if real_mask.sum() > 0:
        real_acc = accuracy_score(all_labels[real_mask], all_preds[real_mask]) * 100
    else:
        real_acc = 0.0
        
    if fake_mask.sum() > 0:
        fake_acc = accuracy_score(all_labels[fake_mask], all_preds[fake_mask]) * 100
    else:
        fake_acc = 0.0
    
    return {
        'loss': val_loss,
        'acc': val_acc,
        'ap': val_ap,
        'auc': val_auc,
        'real_acc': real_acc,
        'fake_acc': fake_acc
    }


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training Frequency Branch - Hybrid NPR Detector")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Spatial model: {args.spatial_model_path}")
    print(f"Dataset: {args.dataroot}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*70}\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'checkpoints/hybrid_frequency_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Logger tees stdout to the log file, so plain print() is logged
    Logger(os.path.join(output_dir, 'train.log'))
    print(f"Training started at {datetime.now()}")
    print(f"Arguments: {args}")
    
    # Load model
    print("Loading Hybrid NPR Detector...")
    model = HybridNPRDetector(
        spatial_model_path=args.spatial_model_path,
        feature_dim=args.feature_dim
    )
    model = model.to(device)
    
    # Print parameter statistics
    params_info = model.get_trainable_params()
    print(f"\nParameter Statistics:")
    print(f"  Total: {params_info['total']:,}")
    print(f"  Trainable: {params_info['trainable']:,}")
    print(f"  Frozen: {params_info['frozen']:,}")
    print(f"  Trainable ratio: {100*params_info['trainable']/params_info['total']:.2f}%\n")
    
    print(f"Model parameters: {params_info}")

    # Data loaders — create_dataloader(opt) expects an options object with the
    # same fields the training pipeline uses (see options/base_options.py)
    from argparse import Namespace

    def make_data_opt(root, is_train):
        return Namespace(
            dataroot=root, classes=args.classes, mode='binary', isTrain=is_train,
            no_crop=False, no_flip=not is_train, no_resize=False,
            cropSize=224, loadSize=256, class_bal=False, serial_batches=not is_train,
            batch_size=args.batch_size, num_threads=args.num_workers,
            rz_interp=['bilinear'], blur_prob=0.0, blur_sig=[0.5],
            jpg_prob=0.0, jpg_method=['cv2'], jpg_qual=[75],
        )

    print("Loading data...")
    train_loader = create_dataloader(make_data_opt(os.path.join(args.dataroot, 'train'), True))
    val_loader = create_dataloader(make_data_opt(os.path.join(args.dataroot, 'val'), False))
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}\n")
    
    # Optimizer - chỉ optimize trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed precision scaler (only meaningful on CUDA)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    training_history = []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Learning rate decay
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.2f}%")
        print(f"  Val AP: {val_metrics['ap']:.2f}% | Val AUC: {val_metrics['auc']:.2f}%")
        print(f"  Real Acc: {val_metrics['real_acc']:.2f}% | Fake Acc: {val_metrics['fake_acc']:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Log
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train: {train_loss:.4f}/{train_acc:.2f}% - "
              f"Val: {val_metrics['loss']:.4f}/{val_metrics['acc']:.2f}%/{val_metrics['ap']:.2f}%")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_ap': val_metrics['ap'],
            'val_auc': val_metrics['auc'],
            'real_acc': val_metrics['real_acc'],
            'fake_acc': val_metrics['fake_acc'],
            'lr': current_lr
        })
        
        # Save best model
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            best_epoch = epoch + 1
            
            save_path = os.path.join(output_dir, 'model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"\n  ✓ Saved best model (acc={val_metrics['acc']:.2f}%)")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(output_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved checkpoint: epoch_{epoch+1}")
    
    # Training completed
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print(f"\n{'='*70}")
    print(f"Training Completed!")
    print(f"{'='*70}")
    print(f"Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"Training Time: {hours}h {minutes}m")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*70}\n")
    
    print(f"Training completed. Best acc: {best_acc:.2f}% at epoch {best_epoch}")
    print(f"Total time: {hours}h {minutes}m")
    
    # Save training history
    import csv
    history_path = os.path.join(output_dir, 'training_history.csv')
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=training_history[0].keys())
        writer.writeheader()
        writer.writerows(training_history)
    
    print(f"Training history saved to: {history_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Frequency Branch for Hybrid NPR Detector')
    
    # Model
    parser.add_argument('--spatial_model_path', type=str, 
                        default='trained_model_forensynths_100acc/model_epoch_last.pth',
                        help='Path to pretrained spatial model')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Dimension of frequency features')
    
    # Data
    parser.add_argument('--dataroot', type=str, default='dataset/ForenSynths',
                        help='Path to dataset root')
    parser.add_argument('--classes', nargs='+', default=['car', 'cat', 'chair', 'horse'],
                        help='Classes to use')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)
