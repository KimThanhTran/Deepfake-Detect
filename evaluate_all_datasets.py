"""
Comprehensive evaluation script for trained model on all available datasets
"""
import os
import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from options.test_options import TestOptions
from data import create_dataloader
import csv

print("=" * 70)
print("📊 COMPREHENSIVE EVALUATION - ALL DATASETS")
print("=" * 70)

# Model path
MODEL_PATH = "trained_model_forensynths_100acc/model_epoch_last.pth"

# All available datasets
DATASETS = [
    {"name": "ForenSynths", "path": "dataset/ForenSynths/val", "classes": ["car", "cat", "chair", "horse"]},
    {"name": "GenImage", "path": "dataset/GenImage", "classes": []},
    {"name": "DiffusionForensics", "path": "dataset/DiffusionForensics", "classes": []},
    {"name": "UniversalFakeDetect", "path": "dataset/UniversalFakeDetect", "classes": []},
    {"name": "GANGen-Detection", "path": "dataset/GANGen-Detection", "classes": []},
]

# Load model
print(f"\n🔄 Loading model: {MODEL_PATH}")
model = resnet50(num_classes=1)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
device = 'cpu'
model.to(device)
print(f"✅ Model loaded on {device}\n")

# Results storage
results = []

# Evaluate on each dataset
for dataset_info in DATASETS:
    dataset_name = dataset_info["name"]
    dataset_path = dataset_info["path"]
    classes = dataset_info["classes"]
    
    print("=" * 70)
    print(f"📂 Evaluating: {dataset_name}")
    print("=" * 70)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"⚠️  Dataset not found: {dataset_path}")
        print(f"   Skipping {dataset_name}\n")
        results.append({
            "Dataset": dataset_name,
            "Status": "Not Found",
            "Accuracy": "N/A",
            "AP": "N/A",
            "ROC-AUC": "N/A",
            "Real_Acc": "N/A",
            "Fake_Acc": "N/A",
            "Total_Samples": 0
        })
        continue
    
    try:
        # Setup options
        opt = TestOptions().parse(print_options=False)
        opt.dataroot = dataset_path
        opt.classes = classes
        opt.batch_size = 16
        opt.num_threads = 0
        opt.mode = "binary"
        
        # Load data
        data_loader = create_dataloader(opt)
        total_samples = len(data_loader.dataset)
        
        print(f"📊 Total samples: {total_samples:,}")
        print("🔍 Running inference...")
        
        # Inference
        y_true, y_pred = [], []
        with torch.no_grad():
            for i, (img, label) in enumerate(data_loader, 1):
                if i % 50 == 0:
                    print(f"   Progress: {i}/{len(data_loader)} batches", end='\r')
                
                in_tens = img.to(device)
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())
        
        print(f"   Progress: {len(data_loader)}/{len(data_loader)} batches - Done!  ")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred > 0.5)
        
        # Real/Fake accuracy
        if np.sum(y_true == 0) > 0:
            r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
        else:
            r_acc = np.nan
        
        if np.sum(y_true == 1) > 0:
            f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
        else:
            f_acc = np.nan
        
        # AP and ROC-AUC
        try:
            ap = average_precision_score(y_true, y_pred)
        except:
            ap = np.nan
        
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except:
            roc_auc = np.nan
        
        print(f"\n✅ Results:")
        print(f"   Accuracy: {acc*100:.2f}%")
        print(f"   AP: {ap*100:.2f}%")
        print(f"   ROC-AUC: {roc_auc*100:.2f}%")
        if not np.isnan(r_acc):
            print(f"   Real Accuracy: {r_acc*100:.2f}%")
        if not np.isnan(f_acc):
            print(f"   Fake Accuracy: {f_acc*100:.2f}%")
        print()
        
        results.append({
            "Dataset": dataset_name,
            "Status": "Success",
            "Accuracy": f"{acc*100:.2f}%",
            "AP": f"{ap*100:.2f}%",
            "ROC-AUC": f"{roc_auc*100:.2f}%",
            "Real_Acc": f"{r_acc*100:.2f}%" if not np.isnan(r_acc) else "N/A",
            "Fake_Acc": f"{f_acc*100:.2f}%" if not np.isnan(f_acc) else "N/A",
            "Total_Samples": total_samples
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")
        results.append({
            "Dataset": dataset_name,
            "Status": f"Error: {str(e)[:50]}",
            "Accuracy": "N/A",
            "AP": "N/A",
            "ROC-AUC": "N/A",
            "Real_Acc": "N/A",
            "Fake_Acc": "N/A",
            "Total_Samples": 0
        })

# Save results to CSV
print("=" * 70)
print("💾 SAVING RESULTS")
print("=" * 70)

csv_file = "all_datasets_evaluation.csv"
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Results saved to: {csv_file}")

# Print summary table
print("\n" + "=" * 70)
print("📊 EVALUATION SUMMARY")
print("=" * 70)
print(f"\n{'Dataset':<25} {'Accuracy':<12} {'AP':<12} {'ROC-AUC':<12} {'Samples':<10}")
print("-" * 70)
for r in results:
    print(f"{r['Dataset']:<25} {r['Accuracy']:<12} {r['AP']:<12} {r['ROC-AUC']:<12} {r['Total_Samples']:<10,}")

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETED!")
print("=" * 70)
