# ⚡ Kaggle Training - Quick Reference

## 🎯 Upload Checklist

### 1. Code (kaggle_upload.zip) ✅
- Run: `.\prepare_kaggle_upload.ps1`
- Zip folder: `kaggle_upload`
- Size: ~90 KB

### 2. Trained Model
- File: `trained_model_forensynths_100acc/model_epoch_last.pth`
- Size: ~100 MB
- Upload to: https://www.kaggle.com/datasets (New Dataset)
- Name: `npr-trained-model`

### 3. Dataset ForenSynths
- Option A: Find public dataset on Kaggle
- Option B: Upload your dataset
  - Folder: `dataset/ForenSynths/`
  - Name: `forensynths-dataset`

---

## 📝 Kaggle Notebook Setup (3 minutes)

```python
# Cell 1: Extract code
!unzip /kaggle/input/npr-code/kaggle_upload.zip -d . -q
!ls

# Cell 2: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.cuda.is_available()}")

# Cell 3: Install dependencies
!pip install scikit-learn -q

# Cell 4: Set paths (EDIT THESE!)
DATASET_PATH = "/kaggle/input/forensynths-dataset"
MODEL_PATH = "/kaggle/input/npr-trained-model/model_epoch_last.pth"

# Cell 5: Train (5 epochs - 1.5 hours)
!python train_frequency.py \
    --spatial_model_path {MODEL_PATH} \
    --dataroot {DATASET_PATH} \
    --batch_size 32 \
    --epochs 5 \
    --lr 0.001 \
    --num_workers 4

# Cell 6: Check results
import pandas as pd
history = pd.read_csv('frequency_checkpoints/training_history.csv')
print(history)
print(f"\nBest Accuracy: {history['val_accuracy'].max():.2f}%")

# Cell 7: List outputs
!ls -lh frequency_checkpoints/
```

---

## 📥 Download Results

After training completes:

**Files to download:**
```
✅ frequency_checkpoints/hybrid_model_best.pth  (~100 MB)
✅ frequency_checkpoints/training_history.csv   (~1 KB)
```

**Where:** Kaggle notebook → Output tab → Download

---

## 🖥️ Local Evaluation (CPU only, no GPU needed)

```powershell
# After downloading model to frequency_checkpoints/

# Evaluate on 3 datasets
python evaluate_hybrid.py `
    --model_path frequency_checkpoints/hybrid_model_best.pth `
    --dataroot dataset/ForenSynths `
    --batch_size 16

python evaluate_hybrid.py `
    --model_path frequency_checkpoints/hybrid_model_best.pth `
    --dataroot dataset/UniversalFakeDetect `
    --batch_size 16

python evaluate_hybrid.py `
    --model_path frequency_checkpoints/hybrid_model_best.pth `
    --dataroot dataset/GANGen-Detection `
    --batch_size 16
```

---

## ⏱️ Time Estimate

| Step | Time |
|------|------|
| Upload datasets/model | 5 min |
| Setup notebook | 3 min |
| Training (5 epochs) | 1.5 hours |
| Training (10 epochs) | 3 hours |
| Download results | 2 min |
| Local evaluation | 30 min |
| **TOTAL (5 epochs)** | **~2 hours** |

---

## 🎯 Expected Results

```
Dataset              | Baseline | With Frequency | Gain
---------------------|----------|----------------|------
ForenSynths          | 100.0%   | 100.0%         | 0%
UniversalFakeDetect  | 91.8%    | 94-95%         | +2-3%
GANGen-Detection     | 64.4%    | 72-76%         | +8-12%
Mean                 | 85.4%    | 89-90%         | +4-6%
```

**Key Win:** GANGen Real Accuracy: 33% → 55-65% 🎯

---

## 🐛 Common Issues

### Dataset not found
```python
!ls /kaggle/input/  # Check exact names
DATASET_PATH = "/kaggle/input/YOUR-EXACT-NAME"
```

### CUDA OOM
```python
--batch_size 16  # Reduce from 32
```

### Training too slow
```python
--epochs 5       # Instead of 10
--num_workers 2  # Instead of 4
```

---

## 📧 Help

Stuck? Copy error message và hỏi lại tôi!
