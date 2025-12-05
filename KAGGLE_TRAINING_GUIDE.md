# Hướng Dẫn Train Frequency Branch Trên Kaggle

## 📋 Tổng Quan

Train frequency branch trên Kaggle GPU miễn phí thay vì máy local (vì máy local thiếu PyTorch CUDA).

**Thời gian:** 
- Setup: ~10 phút
- Training: 1.5-3 giờ (5-10 epochs)
- Total: ~2-3 giờ

---

## 🚀 Bước 1: Chuẩn Bị Dataset (5 phút)

### Option A: Sử dụng Dataset Public Trên Kaggle

1. Tìm ForenSynths dataset trên Kaggle:
   - Link: https://www.kaggle.com/datasets/search?q=forensynths
   - Hoặc search "deepfake forensynths"

### Option B: Upload Dataset Của Bạn (Khuyến nghị)

1. **Tạo Kaggle Dataset:**
   - Vào: https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Title: `forensynths-deepfake-dataset`

2. **Upload data:**
   ```
   Folder structure cần upload:
   forensynths/
   ├── train/
   │   ├── 0_real/      (ảnh thật)
   │   └── 1_fake/      (ảnh fake)
   └── val/
       ├── 0_real/
       └── 1_fake/
   ```

3. **Zip và upload:**
   - Zip folder `forensynths` trên máy bạn
   - Upload lên Kaggle
   - Set visibility: Private hoặc Public
   - Click "Create"

---

## 🚀 Bước 2: Upload Model Đã Train (2 phút)

Bạn đã có model trained 100% accuracy, cần upload lên Kaggle.

1. **Tạo Kaggle Dataset cho model:**
   - Vào: https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Title: `npr-deepfake-trained-model`

2. **Upload model file:**
   ```
   File cần upload:
   - trained_model_forensynths_100acc/model_epoch_last.pth
   ```

3. **Upload:**
   - Kéo thả file `model_epoch_last.pth` vào
   - Click "Create"

---

## 🚀 Bước 3: Tạo Kaggle Notebook (2 phút)

1. **Tạo notebook mới:**
   - Vào: https://www.kaggle.com/code
   - Click "New Notebook"

2. **Enable GPU:**
   - Bên phải sidebar → "Accelerator"
   - Chọn **GPU T4** hoặc **GPU P100**
   - Lưu ý: GPU có giới hạn 30 giờ/tuần

3. **Upload code files:**
   Click "File" → "Upload Files", upload các files này:
   ```
   - train_frequency.py
   - evaluate_hybrid.py
   - visualize_frequency.py
   - networks/frequency_branch.py
   - networks/__init__.py
   - networks/base_model.py
   - networks/resnet.py
   - data/__init__.py
   - data/datasets.py
   - options/__init__.py
   - options/base_options.py
   - options/train_options.py
   - util.py
   ```

   **Hoặc:** Copy toàn bộ folder structure lên (nhanh hơn)

4. **Add Data:**
   - Click "Add Data" (bên phải)
   - Search datasets bạn vừa tạo:
     - `forensynths-deepfake-dataset`
     - `npr-deepfake-trained-model`
   - Click "Add" cho cả 2

---

## 🚀 Bước 4: Copy Notebook Code (1 phút)

Tôi đã tạo sẵn notebook: `kaggle_frequency_train.ipynb`

**Cách 1: Upload notebook**
- Upload file `kaggle_frequency_train.ipynb` vào Kaggle
- Kaggle sẽ tự động parse và tạo cells

**Cách 2: Copy từng cell**
- Mở file `kaggle_frequency_train.ipynb` bằng text editor
- Copy code từng cell vào Kaggle notebook

---

## 🚀 Bước 5: Chỉnh Sửa Paths (1 phút)

Trong notebook, sửa 2 paths này:

### Cell "Import Dataset":
```python
# Sửa path này theo dataset name của bạn
DATASET_PATH = "/kaggle/input/forensynths-deepfake-dataset"
```

### Cell "Upload Trained Model":
```python
# Sửa path này theo dataset name của bạn
MODEL_PATH = "/kaggle/input/npr-deepfake-trained-model/model_epoch_last.pth"
```

**Check paths:**
- Run cell `!ls /kaggle/input/` để xem exact names

---

## 🚀 Bước 6: Run Training! (1.5-3 giờ)

1. **Chạy từng cell từ trên xuống:**
   - Cell 1: Check GPU ✅
   - Cell 2: Install dependencies ✅
   - Cell 3: Check dataset ✅
   - Cell 4: Check model ✅
   - Cell 5: **START TRAINING** ⏳

2. **Chọn training time:**
   - **5 epochs** (~1.5 giờ): Đủ tốt, nhanh
   - **10 epochs** (~3 giờ): Tốt nhất

3. **Giám sát training:**
   - Xem output real-time
   - Check accuracy mỗi epoch
   - Nếu accuracy > 98%, đã tốt rồi

---

## 🚀 Bước 7: Download Results (2 phút)

Sau khi training xong:

1. **Run cell "Check Training Results":**
   - Xem learning curves
   - Best accuracy

2. **Download files:**
   - Click "Output" tab (phía dưới)
   - Download:
     ```
     ✅ hybrid_model_best.pth (model trained) - ~100 MB
     ✅ training_history.csv (metrics log)
     ✅ training_curves.png (visualization)
     ```

3. **Lưu về máy:**
   - Copy vào folder `NPR-DeepfakeDetection/frequency_checkpoints/`

---

## 📊 Bước 8: Evaluate Trên Máy Local (30 phút)

Sau khi download model về:

```bash
# Evaluate trên ForenSynths
python evaluate_hybrid.py \
    --model_path frequency_checkpoints/hybrid_model_best.pth \
    --dataroot dataset/ForenSynths \
    --batch_size 16

# Evaluate trên UniversalFakeDetect
python evaluate_hybrid.py \
    --model_path frequency_checkpoints/hybrid_model_best.pth \
    --dataroot dataset/UniversalFakeDetect \
    --batch_size 16

# Evaluate trên GANGen-Detection
python evaluate_hybrid.py \
    --model_path frequency_checkpoints/hybrid_model_best.pth \
    --dataroot dataset/GANGen-Detection \
    --batch_size 16
```

**Lưu ý:** Evaluation chỉ cần CPU, không cần GPU!

---

## ✅ Expected Results

Sau khi training và evaluate xong:

```
Dataset              | Baseline | Hybrid | Improvement
---------------------|----------|--------|------------
ForenSynths          | 100.0%   | 100.0% | 0%
UniversalFakeDetect  | 91.8%    | 94-95% | +2-3%
GANGen-Detection     | 64.4%    | 72-76% | +8-12%
---------------------|----------|--------|------------
Overall Mean         | 85.4%    | 89-90% | +4-6%
```

**Key improvement:**
- GANGen Real Accuracy: 33% → 55-65% (+22-32%) 🎯

---

## 🐛 Troubleshooting

### 1. "Dataset not found"
```python
# Check available datasets
!ls /kaggle/input/

# Sửa DATASET_PATH theo tên chính xác
```

### 2. "CUDA Out of Memory"
```python
# Giảm batch_size
--batch_size 16  # thay vì 32
```

### 3. "Model not loading"
```python
# Check model file size
!ls -lh /kaggle/input/npr-deepfake-trained-model/

# File phải ~100 MB
```

### 4. Training quá lâu
```python
# Giảm epochs xuống 5
--epochs 5

# Hoặc train trên subset
--classes car cat  # chỉ 2 classes
```

### 5. Kaggle GPU hết quota
- Free tier: 30 giờ/tuần
- Nếu hết: Đợi tuần sau, hoặc upgrade account

---

## 📁 Files Cần Upload Lên Kaggle

### Minimum (Essential):
```
train_frequency.py          (12 KB)
networks/frequency_branch.py (11 KB)
networks/resnet.py          (from existing)
networks/base_model.py      (from existing)
data/datasets.py            (from existing)
options/train_options.py    (from existing)
util.py                     (from existing)
```

### Recommended (Full):
```
Toàn bộ folder NPR-DeepfakeDetection/
  - Zip lại
  - Upload vào Kaggle Files
  - Extract trong notebook
```

---

## 🎯 Quick Start Commands

```bash
# 1. Tạo zip để upload
cd NPR-DeepfakeDetection
tar -czf npr_code.tar.gz train_frequency.py networks/ data/ options/ util.py

# 2. Upload npr_code.tar.gz lên Kaggle

# 3. Extract trong Kaggle notebook:
!tar -xzf /kaggle/input/npr-code/npr_code.tar.gz

# 4. Train!
!python train_frequency.py \
    --spatial_model_path /kaggle/input/npr-model/model_epoch_last.pth \
    --dataroot /kaggle/input/forensynths-dataset \
    --batch_size 32 \
    --epochs 10
```

---

## 💡 Tips

1. **Save work frequently:**
   - Kaggle auto-saves, nhưng nên click "Save Version" thủ công
   - Tránh mất code nếu session timeout

2. **Monitor GPU:**
   ```python
   import torch
   print(f"GPU Memory Used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
   ```

3. **Early stopping:**
   - Nếu accuracy > 98% ở epoch 5, có thể stop sớm
   - Không cần chờ hết 10 epochs

4. **Commit notebook:**
   - Sau khi training xong, click "Save Version"
   - Title: "Frequency Branch Training - 98% Accuracy"
   - Có thể share link trong report

---

## ✉️ Cần Hỗ Trợ?

Nếu gặp lỗi trong quá trình setup hoặc training:

1. Check Kaggle output logs
2. Copy error message
3. Hỏi lại tôi với error cụ thể

Good luck! 🚀
