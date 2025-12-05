# Phát hiện Deepfake bằng NPR — Bản cải tiến

<p align="center">
	Beijing Jiaotong University, YanShan University, A*Star
</p>

<img src="./NPR.png" width="100%" alt="Sơ đồ tổng quan">

Tham chiếu tới bài báo CVPR'24: [Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection](https://arxiv.org/abs/2312.10461).
```
@misc{tan2023rethinking,
    title={Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection}, 
    author={Chuangchuang Tan and Huan Liu and Yao Zhao and Shikui Wei and Guanghua Gu and Ping Liu and Yunchao Wei},
    year={2023},
    eprint={2312.10461},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Tin mới 🆕
- `2024/02`: NPR được chấp nhận tại CVPR 2024.
- `2024/05`: [Demo trực tuyến](https://huggingface.co/spaces/tancc/Generalizable_Deepfake_Detection-NPR-CVPR2024)

<a href="https://huggingface.co/spaces/tancc/Generalizable_Deepfake_Detection-NPR-CVPR2024"><img src="assets/demo_detection.gif" width="70%"></a>

## Bản cải tiến có gì hơn mẫu gốc?

Bản này bổ sung các cải tiến thực tế nhằm tăng khả năng tổng quát, ổn định huấn luyện và thân thiện Windows, vẫn giữ lõi NPR:

- Adaptive NPR: phần dư NPR đa tỉ lệ có thể học (`--adaptive_npr`).
- Tinh chỉnh nhánh tần số: ổn định trộn đặc trưng trong `networks/frequency_branch.py`.
- Nâng cấp huấn luyện: mixed precision (`--use_amp`), label smoothing (`--label_smoothing`), cosine LR (`--cosine_lr`).
- Chế độ GenImage: transform chuẩn và seed=70 để so sánh công bằng (`--genimage_mode`).
- Script đánh giá mạnh: `evaluate.py`, `evaluate_all_datasets.py`, `evaluate_genimage.py` xuất CSV (Acc, ROC-AUC, PR-AUC, FPR, chênh lệch tổng quát).
- Tài liệu và lệnh PowerShell cho Windows; xử lý đường dẫn dài, script tải dữ liệu.

## Kết quả tổng hợp (từ TOTAL_RESULTS.csv)

- ForenSynths Val:
  - Baseline (NPR.pth): Accuracy 46.70% (model gốc pretrained).
  - Sau khi train 30 epoch: Accuracy 100.00%, AP 100.00%, ROC-AUC 100.00%.
- UniversalFakeDetect (cross-dataset): Accuracy 91.84%, AP 97.36%, ROC-AUC 97.53%, Real_Acc 94.01%, Fake_Acc 89.68% (16,000 mẫu) → tổng quát tốt.
- GANGen-Detection (cross-dataset): Accuracy 64.37%, AP 73.61%, ROC-AUC 78.54%, Real_Acc 32.96%, Fake_Acc 95.77% (36,000 mẫu) → phát hiện fake mạnh, real còn yếu.
- 8-GAN (Epoch 29): ProGAN 99.60%, StyleGAN 96.20%, StyleGAN2 97.20%, BigGAN 84.20%, CycleGAN 85.90%, StarGAN 99.70%, GauGAN 80.20%, DeepFake 68.10% → trung bình 88.90%.
- Mean 3 bộ chính (ForenSynths + UniversalFD + GANGen): 85.40%.
- Cải thiện so với baseline ForenSynths: +53.30% (46.7% → 100.0%), vượt mục tiêu 72–75% (+25–28%).

## Khởi chạy nhanh (Windows PowerShell)

1) Tạo môi trường và cài đặt phụ thuộc

```powershell
python -m venv .venv
 .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2 torchvision==0.17.2 --prefer-binary

# Kiểm tra CUDA
python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
else:
    print('CUDA chưa bật: hãy cập nhật driver NVIDIA (R530+ cho cu121).')
PY
```

2) Tải dữ liệu

```powershell
pwsh .\download_dataset.ps1
```

3) Kiểm thử nhanh (CPU/GPU)

```powershell
python .\test.py --model_path .\NPR.pth --batch_size 8 --fast --max_dirs 1 --max_images 64
```

4) Đánh giá đầy đủ

```powershell
python .\test.py --model_path .\NPR.pth --batch_size 64
# Hoặc quét tất cả bộ chuẩn và xuất CSV tổng hợp
python .\evaluate.py --model_path .\NPR.pth --batch_size 32 --out_csv .\results\evaluation_details.csv
```

## Dữ liệu cần thiết
<!-- 
Download dataset from [CNNDetection CVPR2020 (Table1 results)](https://github.com/peterwang512/CNNDetection), [GANGen-Detection (Table2 results)](https://github.com/chuangchuangtan/GANGen-Detection) ([googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing)), [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect) ([googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=drive_link)), [DIRE 2023ICCV](https://github.com/ZhendongWang6/DIRE) ([googledrive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf?usp=sharing)), Diffusion1kStep [googledrive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq?usp=sharing).
-->
|                        | paper  | Url  |
|:----------------------:|:-----:|:-----:|
| Train set              | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)                   | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                 | 
| Val   set              | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)                   | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                 | 
| Table1 Test            | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)                   | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                 | 
| Table2 Test            | [FreqNet AAAI2024](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)        | [googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing)   | 
| Table3 Test            | [DIRE ICCV2023](https://github.com/ZhendongWang6/DIRE)                                  | [googledrive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf?usp=sharing)   | 
| Table4 Test            | [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect)        | [googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=sharing)| 
| Table5 Test            | Diffusion1kStep                                                                         | [googledrive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq?usp=sharing)   | 

```
pip install gdown==4.7.1

chmod 777 ./download_dataset.sh

./download_dataset.sh
```
## Directory structure
<details>
<summary> Click to expand the folder tree structure. </summary>

```
datasets
|-- ForenSynths_train_val
|   |-- train
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   `-- horse
|   `-- val
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   `-- horse
|   |-- test
|       |-- biggan
|       |-- cyclegan
|       |-- deepfake
|       |-- gaugan
|       |-- progan
|       |-- stargan
|       |-- stylegan
|       `-- stylegan2
`-- Generalization_Test
    |-- ForenSynths_test       # Table1
    |   |-- biggan
    |   |-- cyclegan
    |   |-- deepfake
    |   |-- gaugan
    |   |-- progan
    |   |-- stargan
    |   |-- stylegan
    |   `-- stylegan2
    |-- GANGen-Detection     # Table2
    |   |-- AttGAN
    |   |-- BEGAN
    |   |-- CramerGAN
    |   |-- InfoMaxGAN
    |   |-- MMDGAN
    |   |-- RelGAN
    |   |-- S3GAN
    |   |-- SNGAN
    |   `-- STGAN
    |-- DiffusionForensics  # Table3
    |   |-- adm
    |   |-- ddpm
    |   |-- iddpm
    |   |-- ldm
    |   |-- pndm
    |   |-- sdv1_new
    |   |-- sdv2
    |   `-- vqdiffusion
    `-- UniversalFakeDetect # Table4
    |   |-- dalle
    |   |-- glide_100_10
    |   |-- glide_100_27
    |   |-- glide_50_27
    |   |-- guided          # Also known as ADM.
    |   |-- ldm_100
    |   |-- ldm_200
    |   `-- ldm_200_cfg
    |-- Diffusion1kStep     # Table5
        |-- DALLE
        |-- ddpm
        |-- guided-diffusion    # Also known as ADM.
        |-- improved-diffusion  # Also known as IDDPM.
        `-- midjourney


```
</details>

## Huấn luyện mô hình 

Huấn luyện NPR cơ bản (Linux)
```sh
CUDA_VISIBLE_DEVICES=0 ./pytorch18/bin/python train.py --name 4class-resnet-car-cat-chair-horse --dataroot ./datasets/ForenSynths_train_val --classes car,cat,chair,horse --batch_size 32 --delr_freq 10 --lr 0.0002 --niter 50
```

Huấn luyện cải tiến trên Windows (Adaptive NPR, AMP, smoothing, cosine LR)
```powershell
python .\train.py --name npr_adaptive --dataroot .\datasets\ForenSynths_train_val \
    --classes car,cat,chair,horse --batch_size 32 --lr 2e-4 --niter 50 \
    --adaptive_npr --use_amp --label_smoothing 0.05 --cosine_lr
```

## Kiểm thử bộ phát hiện
Có thể chỉnh `dataroot` trong `test.py` nếu cần.
```powershell
python .\test.py --model_path .\NPR.pth --batch_size 32
```

## Đánh giá (Accuracy, AUC, FPR, Tổng quát hóa)

Sinh CSV chi tiết gồm Acc, ROC-AUC, PR-AUC, FPR và chênh lệch tổng quát (so với split nội miền tùy chọn):

```powershell
python .\evaluate.py --model_path .\NPR.pth --batch_size 32 --out_csv .\results\evaluation_details.csv

# Optional in-domain reference to compute generalization gap
python .\evaluate.py --model_path .\NPR.pth --batch_size 32 --in_domain_root .\dataset\ForenSynths\test
```

Script sẽ quét các benchmark đã biết và ghi `results/evaluation_details.csv`.

### Windows quick start (PowerShell)

Tái hiện đầy đủ các bảng:

```powershell
# ForenSynths official test split by default
python .\test.py --model_path .\NPR.pth --batch_size 64

# Consolidated CSV across tables
python .\run_all_tests.py --model_path .\NPR.pth --batch_size 64
```

### Tùy chọn: Bật Adaptive NPR và nâng cấp huấn luyện

Bật các thành phần cải tiến để tổng quát tốt hơn:
- AdaptiveNPR (`--adaptive_npr`)
- Mixed precision (`--use_amp`)
- Label smoothing (`--label_smoothing 0.05`)
- Cosine LR (`--cosine_lr`)

## Đánh giá GenImage

For fair comparison with GenImage numbers, use the provided script which applies the required transform (translate & duplicate), sets random seed=70, and evaluates the official subsets:

```powershell
# Replace <GENIMAGE_ROOT> with the path containing subset folders: ADM, biggan, glide, midjourney, sdv5, vqdm, wukong.
# Each subset should have 'train' and 'val' with class folders 'ai' and 'nature'. The script evaluates the 'val' split.
python .\evaluate_genimage.py --genimage_root <GENIMAGE_ROOT> --model_path .\NPR.pth --batch_size 32 --out_csv .\results\results_genimage.csv
```

This will print per-subset Acc and A.P. and write a CSV to `results/results_genimage.csv`.

Ghi chú:
- Nếu `torch.cuda.is_available()` là False sau khi cài cu121, hãy cập nhật driver NVIDIA (GeForce Experience hoặc trang driver). Khởi động lại Windows và kiểm tra lại.
- Với thiết lập riêng của GenImage và AIGCDetectBenchmark, xem các ghi chú bên dưới.

### Huấn luyện GenImage (sdv4 → cross-subset)

To reproduce the GenImage fine-tuning scenario (train on sdv4, evaluate on other subsets), train with GenImage-compatible settings and seed=70. On Windows PowerShell:

```powershell
# Train on sdv4 (ai/nature) with GenImage transform and seed=70.
# <GENIMAGE_ROOT> contains subset folders such as "Stable Diffusion V1.4" (sdv4), "sdv5", ADM, etc.
# We set classes to sdv4 so training reads <GENIMAGE_ROOT>\train\sdv4\{ai,nature}.
python .\train.py \
    --name genimg_sdv4_ft \
    --dataroot <GENIMAGE_ROOT> \
    --classes sdv4 \
    --train_split train --val_split val \
    --batch_size 32 --lr 2e-4 --niter 50 \
    --genimage_mode --seed 70 \
    --use_amp --label_smoothing 0.05 --cosine_lr \
    --pretrained_path .\NPR.pth \
    --skip_bench_eval

# After training, evaluate across GenImage subsets using the last checkpoint.
# Replace <CKPT> with the path printed by training, e.g., .\checkpoints\genimg_sdv4_ft_YYYY_MM_DD_HH_MM_SS\model_epoch_last.pth
python .\evaluate_genimage.py \
    --genimage_root <GENIMAGE_ROOT> \
    --model_path <CKPT> \
    --batch_size 32 \
    --out_csv .\results\results_genimage_ft.csv
```

Mẹo:
- Dùng `--skip_bench_eval` khi train GenImage để bỏ vòng đánh giá ForenSynths tích hợp.
- Nếu chưa có tensorboardX, code sẽ dùng `torch.utils.tensorboard` (cài TensorBoard bằng `pip install tensorboard`).

### So sánh với kết quả bên ngoài mà không cần chạy lại

If you already have an external results CSV (e.g., exported from your Google Drive sheet) with columns `subset,acc,ap`, you can compare it with our local CSV without re-running evaluation:

```powershell
python .\tools\compare_results.py --ours .\results\results_genimage.csv --baseline <PATH_TO_BASELINE_CSV> --out .\results\results_comparison.csv
```

Ghi chú:
- Giá trị có thể ở dạng 0–1 (sẽ quy đổi thành %) hoặc 0–100 (%). Cột cần có `subset,acc,ap`.
- Script xuất chênh lệch theo từng tập và dòng trung bình.

## Kết quả phát hiện

### [AIGCDetectBenchmark](https://drive.google.com/drive/folders/1p4ewuAo7d5LbNJ4cKyh10Xl9Fg2yoFOw) dùng [checkpoint ProGAN-4class](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/main/model_epoch_last_3090.pth)

Khi test AIGCDetectBenchmark, đặt `no_resize` và `no_crop` = True, `batch_size` = 1.
Để xử lý ảnh kích thước lẻ, thêm đoạn sau trong [network/resnet.py](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/e2dbbe673c69c0c7237726e809a725a0308ec43d/networks/resnet.py#L163):
```
n,c,w,h = x.shape
if w%2 == 1 : x = x[:,:,:-1,:]
if h%2 == 1 : x = x[:,:,:,:-1]
```

| Generator   |  CNNSpot | FreDect |   Fusing  | GramNet |   LNP   |  LGrad  |  DIRE-G | DIRE-D |  UnivFD |  RPTCon | NPR  |
|  :---------:| :-----:  |:-------:| :--------:|:-------:|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:----:|
| ProGAN      |  100.00  |  99.36  |   100.00  |  99.99  |  99.67  |  99.83  |  95.19  |  52.75 |  99.81  |  100.00 | 99.9 |
| StyleGan    |  90.17   |  78.02  |   85.20   |  87.05  |  91.75  |  91.08  |  83.03  |  51.31 |  84.93  |  92.77  | 96.1 |
| BigGAN      |  71.17   |  81.97  |   77.40   |  67.33  |  77.75  |  85.62  |  70.12  |  49.70 |  95.08  |  95.80  | 87.3 |
| CycleGAN    |  87.62   |  78.77  |   87.00   |  86.07  |  84.10  |  86.94  |  74.19  |  49.58 |  98.33  |  70.17  | 90.3 |
| StarGAN     |  94.60   |  94.62  |   97.00   |  95.05  |  99.92  |  99.27  |  95.47  |  46.72 |  95.75  |  99.97  | 99.6 |
| GauGAN      |  81.42   |  80.57  |   77.00   |  69.35  |  75.39  |  78.46  |  67.79  |  51.23 |  99.47  |  71.58  | 85.4 |
| Stylegan2   |  86.91   |  66.19  |   83.30   |  87.28  |  94.64  |  85.32  |  75.31  |  51.72 |  74.96  |  89.55  | 98.1 |
| WFIR        |  91.65   |  50.75  |   66.80   |  86.80  |  70.85  |  55.70  |  58.05  |  53.30 |  86.90  |  85.80  | 60.7 |
| ADM         |  60.39   |  63.42  |   49.00   |  58.61  |  84.73  |  67.15  |  75.78  |  98.25 |  66.87  |  82.17  | 84.9 |
| Glide       |  58.07   |  54.13  |   57.20   |  54.50  |  80.52  |  66.11  |  71.75  |  92.42 |  62.46  |  83.79  | 96.7 |
| Midjourney  |  51.39   |  45.87  |   52.20   |  50.02  |  65.55  |  65.35  |  58.01  |  89.45 |  56.13  |  90.12  | 92.6 |
| SDv1.4      |  50.57   |  38.79  |   51.00   |  51.70  |  85.55  |  63.02  |  49.74  |  91.24 |  63.66  |  95.38  | 97.4 |
| SDv1.5      |  50.53   |  39.21  |   51.40   |  52.16  |  85.67  |  63.67  |  49.83  |  91.63 |  63.49  |  95.30  | 97.5 |
| VQDM        |  56.46   |  77.80  |   55.10   |  52.86  |  74.46  |  72.99  |  53.68  |  91.90 |  85.31  |  88.91  | 90.1 |
| Wukong      |  51.03   |  40.30  |   51.70   |  50.76  |  82.06  |  59.55  |  54.46  |  90.90 |  70.93  |  91.07  | 91.7 |
| DALLE2      |  50.45   |  34.70  |   52.80   |  49.25  |  88.75  |  65.45  |  66.48  |  92.45 |  50.75  |  96.60  | 99.6 |
| Average     |  70.78   |  64.03  |   68.38   |  68.67  |  83.84  |  75.34  |  68.68  |  71.53 |  78.43  |  89.31  | **91.7** |

### [GenImage](https://github.com/GenImage-Dataset/GenImage)

<details>
<summary> (1) Change "resize" to "translate and duplicate". (2) Set random seed to 70. (3) During testing, set no_crop to False. </summary>

(1)
```
dset = datasets.ImageFolder(
    root,
    transforms.Compose([
        # rz_func,
	transforms.Lambda(lambda img: translate_duplicate(img, opt.cropSize)),
	crop_func,
	flip_func,
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

import math
def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size
        
        new_width = width * math.ceil(cropSize/width)
        new_height = height * math.ceil(cropSize/height)
        
        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img
```
(2)
Set [random seed](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/bed4c2c9eb9b2000e9a24233d3b010afa3452f12/train.py#L48) to 70.

(3)
During testing, set [no_crop](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/bed4c2c9eb9b2000e9a24233d3b010afa3452f12/train.py#L69) to False. And set [test config](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/bed4c2c9eb9b2000e9a24233d3b010afa3452f12/train.py#L30)
```
vals =       ['ADM', 'biggan', 'glide', 'midjourney', 'sdv5', 'vqdm', 'wukong']
multiclass = [ 0,     0,        0,       0,            0,      0,      0      ]
```
</details>

```
./pytorch18/bin/python  train.py --dataroot {GenImage Path} --name sdv4_bs32_ --batch_size 32 --lr 0.0002 --niter 1  --cropSize 224 --classes sdv4 
```

Train with sdv4 as the training set, using a random seed of 70. [Pretrained checkpoint](https://drive.google.com/drive/folders/1_mD17F94xMbJqEAsWRW1gVsZ5db6YamI?usp=sharing).

|Generator   | Acc. | A.P. |
|:----------:|:----:|:----:|
| ADM        | 87.8 | 96.0 |
| biggan     | 80.7 | 89.8 |
| glide      | 93.2 | 99.1 |
| midjourney | 91.7 | 97.9 |
| sdv5       | 94.4 | 99.9 |
| vqdm       | 88.7 | 96.1 |
| wukong     | 94.0 | 99.7 |
| Mean       | 90.1 | 96.9 |

<!--
| <font size=2>Method</font>|<font size=2>ProGAN</font> |       |<font size=2>StyleGAN</font>|     |<font size=2>StyleGAN2</font>|    |<font size=2>BigGAN</font>|       |<font size=2>CycleGAN</font> |      |<font size=2>StarGAN</font>|       |<font size=2>GauGAN</font> |       |<font size=2>Deepfake</font>|    | <font size=2>Mean</font> |      |
|:----------------------:|:-----:|:-----:|:------:|:---:|:-------:|:--:|:----:|:-----:|:-------:|:----:|:----: |:-----:|:---:  |:-----:|:----:|:----:|:----:|:----:|
|                        | Acc.  | A.P.  | Acc.   | A.P.| Acc.  | A.P. | Acc.| A.P.   | Acc.    | A.P. | Acc.  | A.P.  | Acc.  | A.P.  | Acc. | A.P. | Acc. | A.P. |
| CNNDetection           | 91.4  | 99.4  | 63.8   | 91.4| 76.4  | 97.5 | 52.9| 73.3   | 72.7    | 88.6 | 63.8  | 90.8  | 63.9  | 92.2  | 51.7 | 62.3 | 67.1 | 86.9 |
| Frank                  | 90.3  | 85.2  | 74.5   | 72.0| 73.1  | 71.4 | 88.7| 86.0   | 75.5    | 71.2 | 99.5  | 99.5  | 69.2  | 77.4  | 60.7 | 49.1 | 78.9 | 76.5 |
| Durall                 | 81.1  | 74.4  | 54.4   | 52.6| 66.8  | 62.0 | 60.1| 56.3   | 69.0    | 64.0 | 98.1  | 98.1  | 61.9  | 57.4  | 50.2 | 50.0 | 67.7 | 64.4 |
| Patchfor               | 97.8  | 100.0 | 82.6   | 93.1| 83.6  | 98.5 | 64.7| 69.5   | 74.5    | 87.2 | 100.0 | 100.0 | 57.2  | 55.4  | 85.0 | 93.2 | 80.7 | 87.1 |
| F3Net                  | 99.4  | 100.0 | 92.6   | 99.7| 88.0  | 99.8 | 65.3| 69.9   | 76.4    | 84.3 | 100.0 | 100.0 | 58.1  | 56.7  | 63.5 | 78.8 | 80.4 | 86.2 |
| SelfBland              | 58.8  | 65.2  | 50.1   | 47.7| 48.6  | 47.4 | 51.1| 51.9   | 59.2    | 65.3 | 74.5  | 89.2  | 59.2  | 65.5  | 93.8 | 99.3 | 61.9 | 66.4 |
| GANDetection           | 82.7  | 95.1  | 74.4   | 92.9| 69.9  | 87.9 | 76.3| 89.9   | 85.2    | 95.5 | 68.8  | 99.7  | 61.4  | 75.8  | 60.0 | 83.9 | 72.3 | 90.1 |
| BiHPF                  | 90.7  | 86.2  | 76.9   | 75.1| 76.2  | 74.7 | 84.9| 81.7   | 81.9    | 78.9 | 94.4  | 94.4  | 69.5  | 78.1  | 54.4 | 54.6 | 78.6 | 77.9 |
| FrePGAN                | 99.0  | 99.9  | 80.7   | 89.6| 84.1  | 98.6 | 69.2| 71.1   | 71.1    | 74.4 | 99.9  | 100.0 | 60.3  | 71.7  | 70.9 | 91.9 | 79.4 | 87.2 |
| LGrad                  | 99.9  | 100.0 | 94.8   | 99.9| 96.0  | 99.9 | 82.9| 90.7   | 85.3    | 94.0 | 99.6  | 100.0 | 72.4  | 79.3  | 58.0 | 67.9 | 86.1 | 91.5 |
| Ojha                   | 99.7  | 100.0 | 89.0   | 98.7| 83.9  | 98.4 | 90.5| 99.1   | 87.9    | 99.8 | 91.4  | 100.0 | 89.9  | 100.0 | 80.2 | 90.2 | 89.1 | 98.3 |
| NPR(our)               | 99.8  | 100.0 | 96.3   | 99.8| 97.3  | 100.0| 87.5| 94.5   | 95.0    | 99.5 | 99.7  | 100.0 | 86.6  | 88.8  | 77.4 | 86.2 | 92.5 | 96.1 |
-->

## Ghi nhận

Kho mã này tham khảo một phần từ [CNNDetection](https://github.com/PeterWang512/CNNDetection) (Wang et al., CVPR 2020).

- Giấy phép: CNNDetection phát hành theo CC BY-NC-SA 4.0. Các phần kế thừa giữ nguyên nghĩa vụ phi thương mại và chia sẻ tương tự.
- Nếu bạn xây dựng tiếp trên công việc này, vui lòng trích dẫn CNNDetection:

```
@inproceedings{wang2019cnngenerated,
    title={CNN-generated images are surprisingly easy to spot...for now},
    author={Wang, Sheng-Yu and Wang, Oliver and Zhang, Richard and Owens, Andrew and Efros, Alexei A},
    booktitle={CVPR},
    year={2020}
}
```

Xem mục Third-Party Notices để biết chi tiết giấy phép và liên kết.
