"""
Evaluate an NPR checkpoint on ForenSynths-style test sets and produce a full
metrics report: accuracy, real/fake accuracy, AP, ROC-AUC, and confusion
matrices (CSV + PNG figures) suitable for a paper.

Protocol matches the official NPR test.py for ForenSynths:
Resize (256, 256) bilinear, no crop, ImageNet normalization.

Usage (inside the train container):
    python tools/eval_report.py --model_path weights/NPR.pth \
        --dataroot dataset/ForenSynths --out_dir results/baseline \
        --subsets progan stylegan stylegan2 biggan cyclegan stargan gaugan deepfake
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import build_npr_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def collect_samples(subset_dir):
    """Return [(path, label)] for a subset. Handles both layouts:
    <subset>/{0_real,1_fake}/... and <subset>/<class>/{0_real,1_fake}/..."""
    samples = []

    def add_dir(d, label):
        for root, _, files in os.walk(d):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    samples.append((os.path.join(root, f), label))

    entries = sorted(os.listdir(subset_dir))
    if '0_real' in entries or '1_fake' in entries:
        class_dirs = [subset_dir]
    else:
        class_dirs = [os.path.join(subset_dir, e) for e in entries
                      if os.path.isdir(os.path.join(subset_dir, e))]
    for cd in class_dirs:
        real_d = os.path.join(cd, '0_real')
        fake_d = os.path.join(cd, '1_fake')
        if os.path.isdir(real_d):
            add_dir(real_d, 0)
        if os.path.isdir(fake_d):
            add_dir(fake_d, 1)
    return samples


class ListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def evaluate_subset(model, device, subset_dir, batch_size, num_workers):
    samples = collect_samples(subset_dir)
    if not samples:
        return None
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(ListDataset(samples, tf), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    y_true, y_score = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.sigmoid(model(imgs.to(device))).flatten().cpu().numpy()
            y_score.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        'n_real': int((y_true == 0).sum()),
        'n_fake': int((y_true == 1).sum()),
        'acc': accuracy_score(y_true, y_pred),
        'real_acc': accuracy_score(y_true[y_true == 0], y_pred[y_true == 0]) if (y_true == 0).any() else np.nan,
        'fake_acc': accuracy_score(y_true[y_true == 1], y_pred[y_true == 1]) if (y_true == 1).any() else np.nan,
        'ap': average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan,
        'auc': roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }
    return row


def plot_confusions(rows, out_path, model_label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(rows)
    cols = min(4, n)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis('off')
    for ax, (subset, r) in zip(axes, rows.items()):
        cm = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)',
                        ha='center', va='center', fontsize=9, color=color)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Real', 'Fake'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Real', 'Fake'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'{subset} (acc {r["acc"]*100:.1f}%)', fontsize=10)
    fig.suptitle(f'Confusion matrices — {model_label}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--dataroot', default='dataset/ForenSynths')
    p.add_argument('--subsets', nargs='*', default=None,
                   help='subset folder names; default: all folders in dataroot')
    p.add_argument('--out_dir', default='results/eval')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--label', default=None, help='model label used in figures')
    p.add_argument('--arch', choices=['npr', 'hybrid'], default='npr',
                   help='npr: plain NPR checkpoint; hybrid: HybridNPRDetector checkpoint')
    p.add_argument('--spatial_model_path', default='weights/NPR.pth',
                   help='spatial NPR checkpoint used to construct the hybrid model')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.arch == 'hybrid':
        from networks.frequency_branch import HybridNPRDetector
        model = HybridNPRDetector(spatial_model_path=args.spatial_model_path)
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        adaptive = False
    else:
        model, adaptive = build_npr_model(args.model_path)
    model.to(device).eval()
    label = args.label or os.path.basename(args.model_path)
    print(f'Model: {args.model_path} (adaptive_npr={adaptive}) on {device}')

    subsets = args.subsets or sorted(
        d for d in os.listdir(args.dataroot) if os.path.isdir(os.path.join(args.dataroot, d)))
    os.makedirs(args.out_dir, exist_ok=True)

    rows = {}
    for s in subsets:
        sd = os.path.join(args.dataroot, s)
        if not os.path.isdir(sd):
            print(f'[skip] {s}: not found')
            continue
        r = evaluate_subset(model, device, sd, args.batch_size, args.num_workers)
        if r is None:
            print(f'[skip] {s}: no images')
            continue
        rows[s] = r
        print(f'{s:12s} acc={r["acc"]*100:6.2f}%  real={r["real_acc"]*100:6.2f}%  '
              f'fake={r["fake_acc"]*100:6.2f}%  AP={r["ap"]*100:6.2f}%  AUC={r["auc"]*100:6.2f}%  '
              f'[TN={r["tn"]} FP={r["fp"]} FN={r["fn"]} TP={r["tp"]}]')

    if not rows:
        print('No subsets evaluated.')
        return

    # Mean row
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in ['acc', 'real_acc', 'fake_acc', 'ap', 'auc']}
    print(f'{"MEAN":12s} acc={mean["acc"]*100:6.2f}%  real={mean["real_acc"]*100:6.2f}%  '
          f'fake={mean["fake_acc"]*100:6.2f}%  AP={mean["ap"]*100:6.2f}%  AUC={mean["auc"]*100:6.2f}%')

    csv_path = os.path.join(args.out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'n_real', 'n_fake', 'acc', 'real_acc', 'fake_acc', 'ap', 'auc', 'tn', 'fp', 'fn', 'tp'])
        for s, r in rows.items():
            w.writerow([s, r['n_real'], r['n_fake'],
                        f'{r["acc"]*100:.2f}', f'{r["real_acc"]*100:.2f}', f'{r["fake_acc"]*100:.2f}',
                        f'{r["ap"]*100:.2f}', f'{r["auc"]*100:.2f}',
                        r['tn'], r['fp'], r['fn'], r['tp']])
        w.writerow(['Mean', '', '',
                    f'{mean["acc"]*100:.2f}', f'{mean["real_acc"]*100:.2f}', f'{mean["fake_acc"]*100:.2f}',
                    f'{mean["ap"]*100:.2f}', f'{mean["auc"]*100:.2f}', '', '', '', ''])
    print(f'Saved {csv_path}')

    png_path = os.path.join(args.out_dir, 'confusion_matrices.png')
    plot_confusions(rows, png_path, label)
    print(f'Saved {png_path}')


if __name__ == '__main__':
    main()
