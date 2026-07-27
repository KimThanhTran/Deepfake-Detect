"""Aggregated 2x2 confusion-matrix figure for the paper: Baseline NPR vs
AdaptiveNPR+Aug on GANGen-Detection (sums TN/FP/FN/TP over all GAN subsets).
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sum_counts(metrics_csv):
    tn = fp = fn = tp = 0
    with open(metrics_csv) as f:
        for r in csv.DictReader(f):
            if r['subset'] == 'Mean' or not r.get('tn'):
                continue
            tn += int(r['tn']); fp += int(r['fp']); fn += int(r['fn']); tp += int(r['tp'])
    return tn, fp, fn, tp


def panel(ax, counts, title):
    tn, fp, fn, tp = counts
    cm = np.array([[tn, fp], [fn, tp]], float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{int(cm[i,j])}\n({norm[i,j]*100:.1f}%)',
                    ha='center', va='center', fontsize=12,
                    color='white' if norm[i, j] > 0.5 else 'black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Thật', 'Giả'], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Thật', 'Giả'], fontsize=11)
    ax.set_xlabel('Dự đoán', fontsize=11)
    ax.set_ylabel('Thực tế', fontsize=11)
    ax.set_title(title, fontsize=12)


def main():
    base = sum_counts('results/baseline_NPR_GANGen-Detection/metrics.csv')
    impr = sum_counts('results/npr_adaptive_aug_GANGen-Detection/metrics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))
    panel(axes[0], base, '(a) NPR baseline')
    panel(axes[1], impr, '(b) Adaptive NPR + Aug')
    fig.tight_layout()
    out = sys.argv[1] if len(sys.argv) > 1 else 'results/fig_confusion_gangen.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('Saved', out)
    print('baseline TN/FP/FN/TP =', base)
    print('improved TN/FP/FN/TP =', impr)


if __name__ == '__main__':
    main()
