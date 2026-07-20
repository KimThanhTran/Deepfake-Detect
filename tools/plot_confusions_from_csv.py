"""Regenerate confusion-matrix figures from a metrics.csv produced by eval_report.py.

Usage:
    python tools/plot_confusions_from_csv.py results/baseline_NPR/metrics.csv "NPR.pth (baseline)"
"""
import csv
import os
import sys

import numpy as np


def main(csv_path, label):
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r['subset'] == 'Mean' or not r['tn']:
                continue
            rows[r['subset']] = {k: int(r[k]) for k in ('tn', 'fp', 'fn', 'tp')} | {
                'acc': float(r['acc'])}

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
        ax.set_title(f'{subset} (acc {r["acc"]:.1f}%)', fontsize=10)
    fig.suptitle(f'Confusion matrices — {label}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(csv_path), 'confusion_matrices.png')
    fig.savefig(out, dpi=150)
    print(f'Saved {out}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '')
