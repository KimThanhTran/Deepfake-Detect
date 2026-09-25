"""
Score-level fusion of two eval_report.py runs saved with --save_scores.
Fused fake probability = w * score_a + (1 - w) * score_b, then the same metrics
as eval_report.py (per subset + Mean, metrics.csv + confusion_matrices.png).

Usage:
    python tools/fuse_scores.py --a results/scores/NPR_GANGen-Detection \
        --b results/scores/npr_adaptive_aug_GANGen-Detection \
        --out_dir results/fusion_GANGen-Detection --weight 0.5
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_report import metrics_row, write_report


def load_scores(run_dir):
    scores = {}
    with open(os.path.join(run_dir, 'scores.csv')) as f:
        for r in csv.DictReader(f):
            scores[(r['subset'], r['relpath'])] = (int(r['label']), float(r['score']))
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True, help='run dir containing scores.csv (weight w)')
    p.add_argument('--b', required=True, help='run dir containing scores.csv (weight 1-w)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--weight', type=float, default=0.5)
    p.add_argument('--label', default=None)
    args = p.parse_args()

    a, b = load_scores(args.a), load_scores(args.b)
    if a.keys() != b.keys():
        raise SystemExit(f'Image sets differ: {len(a.keys() - b.keys())} only in a, '
                         f'{len(b.keys() - a.keys())} only in b')

    by_subset = defaultdict(lambda: ([], []))
    for key in sorted(a):
        (label_a, sa), (label_b, sb) = a[key], b[key]
        assert label_a == label_b, key
        y_true, y_score = by_subset[key[0]]
        y_true.append(label_a)
        y_score.append(args.weight * sa + (1 - args.weight) * sb)

    rows = {}
    for subset in sorted(by_subset):
        y_true, y_score = by_subset[subset]
        r = metrics_row(np.array(y_true), np.array(y_score))
        rows[subset] = r
        print(f'{subset:12s} acc={r["acc"]*100:6.2f}%  real={r["real_acc"]*100:6.2f}%  '
              f'fake={r["fake_acc"]*100:6.2f}%  AUC={r["auc"]*100:6.2f}%')

    write_report(rows, args.out_dir, args.label or f'fusion w={args.weight}')


if __name__ == '__main__':
    main()
