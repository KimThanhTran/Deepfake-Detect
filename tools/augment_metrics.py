"""Retrofit precision/recall/F1/balanced-acc/FPR/MCC onto existing metrics.csv files
(computed from their saved TN/FP/FN/TP), writing metrics_full.csv next to each.
Also writes a combined results/summary_all_runs.csv with one Mean row per run.

Usage: python tools/augment_metrics.py [results_dir]
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from eval_report import derived_metrics

BASE_KEYS = ['acc', 'real_acc', 'fake_acc', 'ap', 'auc']
NEW_KEYS = ['precision', 'recall', 'f1', 'balanced_acc', 'fpr', 'mcc']


def augment_file(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    subset_rows = [r for r in rows if r['subset'] != 'Mean' and r.get('tn')]
    out_rows = []
    for r in subset_rows:
        d = derived_metrics(int(r['tn']), int(r['fp']), int(r['fn']), int(r['tp']))
        out = {k: r[k] for k in ['subset', 'n_real', 'n_fake'] + BASE_KEYS}
        out.update({k: f'{d[k]*100:.2f}' for k in NEW_KEYS})
        out.update({k: r[k] for k in ['tn', 'fp', 'fn', 'tp']})
        out_rows.append(out)
    mean = {'subset': 'Mean', 'n_real': '', 'n_fake': '', 'tn': '', 'fp': '', 'fn': '', 'tp': ''}
    for k in BASE_KEYS + NEW_KEYS:
        vals = [float(r[k]) for r in out_rows]
        mean[k] = f'{sum(vals)/len(vals):.2f}'
    out_path = os.path.join(os.path.dirname(path), 'metrics_full.csv')
    fields = ['subset', 'n_real', 'n_fake'] + BASE_KEYS + NEW_KEYS + ['tn', 'fp', 'fn', 'tp']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows + [mean])
    return mean


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'results'
    summary = []
    for path in sorted(glob.glob(os.path.join(root, '*', 'metrics.csv'))):
        run = os.path.basename(os.path.dirname(path))
        mean = augment_file(path)
        mean['run'] = run
        summary.append(mean)
        print(f'{run:45s} acc={mean["acc"]}% recall={mean["recall"]}% precision={mean["precision"]}% f1={mean["f1"]}% bal_acc={mean["balanced_acc"]}%')
    out = os.path.join(root, 'summary_all_runs.csv')
    fields = ['run'] + BASE_KEYS + NEW_KEYS
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(summary)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
