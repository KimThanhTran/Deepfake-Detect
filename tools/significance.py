"""
Paired significance tests: fusion (w*A + (1-w)*B) vs model A, per benchmark and over the 5-benchmark mean.

- Paired stratified bootstrap (resample images within each subset x label, same indices for both
  systems) of the reported metrics: mean over subsets, threshold 0.5.
- Exact McNemar test on per-image correctness at threshold 0.5 (pooled over the benchmark).

Usage (inside the train container):
    python tools/significance.py --n_boot 1000 --out results/significance.csv
"""
import os
import csv
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score

BENCHES = ['ForenSynths-test', 'GANGen-Detection', 'UniversalFakeDetect', 'DiffusionForensics', 'Diffusion1kStep']
METRICS = ['acc', 'real_acc', 'fake_acc', 'ap', 'auc', 'f1', 'mcc']


def load(path):
    by = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            by[r['subset']][r['relpath']] = (int(r['label']), float(r['score']))
    return by


def subset_metrics(y, s):
    p = s > 0.5
    tp = np.sum(p & (y == 1)); tn = np.sum(~p & (y == 0))
    fp = np.sum(p & (y == 0)); fn = np.sum(~p & (y == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        'acc': (tp + tn) / len(y),
        'real_acc': tn / (tn + fp),
        'fake_acc': rec,
        'ap': average_precision_score(y, s),
        'auc': roc_auc_score(y, s),
        'f1': 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        'mcc': (tp * tn - fp * fn) / den if den else 0.0,
    }


def mean_metrics(subsets, idx=None):
    vals = defaultdict(list)
    for k, (y, s) in subsets.items():
        if idx is not None:
            y, s = y[idx[k]], s[idx[k]]
        for m, v in subset_metrics(y, s).items():
            vals[m].append(v)
    return {m: float(np.mean(v)) for m, v in vals.items()}


def p_two_sided(diffs):
    n = len(diffs)
    lo = (np.sum(diffs <= 0) + 1) / (n + 1)
    hi = (np.sum(diffs >= 0) + 1) / (n + 1)
    return min(1.0, 2 * min(lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='results/scores/NPR_{}', help='baseline run dir pattern')
    ap.add_argument('--b', default='results/scores/npr_adaptive_aug_{}', help='second model run dir pattern')
    ap.add_argument('--weight', type=float, default=0.5)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/significance.csv')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = {}  # bench -> (base_subsets, fusion_subsets, strata)
    for bench in BENCHES:
        A = load(os.path.join(args.a.format(bench), 'scores.csv'))
        B = load(os.path.join(args.b.format(bench), 'scores.csv'))
        base, fus, strata = {}, {}, {}
        for sub in sorted(A):
            keys = sorted(A[sub])
            assert set(keys) == set(B[sub]), f'{bench}/{sub}: image sets differ'
            y = np.array([A[sub][k][0] for k in keys])
            sa = np.array([A[sub][k][1] for k in keys])
            sb = np.array([B[sub][k][1] for k in keys])
            base[sub] = (y, sa)
            fus[sub] = (y, args.weight * sa + (1 - args.weight) * sb)
            strata[sub] = (np.flatnonzero(y == 0), np.flatnonzero(y == 1))
        data[bench] = (base, fus, strata)

    point = {b: (mean_metrics(d[0]), mean_metrics(d[1])) for b, d in data.items()}
    boot = {b: {m: [] for m in METRICS} for b in BENCHES + ['Mean5']}
    for _ in range(args.n_boot):
        mean_diff = defaultdict(float)
        for bench, (base, fus, strata) in data.items():
            idx = {k: np.concatenate([rng.choice(r, len(r)), rng.choice(f, len(f))])
                   for k, (r, f) in strata.items()}
            mb, mf = mean_metrics(base, idx), mean_metrics(fus, idx)
            for m in METRICS:
                d = mf[m] - mb[m]
                boot[bench][m].append(d)
                mean_diff[m] += d / len(BENCHES)
        for m in METRICS:
            boot['Mean5'][m].append(mean_diff[m])

    rows = []
    for bench in BENCHES + ['Mean5']:
        if bench == 'Mean5':
            pb = {m: np.mean([point[b][0][m] for b in BENCHES]) for m in METRICS}
            pf = {m: np.mean([point[b][1][m] for b in BENCHES]) for m in METRICS}
        else:
            pb, pf = point[bench]
        for m in METRICS:
            d = np.array(boot[bench][m])
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({'benchmark': bench, 'metric': m,
                         'baseline': f'{pb[m]*100:.2f}', 'fusion': f'{pf[m]*100:.2f}',
                         'diff': f'{(pf[m]-pb[m])*100:+.2f}',
                         'ci95_low': f'{lo*100:+.2f}', 'ci95_high': f'{hi*100:+.2f}',
                         'p_boot': f'{p_two_sided(d):.4f}'})

    mcn = []
    for bench, (base, fus, _) in data.items():
        y = np.concatenate([v[0] for v in base.values()])
        cb = np.concatenate([(v[1] > 0.5) == (v[0] == 1) for v in base.values()])
        cf = np.concatenate([(v[1] > 0.5) == (v[0] == 1) for v in fus.values()])
        b_only, f_only = int(np.sum(cb & ~cf)), int(np.sum(~cb & cf))
        p = binomtest(f_only, b_only + f_only, 0.5).pvalue if b_only + f_only else 1.0
        mcn.append((bench, len(y), b_only, f_only, p))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(args.out.replace('.csv', '_mcnemar.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['benchmark', 'n_images', 'only_baseline_correct', 'only_fusion_correct', 'p_exact'])
        w.writerows([(b, n, x, z, f'{p:.3g}') for b, n, x, z, p in mcn])

    print(f'Paired stratified bootstrap, n={args.n_boot} (diff = fusion - baseline, % points)')
    for r in rows:
        sig = '*' if float(r['p_boot']) < 0.05 else ' '
        print(f"{r['benchmark']:20s} {r['metric']:9s} {r['baseline']:>6s} -> {r['fusion']:>6s} "
              f"{r['diff']:>7s} [{r['ci95_low']}, {r['ci95_high']}] p={r['p_boot']} {sig}")
    print('\nExact McNemar (accuracy @0.5, pooled per benchmark)')
    for b, n, x, z, p in mcn:
        print(f'{b:20s} n={n:6d} only_base={x:5d} only_fusion={z:5d} p={p:.3g}')


if __name__ == '__main__':
    main()
