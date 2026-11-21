"""
Score-fusion variants of NPR (A) + AdaptiveNPR+Aug(strong) (B) whose parameters never see the benchmark
they are evaluated on:
  mean        p = 0.5*pA + 0.5*pB                                   (no parameters)
  logit_mean  p = sigmoid((logit pA + logit pB) / 2)                (no parameters; keeps saturated-score order)
  platt_val   per-model Platt scaling fit on ForenSynths-val, then mean of calibrated probs
  stack_val   logistic regression on [logit pA, logit pB] fit on ForenSynths-val
  lobo_w      p = w*pA + (1-w)*pB, w chosen on the OTHER four benchmarks (leave-one-benchmark-out),
              criterion: mean over those benchmarks of (Accuracy + AUC) / 2
  lobo_stack  logistic regression on [logit pA, logit pB] fit on the OTHER four benchmarks
              (each subset x label weighted equally)
Every method thresholds at 0.5 and is scored with the paper's protocol (per-subset metrics, mean).

Writes results/fusion_<method>_<bench>/{scores.csv,metrics.csv,confusion_matrices.png} and
results/fusion_variants.csv (method x bench x metric, plus diff vs baseline).
Usage (train container):  python tools/fusion_variants.py
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from eval_report import metrics_row, write_report, METRIC_KEYS

BENCHES = ['ForenSynths-test', 'GANGen-Detection', 'UniversalFakeDetect', 'DiffusionForensics', 'Diffusion1kStep']
A, B = 'results/scores/NPR_{}', 'results/scores/npr_adaptive_aug_{}'
EPS = 1e-7


def load_pair(name):
    """-> {subset: (keys, y, pA, pB)} with both models aligned on the same images."""
    def read(d):
        out = defaultdict(dict)
        with open(os.path.join(d, 'scores.csv')) as f:
            for r in csv.DictReader(f):
                out[r['subset']][r['relpath']] = (int(r['label']), float(r['score']))
        return out
    a, b = read(A.format(name)), read(B.format(name))
    data = {}
    for sub in sorted(a):
        keys = sorted(a[sub])
        assert set(keys) == set(b[sub]), f'{name}/{sub}: image sets differ'
        data[sub] = (keys, np.array([a[sub][k][0] for k in keys]),
                     np.array([a[sub][k][1] for k in keys]), np.array([b[sub][k][1] for k in keys]))
    return data


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    # rounding makes logit(1-EPS) == -logit(EPS) exactly, so contradictory saturated pairs fuse to exactly 0.5
    return np.round(np.log(p / (1 - p)), 6)


def feats(pa, pb):
    return np.column_stack([logit(pa), logit(pb)])


def evaluate(bench_data, fuse):
    """Paper protocol: metrics per subset of fused scores, then mean."""
    rows = {s: metrics_row(y, fuse(pa, pb)) for s, (_, y, pa, pb) in bench_data.items()}
    return rows, {k: float(np.nanmean([r[k] for r in rows.values()])) for k in METRIC_KEYS}


def balanced_fit(X, y, groups):
    w = np.empty(len(y))
    for g in set(groups):
        idx = groups == g
        w[idx] = 1.0 / idx.sum()
    return LogisticRegression(C=1.0, max_iter=1000).fit(X, y, sample_weight=w)


def stacked(data_list):
    X, y, g = [], [], []
    for bi, data in enumerate(data_list):
        for s, (_, ys, pa, pb) in data.items():
            X.append(feats(pa, pb)); y.append(ys); g += [f'{bi}/{s}/{l}' for l in ys]
    return np.vstack(X), np.concatenate(y), np.array(g)


def main():
    data = {b: load_pair(b) for b in BENCHES}
    val = load_pair('val')
    methods = {'mean': {b: (lambda pa, pb: 0.5 * pa + 0.5 * pb) for b in BENCHES},
               'logit_mean': {b: (lambda pa, pb: 1 / (1 + np.exp(-(logit(pa) + logit(pb)) / 2))) for b in BENCHES}}

    Xv, yv, gv = stacked([val])
    ca = balanced_fit(Xv[:, :1], yv, gv)
    cb = balanced_fit(Xv[:, 1:], yv, gv)
    methods['platt_val'] = {b: (lambda pa, pb: 0.5 * ca.predict_proba(logit(pa)[:, None])[:, 1]
                                + 0.5 * cb.predict_proba(logit(pb)[:, None])[:, 1]) for b in BENCHES}
    sv = balanced_fit(Xv, yv, gv)
    methods['stack_val'] = {b: (lambda pa, pb: sv.predict_proba(feats(pa, pb))[:, 1]) for b in BENCHES}

    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    perf = {(b, w): evaluate(data[b], lambda pa, pb, w=w: w * pa + (1 - w) * pb)[1] for b in BENCHES for w in grid}
    chosen = {}
    methods['lobo_w'], methods['lobo_stack'] = {}, {}
    for b in BENCHES:
        others = [o for o in BENCHES if o != b]
        score = {w: np.mean([(perf[(o, w)]['acc'] + perf[(o, w)]['auc']) / 2 for o in others]) for w in grid}
        w_b = max(grid, key=lambda w: (round(score[w], 6), w))  # ties -> more weight on the baseline
        chosen[b] = w_b
        methods['lobo_w'][b] = lambda pa, pb, w=w_b: w * pa + (1 - w) * pb
        st = balanced_fit(*stacked([data[o] for o in others]))
        methods['lobo_stack'][b] = lambda pa, pb, m=st: m.predict_proba(feats(pa, pb))[:, 1]
    print('lobo_w chosen weights (weight on NPR baseline):', chosen)
    print('stack_val coef [NPR, AdaNPR+Aug], intercept:', sv.coef_.round(3).tolist(), sv.intercept_.round(3).tolist())

    base = {b: evaluate(data[b], lambda pa, pb: pa)[1] for b in BENCHES}
    out_rows = []
    for name, per_bench in methods.items():
        means = {}
        for b in BENCHES:
            rows, mean = evaluate(data[b], per_bench[b])
            means[b] = mean
            out = f'results/fusion_{name}_{b}'
            sys.stdout = open(os.devnull, 'w')
            write_report(rows, out, f'fusion[{name}] - {b}')
            sys.stdout = sys.__stdout__
            with open(os.path.join(out, 'scores.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['subset', 'relpath', 'label', 'score'])
                for s, (keys, y, pa, pb) in data[b].items():
                    for k, l, p in zip(keys, y, per_bench[b](pa, pb)):
                        w.writerow([s, k, int(l), f'{p:.9g}'])
        for b in BENCHES + ['Mean5']:
            m = means[b] if b != 'Mean5' else {k: np.mean([means[x][k] for x in BENCHES]) for k in METRIC_KEYS}
            bs = base[b] if b != 'Mean5' else {k: np.mean([base[x][k] for x in BENCHES]) for k in METRIC_KEYS}
            for k in METRIC_KEYS:
                d = (m[k] - bs[k]) * (-1 if k == 'fpr' else 1)
                out_rows.append({'method': name, 'benchmark': b, 'metric': k, 'baseline': f'{bs[k]*100:.2f}',
                                 'fusion': f'{m[k]*100:.2f}', 'gain': f'{d*100:+.2f}'})

    with open('results/fusion_variants.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)

    print(f'\n{"method":11s} {"losses/55":>9s} {"worst":>7s}  ' + '  '.join(f'{k[:7]:>7s}' for k in METRIC_KEYS)
          + '   (Mean5 gain vs baseline, % points; fpr sign flipped)')
    for name in methods:
        rs = [r for r in out_rows if r['method'] == name]
        per = [float(r['gain']) for r in rs if r['benchmark'] != 'Mean5']
        mean5 = {r['metric']: r['gain'] for r in rs if r['benchmark'] == 'Mean5'}
        print(f'{name:11s} {sum(g < -0.005 for g in per):9d} {min(per):+7.2f}  '
              + '  '.join(f'{mean5[k]:>7s}' for k in METRIC_KEYS))
    for name in methods:
        bad = [f"{r['benchmark']}:{r['metric']}({r['gain']})" for r in out_rows
               if r['method'] == name and r['benchmark'] != 'Mean5' and float(r['gain']) < -0.005]
        print(f'  {name} losses: {", ".join(bad) or "none"}')


if __name__ == '__main__':
    main()
