import argparse
import csv
from collections import defaultdict
from statistics import mean

from typing import Dict, List, Optional


def read_details(path: str):
    rows = []
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    'table': row.get('table', ''),
                    'subset': row.get('subset', ''),
                    'acc': float(row['acc']) if row.get('acc') and row['acc'].lower() != 'nan' else None,
                    'auc_roc': float(row['auc_roc']) if row.get('auc_roc') and row['auc_roc'].lower() != 'nan' else None,
                    'auc_pr': float(row['auc_pr']) if row.get('auc_pr') and row['auc_pr'].lower() != 'nan' else None,
                    'fpr': float(row['fpr']) if row.get('fpr') and row['fpr'].lower() != 'nan' else None,
                })
            except Exception:
                # skip malformed rows
                continue
    return rows


def summarize_means(rows):
    # per-table means and overall means
    by_table: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    overall: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        tbl = r['table'] or 'Unknown'
        for k in ('acc', 'auc_roc', 'auc_pr', 'fpr'):
            v = r.get(k)
            if v is None:
                continue
            by_table[tbl][k].append(v)
            overall[k].append(v)
    table_summaries = []
    for tbl, metrics in by_table.items():
        table_summaries.append({
            'table': tbl,
            'mean_acc': f"{mean(metrics['acc']):.2f}" if metrics.get('acc') else '',
            'mean_auc_roc': f"{mean(metrics['auc_roc']):.2f}" if metrics.get('auc_roc') else '',
            'mean_auc_pr': f"{mean(metrics['auc_pr']):.2f}" if metrics.get('auc_pr') else '',
            'mean_fpr': f"{mean(metrics['fpr']):.2f}" if metrics.get('fpr') else '',
        })
    overall_summary = {
        'table': 'OVERALL',
        'mean_acc': f"{mean(overall['acc']):.2f}" if overall.get('acc') else '',
        'mean_auc_roc': f"{mean(overall['auc_roc']):.2f}" if overall.get('auc_roc') else '',
        'mean_auc_pr': f"{mean(overall['auc_pr']):.2f}" if overall.get('auc_pr') else '',
        'mean_fpr': f"{mean(overall['fpr']):.2f}" if overall.get('fpr') else '',
    }
    return table_summaries, overall_summary


def try_confusion_from_preds(preds_csv: Optional[str]):
    if not preds_csv:
        return None
    try:
        import numpy as np
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
    except Exception:
        return None
    y_true = []
    y_score = []
    with open(preds_csv, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                y_true.append(int(row['y_true']))
                y_score.append(float(row['y_score']))
            except Exception:
                pass
    if not y_true:
        return None
    import numpy as np
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except Exception:
        auc_roc = None
    try:
        auc_pr = average_precision_score(y_true, y_score)
    except Exception:
        auc_pr = None
    return {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'acc': f"{acc*100:.2f}", 'precision': f"{prec*100:.2f}", 'recall': f"{rec*100:.2f}", 'f1': f"{f1*100:.2f}",
        'auc_roc': f"{auc_roc*100:.2f}" if auc_roc is not None else '',
        'auc_pr': f"{auc_pr*100:.2f}" if auc_pr is not None else '',
    }


def main():
    p = argparse.ArgumentParser(description='Summarize metrics from evaluation_details.csv and optionally compute confusion matrix from predictions CSV.')
    p.add_argument('--details_csv', default='evaluation_details.csv')
    p.add_argument('--preds_csv', default='', help='Optional CSV with columns y_true,y_score to compute confusion matrix (threshold=0.5).')
    p.add_argument('--out_csv', default='results_metrics_summary.csv')
    args = p.parse_args()

    rows = read_details(args.details_csv)
    table_summaries, overall_summary = summarize_means(rows)

    # write summary CSV
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['table','mean_acc','mean_auc_roc','mean_auc_pr','mean_fpr'])
        w.writeheader()
        for r in table_summaries:
            w.writerow(r)
        w.writerow(overall_summary)
    print('Saved metrics summary to', args.out_csv)

    # Optional confusion matrix
    cm = try_confusion_from_preds(args.preds_csv)
    if cm:
        print('Confusion matrix (threshold=0.5):')
        print(f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
        print('Derived metrics:', {k: cm[k] for k in ('acc','precision','recall','f1','auc_roc','auc_pr')})


if __name__ == '__main__':
    main()
