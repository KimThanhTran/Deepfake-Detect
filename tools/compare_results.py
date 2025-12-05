import argparse
import csv
from collections import defaultdict

# Expected CSV format for both files: columns subset, acc, ap (acc/ap can be as percent or float 0-100)
# Example row: subset,acc,ap  ->  sdv5,94.4,99.9


def normalize_score(x):
    try:
        v = float(str(x).strip())
        # Heuristic: if value <= 1.5, assume it's probability (0-1) and convert to percent
        if v <= 1.5:
            return v * 100.0
        return v
    except Exception:
        return None


def read_csv(path):
    res = {}
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            subset = row.get('subset') or row.get('name') or row.get('dataset')
            if not subset:
                continue
            acc = normalize_score(row.get('acc'))
            ap = normalize_score(row.get('ap') or row.get('auc_pr') or row.get('ap_pr'))
            res[subset.strip()] = {'acc': acc, 'ap': ap}
    return res


def main():
    p = argparse.ArgumentParser(description='Compare our results CSV to a baseline CSV (subset,acc,ap).')
    p.add_argument('--ours', required=True, help='Path to our CSV (subset,acc,ap).')
    p.add_argument('--baseline', required=True, help='Path to baseline CSV (subset,acc,ap).')
    p.add_argument('--out', default=None, help='Optional path to write comparison CSV. If omitted, no file is written.')
    p.add_argument('--no_print', action='store_true', help='Do not print comparison table to console.')
    args = p.parse_args()

    ours = read_csv(args.ours)
    base = read_csv(args.baseline)

    rows = []
    acc_diffs = []
    ap_diffs = []
    subsets = sorted(set(ours.keys()) | set(base.keys()))
    for s in subsets:
        o = ours.get(s)
        b = base.get(s)
        o_acc = o.get('acc') if o else None
        o_ap = o.get('ap') if o else None
        b_acc = b.get('acc') if b else None
        b_ap = b.get('ap') if b else None
        d_acc = (o_acc - b_acc) if (o_acc is not None and b_acc is not None) else None
        d_ap = (o_ap - b_ap) if (o_ap is not None and b_ap is not None) else None
        if d_acc is not None:
            acc_diffs.append(d_acc)
        if d_ap is not None:
            ap_diffs.append(d_ap)
        rows.append({
            'subset': s,
            'ours_acc': f'{o_acc:.1f}' if o_acc is not None else '',
            'base_acc': f'{b_acc:.1f}' if b_acc is not None else '',
            'diff_acc': f'{d_acc:+.1f}' if d_acc is not None else '',
            'ours_ap': f'{o_ap:.1f}' if o_ap is not None else '',
            'base_ap': f'{b_ap:.1f}' if b_ap is not None else '',
            'diff_ap': f'{d_ap:+.1f}' if d_ap is not None else '',
        })

    # Add mean row if any diffs exist
    if acc_diffs or ap_diffs:
        from statistics import mean
        mean_acc = mean(acc_diffs) if acc_diffs else 0.0
        mean_ap = mean(ap_diffs) if ap_diffs else 0.0
        rows.append({
            'subset': 'Mean_Delta',  # avoid non-ASCII for Windows default encodings
            'ours_acc': '',
            'base_acc': '',
            'diff_acc': f'{mean_acc:+.1f}',
            'ours_ap': '',
            'base_ap': '',
            'diff_ap': f'{mean_ap:+.1f}',
        })

    # Optionally write CSV
    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['subset','ours_acc','base_acc','diff_acc','ours_ap','base_ap','diff_ap'])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print('Saved comparison to', args.out)

    # Print to console by default
    if not args.no_print:
        # Compute simple column widths
        headers = ['subset','ours_acc','base_acc','diff_acc','ours_ap','base_ap','diff_ap']
        data = [headers]
        for r in rows:
            data.append([r.get(h, '') for h in headers])
        col_w = [max(len(str(row[i])) for row in data) for i in range(len(headers))]
        # Pretty print
        def fmt_row(vals):
            return '  '.join(str(v).ljust(col_w[i]) for i, v in enumerate(vals))

        print('\nComparison (percent values; diff in points):')
        print(fmt_row(headers))
        print('-' * (sum(col_w) + 2 * (len(col_w) - 1)))
        for r in rows:
            print(fmt_row([r.get(h, '') for h in headers]))


if __name__ == '__main__':
    main()
