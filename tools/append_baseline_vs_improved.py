import csv
from collections import defaultdict

SRC = r".\TOTAL_RESULTS.csv"
FIELDNAMES = ['Dataset','Model','Accuracy','AP','ROC_AUC','Real_Acc','Fake_Acc','Samples','Category','Notes']

def pct_val(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s.replace('%',''))
    except Exception:
        return None

def main():
    with open(SRC, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get('Dataset') and r.get('Model')]

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r['Dataset']].append(r)

    comparisons = []
    for ds, rs in by_ds.items():
        baseline = next((r for r in rs if 'Baseline' in r['Model']), None)
        improved = next((r for r in rs if 'Trained' in r['Model']), None)
        if baseline and improved:
            b_acc = pct_val(baseline.get('Accuracy'))
            i_acc = pct_val(improved.get('Accuracy'))
            b_ap   = pct_val(baseline.get('AP'))
            i_ap   = pct_val(improved.get('AP'))
            b_auc  = pct_val(baseline.get('ROC_AUC'))
            i_auc  = pct_val(improved.get('ROC_AUC'))

            note = f"{baseline.get('Accuracy','N/A')} → {improved.get('Accuracy','N/A')}"
            comparisons.append({
                'Dataset': ds,
                'Model': 'Baseline vs Improved',
                'Accuracy': (f"+{i_acc - b_acc:.2f}%" if b_acc is not None and i_acc is not None else 'N/A'),
                'AP':      (f"+{(i_ap - b_ap):.2f}%" if b_ap is not None and i_ap is not None else 'N/A'),
                'ROC_AUC': (f"+{(i_auc - b_auc):.2f}%" if b_auc is not None and i_auc is not None else 'N/A'),
                'Real_Acc': 'N/A',
                'Fake_Acc': 'N/A',
                'Samples': improved.get('Samples','N/A'),
                'Category': 'Comparison',
                'Notes': note
            })

    if not comparisons:
        print('No baseline/improved pairs found. No rows appended.')
        return

    with open(SRC, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for r in comparisons:
            w.writerow(r)
    print(f"Appended {len(comparisons)} comparison rows to {SRC}")

if __name__ == '__main__':
    main()
