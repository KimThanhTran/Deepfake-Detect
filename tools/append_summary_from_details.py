import csv
from statistics import mean

DETAILS_FILES = [
    r".\results\evaluation_details_diffusionforensics.csv",
    r".\results\evaluation_details_diffusion1k.csv",
]
OUT_TOTAL = r".\TOTAL_RESULTS.csv"

FIELDS = ['Dataset','Model','Accuracy','AP','ROC_AUC','Real_Acc','Fake_Acc','Samples','Category','Notes']

def try_float(x):
    try:
        return float(x)
    except Exception:
        return None

def append_rows(rows):
    with open(OUT_TOTAL, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        for r in rows:
            w.writerow(r)

def make_summary(details_path, dataset_label):
    with open(details_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        accs, rocs, prs = [], [], []
        count = 0
        for r in reader:
            a = try_float(r.get('acc'))
            roc = try_float(r.get('auc_roc'))
            pr = try_float(r.get('auc_pr'))
            if a is not None:
                accs.append(a)
            if roc is not None:
                rocs.append(roc)
            if pr is not None:
                prs.append(pr)
            count += 1
    if not accs:
        return None
    row = {
        'Dataset': dataset_label,
        'Model': 'Trained (30 epochs)',
        'Accuracy': f"{mean(accs):.2f}%",
        'AP': f"{mean(prs):.2f}%" if prs else 'N/A',
        'ROC_AUC': f"{mean(rocs):.2f}%" if rocs else 'N/A',
        'Real_Acc': 'N/A',
        'Fake_Acc': 'N/A',
        'Samples': 'N/A',
        'Category': 'Cross-Dataset',
        'Notes': f"Mean across {count} subsets from details"
    }
    return row

def main():
    rows = []
    for path in DETAILS_FILES:
        if 'diffusionforensics' in path:
            label = 'DiffusionForensics'
        elif 'diffusion1k' in path:
            label = 'Diffusion1kStep'
        else:
            label = path
        s = make_summary(path, label)
        if s:
            rows.append(s)
    if rows:
        append_rows(rows)
        print(f"Appended {len(rows)} summary rows to {OUT_TOTAL}")
    else:
        print("No summaries generated.")

if __name__ == '__main__':
    main()
