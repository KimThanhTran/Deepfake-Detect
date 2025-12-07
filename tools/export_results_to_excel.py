import csv
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter
except ImportError:
    raise SystemExit("Please install openpyxl: pip install openpyxl")

ROOT = Path('.')
CSV_TOTAL = ROOT / 'TOTAL_RESULTS.csv'
CSV_DIFF_FOR = ROOT / 'results' / 'evaluation_details_diffusionforensics.csv'
CSV_DIFF_1K = ROOT / 'results' / 'evaluation_details_diffusion1k.csv'
XLSX_OUT = ROOT / 'TOTAL_RESULT.xlsx'

def read_csv_rows(path):
    if not path.exists():
        return [], []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            return [], []
        header = rows[0]
        data = rows[1:]
        return header, data

def write_sheet(ws, title, header, data):
    ws.title = title
    if header:
        ws.append(header)
        # style header
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center')
    for row in data:
        ws.append(row)
    # autosize
    for col_idx, col in enumerate(ws.columns, start=1):
        max_len = 0
        for cell in col:
            val = cell.value
            if val is None:
                continue
            val_str = str(val)
            if len(val_str) > max_len:
                max_len = len(val_str)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 60)
    ws.freeze_panes = 'A2'

def build_comparison_sheet(wb, total_csv):
    import csv
    ws = wb.create_sheet('Baseline_vs_Improved')
    header = ['Dataset','Baseline Acc','Improved Acc','Δ Accuracy','Baseline AP','Improved AP','Δ AP','Baseline ROC-AUC','Improved ROC-AUC','Δ ROC-AUC','Samples','Notes']
    rows = []
    with open(total_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        items = [row for row in r if row.get('Dataset')]
    # group by dataset
    from collections import defaultdict
    by_ds = defaultdict(list)
    for it in items:
        by_ds[it['Dataset']].append(it)
    def pct_to_float(s):
        try:
            return float(s.strip().replace('%',''))
        except Exception:
            return None
    for ds, rows_ds in by_ds.items():
        def safe_str(x):
            return (x or '')
        base = next((x for x in rows_ds if 'Baseline' in safe_str(x.get('Model'))), None)
        imp = next((x for x in rows_ds if 'Trained' in safe_str(x.get('Model'))), None)
        if not base or not imp:
            continue
        b_acc = base.get('Accuracy','N/A'); i_acc = imp.get('Accuracy','N/A')
        b_ap  = base.get('AP','N/A');       i_ap  = imp.get('AP','N/A')
        b_roc = base.get('ROC_AUC','N/A');  i_roc = imp.get('ROC_AUC','N/A')
        d_acc = 'N/A'; d_ap = 'N/A'; d_roc = 'N/A'
        bf = pct_to_float(b_acc); ifi = pct_to_float(i_acc)
        if bf is not None and ifi is not None:
            d_acc = f"{ifi - bf:+.2f}%"
        bf = pct_to_float(b_ap); ifi = pct_to_float(i_ap)
        if bf is not None and ifi is not None:
            d_ap = f"{ifi - bf:+.2f}%"
        bf = pct_to_float(b_roc); ifi = pct_to_float(i_roc)
        if bf is not None and ifi is not None:
            d_roc = f"{ifi - bf:+.2f}%"
        rows.append([ds, b_acc, i_acc, d_acc, b_ap, i_ap, d_ap, b_roc, i_roc, d_roc, imp.get('Samples','N/A'), imp.get('Notes','')])
    write_sheet(ws, 'Baseline_vs_Improved', header, rows)

def main():
    wb = Workbook()
    # Default sheet
    ws0 = wb.active
    ws0.title = 'README'
    ws0.append(["NPR Deepfake Detection — Tổng hợp kết quả"])
    ws0.append(["Xem các sheet: TOTAL_RESULTS (số liệu), Baseline_vs_Improved (so sánh), DiffusionForensics (chi tiết), Diffusion1kStep (chi tiết), Notes (mô tả)."])

    # TOTAL_RESULTS
    hdr, data = read_csv_rows(CSV_TOTAL)
    ws_total = wb.create_sheet('TOTAL_RESULTS')
    write_sheet(ws_total, 'TOTAL_RESULTS', hdr, data)

    # NOTES: trích các phần mô tả từ TOTAL_RESULTS.csv
    notes_rows = []
    if hdr and data:
        # Lọc các dòng thuộc các mục mô tả
        import csv
        with CSV_TOTAL.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                cat = (r.get('Category') or '').lower()
                ds = r.get('Dataset') or ''
                if cat in ['delta','achievement','summary'] or ds in ['IMPROVEMENTS (Baseline → Trained):','TRAINING CONFIGURATION:','DATASET STATISTICS:','PERFORMANCE ANALYSIS:','RECOMMENDATIONS:']:
                    notes_rows.append([r.get('Dataset',''), r.get('Model',''), r.get('Accuracy',''), r.get('AP',''), r.get('ROC_AUC',''), r.get('Real_Acc',''), r.get('Fake_Acc',''), r.get('Samples',''), r.get('Category',''), r.get('Notes','')])
    ws_notes = wb.create_sheet('Notes')
    write_sheet(ws_notes, 'Notes', ['Dataset','Model','Accuracy','AP','ROC_AUC','Real_Acc','Fake_Acc','Samples','Category','Notes'], notes_rows)

    # DiffusionForensics details
    hdr_df, data_df = read_csv_rows(CSV_DIFF_FOR)
    ws_df = wb.create_sheet('DiffusionForensics')
    write_sheet(ws_df, 'DiffusionForensics', hdr_df, data_df)

    # Diffusion1kStep details
    hdr_d1k, data_d1k = read_csv_rows(CSV_DIFF_1K)
    ws_d1k = wb.create_sheet('Diffusion1kStep')
    write_sheet(ws_d1k, 'Diffusion1kStep', hdr_d1k, data_d1k)

    # Baseline vs Improved
    build_comparison_sheet(wb, str(CSV_TOTAL))

    wb.save(XLSX_OUT)
    print(f"Saved Excel workbook to {XLSX_OUT}")

if __name__ == '__main__':
    main()
