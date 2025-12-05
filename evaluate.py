import os
import csv
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from options.test_options import TestOptions
from networks.resnet import resnet50
from validate import validate


def compute_metrics(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    acc = accuracy_score(y_true, y_score > 0.5)
    fpr = float(np.mean((y_true == 0) & (y_score > 0.5))) if np.any(y_true == 0) else np.nan
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except Exception:
        auc_roc = np.nan
    # AUC-PR (Average Precision)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_pr = auc(recall, precision)
    except Exception:
        auc_pr = np.nan
    return acc, auc_roc, auc_pr, fpr


def main():
    parser = argparse.ArgumentParser(description='Evaluate detector with Accuracy, AUC, FPR, Generalization')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_csv', default='evaluation_details.csv')
    parser.add_argument('--in_domain_root', default=None, help='Optional in-domain validation root for generalization reference')
    args, _ = parser.parse_known_args()

    # Build options for validate()
    opt = TestOptions().parse(print_options=False)
    opt.batch_size = args.batch_size
    opt.num_threads = 0

    # Load model
    model = resnet50(num_classes=1)
    state = torch.load(args.model_path, map_location='cpu')
    from collections import OrderedDict
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    if any(k.startswith('module.') for k in sd.keys()):
        sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
    model.load_state_dict(sd, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Datasets to evaluate (reuse run_all_tests config)
    from run_all_tests import DetectionSets, dataset_root
    rows = []
    ref_generalization = None

    # Optional in-domain reference
    if args.in_domain_root and os.path.isdir(args.in_domain_root):
        opt.dataroot = args.in_domain_root
        opt.classes = ''
        opt.no_resize = False
        opt.no_crop = False
        acc, ap, _, _, y_true, y_pred = validate(model, opt)
        acc_ref, aucroc_ref, aucpr_ref, fpr_ref = compute_metrics(y_true, y_pred)
        ref_generalization = {'acc': acc_ref, 'aucroc': aucroc_ref, 'aucpr': aucpr_ref, 'fpr': fpr_ref}

    for tag, cfg in DetectionSets.items():
        root = os.path.join(dataset_root, cfg['root'])
        if not os.path.isdir(root):
            continue
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if 'force_subdirs' in cfg:
            subdirs = [d for d in cfg['force_subdirs'] if os.path.isdir(os.path.join(root, d))]

        for sd in subdirs:
            sd_path = os.path.join(root, sd)
            entries = [e for e in os.listdir(sd_path) if os.path.isdir(os.path.join(sd_path, e))]
            if not (('0_real' in entries and '1_fake' in entries) or len(entries) >= 2):
                continue
            opt.dataroot = sd_path
            opt.classes = ''
            opt.no_resize = cfg.get('no_resize', False)
            opt.no_crop = cfg.get('no_crop', True)
            acc, ap, _, _, y_true, y_pred = validate(model, opt)
            acc_m, aucroc_m, aucpr_m, fpr_m = compute_metrics(y_true, y_pred)
            gen_gap = None
            if ref_generalization is not None and not np.isnan(ref_generalization['acc']):
                gen_gap = float(ref_generalization['acc']) - float(acc_m)
            rows.append({
                'table': tag,
                'subset': sd,
                'acc': f'{acc_m*100:.2f}',
                'auc_roc': f'{(aucroc_m*100) if not np.isnan(aucroc_m) else np.nan:.2f}',
                'auc_pr': f'{(aucpr_m*100) if not np.isnan(aucpr_m) else np.nan:.2f}',
                'fpr': f'{(fpr_m*100) if not np.isnan(fpr_m) else np.nan:.2f}',
                'generalization_gap_from_ref_acc': f'{(gen_gap*100) if gen_gap is not None else np.nan:.2f}'
            })

    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['table','subset','acc','auc_roc','auc_pr','fpr','generalization_gap_from_ref_acc'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print('Saved evaluation to', args.out_csv)


if __name__ == '__main__':
    main()
