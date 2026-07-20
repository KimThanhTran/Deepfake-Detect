import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt):
    data_loader = create_dataloader(opt)
    device = next(model.parameters()).device

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.to(device)
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Handle edge cases: no positive samples or no negative samples
    if np.sum(y_true == 1) == 0 or np.sum(y_true == 0) == 0:
        try:
            r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5) if np.sum(y_true==0) else np.nan
            f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5) if np.sum(y_true==1) else np.nan
            acc = accuracy_score(y_true, y_pred > 0.5)
        except Exception:
            r_acc = f_acc = acc = np.nan
        # Average precision needs both classes; fall back to 0.0 to avoid crash
        ap = 0.0
        return acc, ap, r_acc, f_acc, y_true, y_pred
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    # Sklearn in some versions expects a column vector for y_score in binary AP.
    # Provide (n_samples, 1) to be robust; y_true kept 1D. Wrap in try/except for safety.
    try:
        ap = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1, 1))
    except Exception:
        # Fallback to 0.0 if AP cannot be computed (should be rare after shape fix)
        ap = 0.0
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    from util import build_npr_model
    model, adaptive = build_npr_model(opt.model_path)
    if adaptive:
        print('[info] AdaptiveNPR checkpoint detected — using adaptive architecture')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
