import os
import sys
import io
import tempfile
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import gradio as gr

# Ensure repo root is on sys.path so we can import `networks.*`
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from networks.resnet import resnet50


# ---------- Inference utilities (reuse GenImage transforms) ----------

def translate_duplicate(img: Image.Image, crop_size: int = 224):
    if min(img.size) < crop_size:
        width, height = img.size
        tiles_x = int(np.ceil(crop_size / width))
        tiles_y = int(np.ceil(crop_size / height))
        new_img = Image.new('RGB', (width * tiles_x, height * tiles_y))
        for i in range(tiles_x):
            for j in range(tiles_y):
                new_img.paste(img, (i * width, j * height))
        return new_img
    return img


def build_transform(crop_size: int = 224):
    crop = transforms.CenterCrop(crop_size)
    return transforms.Compose([
        transforms.Lambda(lambda img: translate_duplicate(img, crop_size)),
        crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------- Model loading ----------
_MODEL = None
_MODEL_PATH = None
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_TRANSFORM = build_transform(224)


def load_model(model_path: str):
    global _MODEL, _MODEL_PATH
    model = resnet50(num_classes=1)
    state = torch.load(model_path, map_location='cpu')
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    from collections import OrderedDict
    if any(k.startswith('module.') for k in sd.keys()):
        sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] Checkpoint mismatch — missing keys: {missing}, unexpected keys: {unexpected}")
    model.to(_DEVICE)
    model.eval()
    _MODEL = model
    _MODEL_PATH = model_path
    status = f"Loaded model: {os.path.basename(model_path)} on {_DEVICE}"
    if missing or unexpected:
        status += f" (WARNING: {len(missing)} missing / {len(unexpected)} unexpected keys — predictions may be unreliable)"
    return status


def ensure_model(model_path: str):
    if _MODEL is None or _MODEL_PATH != model_path:
        return load_model(model_path)
    return f"Model already loaded on {_DEVICE}"


# ---------- Prediction functions ----------

def predict_image(img: Image.Image, model_path: str) -> Tuple[str, float]:
    ensure_model(model_path)
    assert _MODEL is not None
    # Convert numpy array from Gradio to PIL if needed
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    tens = _TRANSFORM(img).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(_MODEL(tens)).flatten().item()
    label = 'AI-generated' if prob >= 0.5 else 'Nature (real)'
    return label, float(prob)


def predict_batch(file_paths: List[str], model_path: str):
    ensure_model(model_path)
    assert _MODEL is not None
    rows = []
    with torch.no_grad():
        batch_tensors = []
        names = []
        for i, path in enumerate(file_paths or []):
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                # Skip invalid files
                continue
            name = os.path.basename(path)
            names.append(name)
            batch_tensors.append(_TRANSFORM(img))
        if not batch_tensors:
            return rows, None
        x = torch.stack(batch_tensors, dim=0).to(_DEVICE)
        probs = torch.sigmoid(_MODEL(x)).flatten().cpu().numpy().tolist()
        for name, p in zip(names, probs):
            rows.append({
                'file': name,
                'prob_ai': round(float(p), 4),
                'pred': 'AI-generated' if p >= 0.5 else 'Nature (real)'
            })

    # Save CSV to a temp file for download
    df_csv = 'file,prob_ai,pred\n' + '\n'.join(f"{r['file']},{r['prob_ai']},{r['pred']}" for r in rows)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp.write(df_csv.encode('utf-8'))
    tmp.flush(); tmp.close()
    return rows, tmp.name


# ---------- CSV comparison (ours vs baseline) ----------
import csv as _csv


def _normalize_score(x):
    try:
        v = float(str(x).strip())
        if v <= 1.5:
            return v * 100.0
        return v
    except Exception:
        return None


def _read_results_csv(path):
    res = {}
    with open(path, 'r', newline='') as f:
        r = _csv.DictReader(f)
        for row in r:
            subset = row.get('subset') or row.get('name') or row.get('dataset')
            if not subset:
                continue
            acc = _normalize_score(row.get('acc'))
            ap = _normalize_score(row.get('ap') or row.get('auc_pr') or row.get('ap_pr'))
            res[subset.strip()] = {'acc': acc, 'ap': ap}
    return res


def compare_csvs(ours_file, baseline_file):
    if ours_file is None or baseline_file is None:
        return [], None
    ours = _read_results_csv(ours_file.name)
    base = _read_results_csv(baseline_file.name)
    subsets = sorted(set(ours.keys()) | set(base.keys()))
    rows = []
    acc_diffs = []
    ap_diffs = []
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
            'ours_acc(%)': f'{o_acc:.1f}' if o_acc is not None else '',
            'base_acc(%)': f'{b_acc:.1f}' if b_acc is not None else '',
            'diff_acc(pts)': f'{d_acc:+.1f}' if d_acc is not None else '',
            'ours_ap(%)': f'{o_ap:.1f}' if o_ap is not None else '',
            'base_ap(%)': f'{b_ap:.1f}' if b_ap is not None else '',
            'diff_ap(pts)': f'{d_ap:+.1f}' if d_ap is not None else '',
        })
    if acc_diffs or ap_diffs:
        from statistics import mean
        rows.append({
            'subset': 'Mean_Delta',
            'ours_acc(%)': '', 'base_acc(%)': '', 'diff_acc(pts)': f'{mean(acc_diffs):+.1f}',
            'ours_ap(%)': '', 'base_ap(%)': '', 'diff_ap(pts)': f'{mean(ap_diffs) if ap_diffs else 0.0:+.1f}',
        })

    # Save temp CSV for download
    header = list(rows[0].keys()) if rows else ['subset','ours_acc(%)','base_acc(%)','diff_acc(pts)','ours_ap(%)','base_ap(%)','diff_ap(pts)']
    csv_buf = io.StringIO()
    w = _csv.DictWriter(csv_buf, fieldnames=header)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8', newline='')
    tmp.write(csv_buf.getvalue()); tmp.flush(); tmp.close()
    return rows, tmp.name


# ---------- Gradio UI ----------

def build_ui():
    with gr.Blocks(title="NPR Deepfake Detection UI") as demo:
        gr.Markdown("# NPR Deepfake Detection\nUpload images to detect AI-generated vs nature, or compare result CSVs.")

        with gr.Accordion("Model", open=True):
            model_path = gr.Textbox(value="./NPR-DeepfakeDetection/NPR.pth", label="Model path (.pth)")
            load_btn = gr.Button("Load/Reload model")
            load_out = gr.Markdown()
            load_btn.click(fn=load_model, inputs=model_path, outputs=load_out)

        with gr.Tab("Single image"):
            inp = gr.Image(type="pil", label="Upload image")
            out_label = gr.Textbox(label="Predicted label")
            out_prob = gr.Number(label="AI probability", precision=4)
            run_btn = gr.Button("Predict")
            run_btn.click(fn=predict_image, inputs=[inp, model_path], outputs=[out_label, out_prob])

        with gr.Tab("Batch (multiple images)"):
            files = gr.Files(label="Upload multiple images", file_types=["image"], type="filepath")
            tbl = gr.Dataframe(headers=["file","prob_ai","pred"], label="Results")
            csv_dl = gr.File(label="Download CSV")
            batch_btn = gr.Button("Run batch")
            batch_btn.click(fn=predict_batch, inputs=[files, model_path], outputs=[tbl, csv_dl])

        with gr.Tab("Compare CSVs"):
            ours = gr.File(label="Ours CSV (subset,acc,ap)")
            base = gr.File(label="Baseline CSV (subset,acc,ap)")
            comp_tbl = gr.Dataframe(label="Comparison")
            comp_csv = gr.File(label="Download comparison CSV")
            comp_btn = gr.Button("Compare")
            comp_btn.click(fn=compare_csvs, inputs=[ours, base], outputs=[comp_tbl, comp_csv])

        gr.Markdown("Note: For best accuracy with GenImage, the internal transform matches the paper: translate-duplicate + center crop 224.")
    return demo


if __name__ == "__main__":
    ui = build_ui()
    # Bind explicitly to localhost and a fixed port; auto-open browser
    ui.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
