import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
import gradio as gr

# Ensure repo root is on sys.path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Face embedding via InsightFace (CPU by default)
try:
    from insightface.app import FaceAnalysis  # type: ignore
except Exception as e:
    FaceAnalysis = None  # type: ignore


ATT_DIR = os.path.join(REPO_ROOT, 'attendance')
ENROLL_PATH = os.path.join(ATT_DIR, 'enrollments.json')
LOG_PATH = os.path.join(ATT_DIR, 'logs.csv')


def ensure_dirs():
    os.makedirs(ATT_DIR, exist_ok=True)
    if not os.path.exists(ENROLL_PATH):
        with open(ENROLL_PATH, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', encoding='utf-8') as f:
            f.write('timestamp,name,similarity,source\n')


_FACE_APP: Optional[FaceAnalysis] = None  # type: ignore
_LAST_ERROR: str = ''


def load_face_app():
    global _FACE_APP
    if FaceAnalysis is None:
        return False, 'InsightFace not installed. Please install insightface and onnxruntime.'
    if _FACE_APP is None:
        try:
            _FACE_APP = FaceAnalysis(name='buffalo_l')
            # ctx_id=-1 for CPU; set to 0 to use GPU if available
            _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as e:
            return False, f'Failed to load FaceAnalysis: {e}'
    return True, 'FaceAnalysis loaded.'


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_embedding(img: Image.Image) -> Optional[np.ndarray]:
    global _LAST_ERROR
    ok, _ = load_face_app()
    if not ok:
        _LAST_ERROR = 'FaceAnalysis not available.'
        return None
    bgr = pil_to_bgr(img)
    try:
        faces = _FACE_APP.get(bgr)
    except Exception as e:
        _LAST_ERROR = f'Face detection error: {e}'
        faces = []
    if not faces:
        _LAST_ERROR = 'No face detected.'
        return None
    # pick largest face by area
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = best.normed_embedding
    return emb.astype(np.float32)


def save_enrollment(name: str, emb: np.ndarray) -> str:
    ensure_dirs()
    with open(ENROLL_PATH, 'r', encoding='utf-8') as f:
        db = json.load(f)
    db[name] = emb.tolist()
    with open(ENROLL_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    return f'Enrolled {name} ({len(emb)}-d embedding).'


def list_enrollments() -> List[str]:
    ensure_dirs()
    with open(ENROLL_PATH, 'r', encoding='utf-8') as f:
        db = json.load(f)
    return sorted(db.keys())


def load_enrollment(name: str) -> Optional[np.ndarray]:
    ensure_dirs()
    with open(ENROLL_PATH, 'r', encoding='utf-8') as f:
        db = json.load(f)
    v = db.get(name)
    if v is None:
        return None
    return np.asarray(v, dtype=np.float32)


def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    # Inputs are L2-normalized; cosine sim ~ dot product
    return float(np.clip(np.dot(u, v), -1.0, 1.0))


def enroll(name: str, img: Image.Image):
    if not name:
        # Update dropdown choices via component update to avoid gr.Dropdown warnings
        return 'Please enter a name.', gr.update(choices=list_enrollments(), value=None)
    if img is None:
        return 'Please upload an enrollment image.', gr.update(choices=list_enrollments(), value=None)
    emb = get_embedding(img)
    if emb is None:
        return 'No face detected. Try another image.', gr.update(choices=list_enrollments(), value=None)
    msg = save_enrollment(name, emb)
    # Return a component update for Dropdown (refresh choices, clear selection)
    return msg, gr.update(choices=list_enrollments(), value=None)


def verify(name: str, img: Image.Image, threshold: float = 0.35):
    if not name:
        return 'Please select an enrolled name.', None, None
    ref = load_enrollment(name)
    if ref is None:
        return f'Enrollment for {name} not found.', None, None
    if img is None:
        return 'Please provide a verification image (webcam or upload).', None, None
    emb = get_embedding(img)
    if emb is None:
        return f'No face detected in verification image. Details: {_LAST_ERROR}', None, None
    sim = cosine_sim(ref, emb)
    result = 'MATCH' if sim >= threshold else 'NO MATCH'
    # Log attendance
    ensure_dirs()
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f'{ts},{name},{sim:.4f},webcam/upload\n')
    return f'{result} (similarity={sim:.3f}, threshold={threshold:.2f})', sim, LOG_PATH


def verify_paths(name: str, file_paths: List[str], threshold: float = 0.35):
    if not name:
        return 'Please select an enrolled name.', None, None
    ref = load_enrollment(name)
    if ref is None:
        return f'Enrollment for {name} not found.', None, None
    if not file_paths:
        return 'Please provide at least one image file path.', None, None
    # Use first valid file
    img = None
    for p in file_paths:
        try:
            img = Image.open(p).convert('RGB')
            break
        except Exception:
            continue
    if img is None:
        return 'Failed to open any provided image path.', None, None
    emb = get_embedding(img)
    if emb is None:
        return f'No face detected in verification image. Details: {_LAST_ERROR}', None, None
    sim = cosine_sim(ref, emb)
    result = 'MATCH' if sim >= threshold else 'NO MATCH'
    ensure_dirs()
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f'{ts},{name},{sim:.4f},filepath\n')
    return f'{result} (similarity={sim:.3f}, threshold={threshold:.2f})', sim, LOG_PATH


def refresh_names():
    """Return a component update to refresh enrolled names in dropdowns."""
    return gr.update(choices=list_enrollments(), value=None)


def build_ui():
    ensure_dirs()
    with gr.Blocks(title='Attendance Verification') as demo:
        gr.Markdown('# Attendance: Face Enrollment & Verification')
        ok, msg = load_face_app()
        status = gr.Markdown(value=msg)

        with gr.Tab('Enroll'):
            name_in = gr.Textbox(label='Name')
            enroll_img = gr.Image(type='pil', label='Enrollment image', sources=['upload'])
            enroll_btn = gr.Button('Enroll')
            enroll_out = gr.Markdown()
            # Allow empty choices and custom values initially; value None avoids warnings when empty
            name_list = gr.Dropdown(
                choices=list_enrollments(),
                label='Enrolled names',
                interactive=True,
                allow_custom_value=True,
                value=None,
            )
            enroll_btn.click(enroll, inputs=[name_in, enroll_img], outputs=[enroll_out, name_list])

        with gr.Tab('Verify (webcam/upload)'):
            select_name = gr.Dropdown(
                choices=list_enrollments(),
                label='Select name',
                interactive=True,
                allow_custom_value=True,
                value=None,
            )
            refresh_btn = gr.Button('Refresh names')
            refresh_btn.click(refresh_names, inputs=[], outputs=[select_name])
            thresh = gr.Slider(minimum=0.1, maximum=0.9, value=0.35, step=0.01, label='Threshold (cos similarity)')
            verify_img = gr.Image(type='pil', label='Webcam or upload', sources=['webcam', 'upload'])
            verify_btn = gr.Button('Verify')
            verify_msg = gr.Markdown()
            sim_num = gr.Number(label='Similarity', precision=4)
            log_file = gr.File(label='Attendance log (CSV)')
            verify_btn.click(verify, inputs=[select_name, verify_img, thresh], outputs=[verify_msg, sim_num, log_file])

            gr.Markdown('Or verify from file path (fallback if upload fails):')
            file_paths = gr.Files(label='Image files', file_types=['image'], type='filepath')
            verify_btn2 = gr.Button('Verify from file(s)')
            verify_btn2.click(verify_paths, inputs=[select_name, file_paths, thresh], outputs=[verify_msg, sim_num, log_file])

        gr.Markdown('Notes:\n- Uses InsightFace (buffalo_l) embeddings.\n- CPU by default; set GPU later if available.\n- Attendance is logged to attendance/logs.csv.')
    return demo


if __name__ == '__main__':
    ui = build_ui()
    # Allow overriding port via environment variable to avoid conflicts
    port_str = os.environ.get('GRADIO_SERVER_PORT', '7861')
    try:
        port = int(port_str)
    except ValueError:
        port = 7861
    ui.launch(server_name='127.0.0.1', server_port=port, inbrowser=True, show_error=True)
