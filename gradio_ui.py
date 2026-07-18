"""
Gradio UI for NPR Deepfake Detection Model
Test your trained model with drag-and-drop interface
"""
import gradio as gr
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from networks.resnet import resnet50
import os

print("=" * 70)
print("NPR DEEPFAKE DETECTION - GRADIO UI")
print("=" * 70)

# Model paths (override via env vars when running in Docker)
MODEL_PATHS = {
    "Baseline (NPR.pth)": os.environ.get(
        "NPR_BASELINE_PATH", "NPR.pth"),
    "Trained Model (ForenSynths)": os.environ.get(
        "NPR_TRAINED_PATH", "trained_model_forensynths_100acc/model_epoch_last.pth")
}

# Load models
print("\nLoading models...")
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            model = resnet50(num_classes=1)
            state_dict = torch.load(path, map_location='cpu')
            # Checkpoint may be wrapped in {'model': ...} and/or saved from DataParallel ('module.' prefix)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()
            models[name] = model
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [ERROR] {name}: {str(e)}")
    else:
        print(f"  [WARNING] {name}: File not found")

if not models:
    print("\n[ERROR] No models loaded! Please check model paths.")
    exit(1)

print(f"\nLoaded {len(models)} model(s)")
print("=" * 70)

# Image preprocessing
# NPR detects up-sampling artifacts (x - interpolate(x, 0.5)); any resize
# resamples the whole image and erases that signal, so the image must be
# tiled up to 224 if too small, then center-cropped — never resized.
def translate_duplicate(image, crop_size=224):
    """Tile small images so CenterCrop never pads/upscales"""
    if min(image.size) < crop_size:
        width, height = image.size
        tiles_x = int(np.ceil(crop_size / width))
        tiles_y = int(np.ceil(crop_size / height))
        new_img = Image.new('RGB', (width * tiles_x, height * tiles_y))
        for i in range(tiles_x):
            for j in range(tiles_y):
                new_img.paste(image, (i * width, j * height))
        return new_img
    return image

def preprocess_image(image):
    """Preprocess image for model input"""
    if image is None:
        return None

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Match the training/eval pipeline: center crop 224, NO resize
    transform = transforms.Compose([
        transforms.Lambda(lambda img: translate_duplicate(img, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return transform(image).unsqueeze(0)

def predict(image, model_name):
    """Run prediction on uploaded image"""
    if image is None:
        return {
            "Error": "Please upload an image",
            "Real": 0.0,
            "Fake": 0.0
        }
    
    if model_name not in models:
        return {
            "Error": f"Model '{model_name}' not loaded",
            "Real": 0.0,
            "Fake": 0.0
        }
    
    try:
        # Preprocess
        input_tensor = preprocess_image(image)
        
        # Inference
        model = models[model_name]
        with torch.no_grad():
            output = model(input_tensor)
            prob_fake = torch.sigmoid(output).item()
            prob_real = 1 - prob_fake
        
        # Determine result - return dict with float values for Gradio Label
        result = {
            "REAL": float(prob_real),
            "FAKE": float(prob_fake)
        }
        
        return result
        
    except Exception as e:
        return {
            "Error": str(e),
            "Real": 0.0,
            "Fake": 0.0
        }

# Create Gradio interface
with gr.Blocks(title="NPR Deepfake Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # NPR Deepfake Detection
    
    Upload an image to detect if it's **REAL** or **FAKE** (AI-generated)
    
    ### Model Performance:
    - **Baseline (NPR.pth)**: 46.7% on ForenSynths
    - **Trained Model**: 100% on ForenSynths, 91.84% on UniversalFakeDetect, 88.9% on 8-GAN Benchmark
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            input_image = gr.Image(
                type="pil",
                label="Upload Image",
                height=400
            )
            
            model_selector = gr.Dropdown(
                choices=list(models.keys()),
                value=list(models.keys())[0],
                label="Select Model",
                interactive=True
            )
            
            predict_btn = gr.Button("Analyze Image", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            # Output
            output_result = gr.Label(
                label="Detection Result",
                num_top_classes=3
            )
    
    gr.Markdown("""
    ---
    ### Notes:
    - **Real**: Original photograph or unmodified image
    - **Fake**: AI-generated image (GAN, Diffusion, etc.)
    - Model trained on ForenSynths dataset with 30 epochs
    - Best performance on: ProGAN (99.6%), StyleGAN (96.2%), StarGAN (99.7%)
    - Lower performance on: DeepFake (68.1%), GauGAN (80.2%)
    
    ### Datasets Tested:
    | Dataset | Accuracy | Samples |
    |---------|----------|---------|
    | ForenSynths | 100% | 1,600 |
    | UniversalFakeDetect | 91.84% | 16,000 |
    | GANGen-Detection | 64.37% | 36,000 |
    """)
    
    # Connect button
    predict_btn.click(
        fn=predict,
        inputs=[input_image, model_selector],
        outputs=output_result
    )

if __name__ == "__main__":
    print("\nStarting Gradio interface...")
    print("Local URL: http://127.0.0.1:7860")
    print("Public URL: Will be generated if share=True")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )
