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

# Model paths
MODEL_PATHS = {
    "Baseline (NPR.pth - 46.7%)": "NPR.pth",
    "Trained Model (100% ForenSynths)": "trained_model_forensynths_100acc/model_epoch_last.pth"
}

# Load models
print("\nLoading models...")
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            model = resnet50(num_classes=1)
            state_dict = torch.load(path, map_location='cpu')
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
def preprocess_image(image):
    """Preprocess image for model input"""
    if image is None:
        return None
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
