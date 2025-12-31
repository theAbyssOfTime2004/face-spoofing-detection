"""
Test binary_logits output từ Local Branch ONNX model
So sánh với PyTorch model để xác định vấn đề
"""
import torch
import cv2
import numpy as np
import yaml
from pathlib import Path
from pipeline.detection import SCRFDDetector

# Import models
from train_local import DeepPixBiS

def test_binary_logits_comparison(config_path='config/config.yaml', num_samples=10):
    """So sánh binary_logits từ PyTorch model và ONNX model"""
    print("="*60)
    print("So sánh Binary Logits: PyTorch vs ONNX")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pipeline_config = config['pipeline']
    
    # Initialize detector
    detector = SCRFDDetector(pipeline_config['detection'])
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    pytorch_model = DeepPixBiS()
    checkpoint = torch.load('checkpoints/best_local.pth', map_location='cpu')
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()
    
    # Load ONNX model
    print("Loading ONNX model...")
    try:
        import onnxruntime as ort
        onnx_session = ort.InferenceSession('models/deeppixbis.onnx')
        onnx_input_name = onnx_session.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading ONNX: {e}")
        return
    
    # Load test images
    data_dir = Path('data/test')
    real_images = list((data_dir / 'normal').glob('*.jpg'))[:num_samples]
    
    print(f"\nTesting {len(real_images)} real images...\n")
    
    pytorch_scores = []
    onnx_scores = []
    
    for img_path in real_images:
        try:
            # Load and preprocess
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            faces = detector.detect(image)
            if len(faces) == 0:
                continue
            
            face_data = faces[0]
            raw_face = detector.extract_raw_face(
                image,
                face_data['bbox'],
                output_size=(224, 224)
            )
            
            if raw_face is None:
                continue
            
            # Preprocess
            face_rgb = cv2.cvtColor(raw_face, cv2.COLOR_BGR2RGB)
            face_normalized = (face_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
            face_input = np.expand_dims(face_normalized.transpose(2, 0, 1), axis=0)
            
            # PyTorch inference
            with torch.no_grad():
                face_tensor = torch.from_numpy(face_input)
                pixel_map, binary_logits = pytorch_model(face_tensor)
                
                # Apply softmax
                probs = torch.softmax(binary_logits, dim=1)
                pytorch_score = float(probs[0][1].item())  # real probability
            
            # ONNX inference
            onnx_outputs = onnx_session.run(None, {onnx_input_name: face_input})
            onnx_logits = onnx_outputs[0][0]  # [fake_logit, real_logit]
            
            # Apply softmax
            max_logit = np.max(onnx_logits)
            logits_shifted = onnx_logits - max_logit
            exp_logits = np.exp(logits_shifted)
            onnx_probs = exp_logits / np.sum(exp_logits)
            onnx_score = float(onnx_probs[1])  # real probability
            
            pytorch_scores.append(pytorch_score)
            onnx_scores.append(onnx_score)
            
            print(f"PyTorch: {pytorch_score:.4f} | ONNX: {onnx_score:.4f} | Diff: {abs(pytorch_score - onnx_score):.4f}")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Summary
    if pytorch_scores and onnx_scores:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"PyTorch mean: {np.mean(pytorch_scores):.4f}")
        print(f"ONNX mean: {np.mean(onnx_scores):.4f}")
        print(f"Difference: {abs(np.mean(pytorch_scores) - np.mean(onnx_scores)):.4f}")
        
        print("\n" + "="*60)
        print("PHÂN TÍCH")
        print("="*60)
        print("Nếu PyTorch và ONNX giống nhau:")
        print("  → Vấn đề không phải do ONNX conversion")
        print("  → Vấn đề là model training (binary_logits không được train tốt)")
        print("\nNếu PyTorch cao hơn ONNX:")
        print("  → Có thể có vấn đề với ONNX conversion")
        print("  → Hoặc preprocessing khác nhau")

if __name__ == '__main__':
    test_binary_logits_comparison()

