"""
Debug: Tại sao Local Branch trong ensemble có mean thấp
So sánh preprocessing giữa individual evaluation và ensemble
"""
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from pipeline.detection import SCRFDDetector
from pipeline.liveness_ensemble import LivenessEnsemble
from train_local import DeepPixBiS

def debug_local_ensemble(config_path='config/config.yaml', num_samples=10):
    """So sánh Local Branch output giữa individual và ensemble"""
    print("="*60)
    print("Debug: Local Branch trong Ensemble")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pipeline_config = config['pipeline']
    
    # Initialize
    detector = SCRFDDetector(pipeline_config['detection'])
    ensemble = LivenessEnsemble(pipeline_config['liveness'])
    
    # Load PyTorch model
    pytorch_model = DeepPixBiS()
    checkpoint = torch.load('checkpoints/best_local.pth', map_location='cpu')
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()
    
    # Load test images
    data_dir = Path('data/test')
    real_images = list((data_dir / 'normal').glob('*.jpg'))[:num_samples]
    
    print(f"\nTesting {len(real_images)} real images...\n")
    
    individual_scores = []
    ensemble_scores = []
    
    for img_path in real_images:
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Detect face
            faces = detector.detect(image)
            if len(faces) == 0:
                continue
            
            face_data = faces[0]
            raw_face = detector.extract_raw_face(
                image,
                face_data['bbox'],
                output_size=(112, 112)  # Ensemble dùng 112x112
            )
            
            if raw_face is None:
                continue
            
            # ===== ENSEMBLE PATH =====
            # Ensemble preprocessing: raw_face (112x112) → resize to 224x224
            face_rgb_ensemble = cv2.cvtColor(raw_face, cv2.COLOR_BGR2RGB)
            face_resized_ensemble = cv2.resize(face_rgb_ensemble, (224, 224))
            face_normalized_ensemble = (face_resized_ensemble.astype(np.float32) / 255.0 - 0.5) / 0.5
            face_input_ensemble = np.expand_dims(face_normalized_ensemble.transpose(2, 0, 1), axis=0)
            
            # Ensemble inference
            local_score_ensemble, _ = ensemble.local_branch.predict(raw_face)
            ensemble_scores.append(local_score_ensemble)
            
            # ===== INDIVIDUAL PATH (giống evaluate.py) =====
            # Individual preprocessing: từ dataset (đã crop và resize)
            # Giả sử raw_face là input, resize về 224x224
            face_rgb_individual = cv2.cvtColor(raw_face, cv2.COLOR_BGR2RGB)
            face_resized_individual = cv2.resize(face_rgb_individual, (224, 224))
            face_normalized_individual = (face_resized_individual.astype(np.float32) / 255.0 - 0.5) / 0.5
            face_input_individual = np.expand_dims(face_normalized_individual.transpose(2, 0, 1), axis=0)
            
            # PyTorch inference (giống individual evaluation)
            with torch.no_grad():
                face_tensor = torch.from_numpy(face_input_individual)
                pixel_map, binary_logits = pytorch_model(face_tensor)
                probs = torch.softmax(binary_logits, dim=1)
                individual_score = float(probs[0][1].item())
            
            individual_scores.append(individual_score)
            
            print(f"Individual: {individual_score:.4f} | Ensemble: {local_score_ensemble:.4f} | Diff: {abs(individual_score - local_score_ensemble):.4f}")
        
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Summary
    if individual_scores and ensemble_scores:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Individual mean: {np.mean(individual_scores):.4f}")
        print(f"Ensemble mean: {np.mean(ensemble_scores):.4f}")
        print(f"Difference: {abs(np.mean(individual_scores) - np.mean(ensemble_scores)):.4f}")
        
        print("\n" + "="*60)
        print("PHÂN TÍCH")
        print("="*60)
        if abs(np.mean(individual_scores) - np.mean(ensemble_scores)) < 0.01:
            print("✓ Individual và Ensemble giống nhau")
            print("  → Vấn đề không phải ở preprocessing")
            print("  → Có thể do test set khác nhau hoặc sampling")
        else:
            print("⚠️ Individual và Ensemble khác nhau")
            print("  → Có vấn đề với preprocessing hoặc model loading")
            print(f"  → Individual cao hơn: {np.mean(individual_scores) > np.mean(ensemble_scores)}")

if __name__ == '__main__':
    debug_local_ensemble()

