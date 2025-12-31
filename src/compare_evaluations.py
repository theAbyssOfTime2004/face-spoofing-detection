"""
So sánh Local Branch scores giữa Individual và Ensemble evaluation
Tìm nguyên nhân tại sao mean thấp trong ensemble
"""
import cv2
import numpy as np
import yaml
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.detection import SCRFDDetector
from pipeline.liveness_ensemble import LivenessEnsemble
from train_local import DeepPixBiS
from dataset import create_dataloader

def compare_evaluations():
    """So sánh 2 cách evaluation"""
    print("="*60)
    print("So sánh Individual vs Ensemble Evaluation")
    print("="*60)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
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
    
    # Load test dataloader (individual evaluation)
    print("\n1. Individual Evaluation (dataloader)...")
    test_loader = create_dataloader(
        'data', 'test',
        batch_size=32,
        image_size=(224, 224),
        augment=False,
        shuffle=False,
        context_expansion_scale=2.7,
        use_raw_crop=True,
        use_full_image_detection=True
    )
    
    individual_scores = []
    individual_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image']
            labels = batch['label']
            
            pixel_map, binary_logits = pytorch_model(images)
            probs = torch.softmax(binary_logits, dim=1)
            real_probs = probs[:, 1].cpu().numpy()
            
            individual_scores.extend(real_probs)
            individual_labels.extend(labels.cpu().numpy())
    
    individual_scores = np.array(individual_scores)
    individual_labels = np.array(individual_labels)
    
    # Ensemble evaluation (giống evaluate_ensemble.py)
    print("2. Ensemble Evaluation (extract_raw_face)...")
    data_dir = Path('data/test')
    real_images = list((data_dir / 'normal').glob('*.jpg'))
    spoof_images = list((data_dir / 'spoof').glob('*.jpg'))
    
    ensemble_real_scores = []
    ensemble_spoof_scores = []
    failed = 0
    
    for img_path in real_images + spoof_images:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                failed += 1
                continue
            
            faces = detector.detect(image)
            if len(faces) == 0:
                failed += 1
                continue
            
            face_data = faces[0]
            raw_face = detector.extract_raw_face(
                image,
                face_data['bbox'],
                output_size=(224, 224)  # Match với Local Branch input size
            )
            
            if raw_face is None:
                failed += 1
                continue
            
            results = ensemble.predict(raw_face, frame_count=0)
            local_score = results.get('local_score', 0.0)
            
            if img_path.parent.name == 'normal':
                ensemble_real_scores.append(local_score)
            else:
                ensemble_spoof_scores.append(local_score)
        
        except Exception as e:
            failed += 1
            continue
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    real_mask = individual_labels == 1
    
    print(f"\nIndividual Evaluation (dataloader):")
    print(f"  Real mean: {individual_scores[real_mask].mean():.4f}")
    print(f"  Real median: {np.median(individual_scores[real_mask]):.4f}")
    print(f"  Real count: {real_mask.sum()}")
    
    print(f"\nEnsemble Evaluation (extract_raw_face):")
    print(f"  Real mean: {np.mean(ensemble_real_scores):.4f}")
    print(f"  Real median: {np.median(ensemble_real_scores):.4f}")
    print(f"  Real count: {len(ensemble_real_scores)}")
    print(f"  Failed: {failed}")
    
    print(f"\nDifference:")
    print(f"  Mean diff: {abs(individual_scores[real_mask].mean() - np.mean(ensemble_real_scores)):.4f}")
    
    print("\n" + "="*60)
    print("PHÂN TÍCH")
    print("="*60)
    if abs(individual_scores[real_mask].mean() - np.mean(ensemble_real_scores)) > 0.1:
        print("⚠️  CÓ SỰ KHÁC BIỆT LỚN!")
        print("  → Preprocessing khác nhau giữa dataloader và ensemble")
        print("  → Có thể do:")
        print("    1. Output size khác nhau (224 vs 112)")
        print("    2. Resize interpolation khác nhau")
        print("    3. Normalization khác nhau")
    else:
        print("✓ Không có sự khác biệt lớn")
        print("  → Vấn đề có thể do test set khác nhau")

if __name__ == '__main__':
    compare_evaluations()

