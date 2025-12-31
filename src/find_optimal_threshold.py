"""
Script để tìm threshold tối ưu cho ensemble liveness detection
"""
import os
import yaml
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
import argparse

from pipeline.quality_gate import QualityGate
from pipeline.detection import SCRFDDetector
from pipeline.liveness_ensemble import LivenessEnsemble


def load_config(config_path):
    """Load config từ YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['pipeline']


def load_test_images(data_dir):
    """Load tất cả ảnh từ test set"""
    test_dir = Path(data_dir) / 'test'
    real_images = list((test_dir / 'normal').glob('*.jpg'))
    spoof_images = list((test_dir / 'spoof').glob('*.jpg'))
    
    images = []
    labels = []  # 1 = real, 0 = spoof
    
    for img_path in real_images:
        images.append(str(img_path))
        labels.append(1)
    
    for img_path in spoof_images:
        images.append(str(img_path))
        labels.append(0)
    
    return images, np.array(labels)


def evaluate_ensemble(images, labels, quality_gate, detector, ensemble, skip_quality=True):
    """Chạy inference trên tất cả ảnh và thu thập scores"""
    scores = []
    valid_labels = []
    
    print(f"\nProcessing {len(images)} images...")
    for i, img_path in enumerate(images):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(images)} images...")
        
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Quality gate (skip nếu cần)
            if not skip_quality:
                quality_result = quality_gate.validate(image)
                if not quality_result['passed']:
                    continue  # Skip ảnh không đạt quality
            
            # Detect face
            faces = detector.detect(image)
            if len(faces) == 0:
                continue  # Skip nếu không detect được face
            
            # Extract raw face crop (không alignment) - cho Liveness Model
            face_data = faces[0]
            raw_face = detector.extract_raw_face(
                image,
                face_data['bbox'],
                output_size=(224, 224)  # Match với Local Branch input size (tránh resize 2 lần)
            )
            if raw_face is None:
                continue
            
            # Ensemble prediction với raw crop
            results = ensemble.predict(raw_face)
            final_score = results['final_score']
            
            scores.append(final_score)
            valid_labels.append(labels[i])
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return np.array(scores), np.array(valid_labels)


def find_optimal_threshold(y_true, y_scores):
    """Tìm threshold tối ưu dựa trên F1-score"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1, thresholds, f1_scores


def main():
    parser = argparse.ArgumentParser(description='Find optimal threshold for ensemble')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--skip-quality', action='store_true', default=True,
                       help='Skip quality gate check')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("="*60)
    print("Finding Optimal Threshold for Ensemble Liveness Detection")
    print("="*60)
    
    # Initialize components
    print("\nInitializing components...")
    quality_gate = QualityGate(config['quality_gate'])
    detector = SCRFDDetector(config['detection'])
    ensemble = LivenessEnsemble(config['liveness'])
    print("  ✓ All components initialized")
    
    # Load test images
    print(f"\nLoading test images from {args.data_dir}...")
    images, labels = load_test_images(args.data_dir)
    print(f"  Found {len(images)} images ({np.sum(labels)} real, {np.sum(1-labels)} spoof)")
    
    # Evaluate ensemble
    scores, valid_labels = evaluate_ensemble(
        images, labels, quality_gate, detector, ensemble, 
        skip_quality=args.skip_quality
    )
    
    print(f"\nSuccessfully processed {len(scores)} images")
    print(f"  Real: {np.sum(valid_labels)}, Spoof: {np.sum(1-valid_labels)}")
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    best_threshold, best_f1, thresholds, f1_scores = find_optimal_threshold(valid_labels, scores)
    
    # Calculate metrics at optimal threshold
    y_pred = (scores > best_threshold).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    acc = accuracy_score(valid_labels, y_pred)
    precision = precision_score(valid_labels, y_pred)
    recall = recall_score(valid_labels, y_pred)
    
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLD RESULTS")
    print("="*60)
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"F1-Score:          {best_f1:.4f}")
    print(f"Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    
    # Score distribution
    real_scores = scores[valid_labels == 1]
    spoof_scores = scores[valid_labels == 0]
    
    print(f"\nScore Distribution:")
    print(f"  Real scores:  Mean={real_scores.mean():.4f}, Std={real_scores.std():.4f}")
    print(f"                Min={real_scores.min():.4f}, Max={real_scores.max():.4f}")
    print(f"  Spoof scores: Mean={spoof_scores.mean():.4f}, Std={spoof_scores.std():.4f}")
    print(f"                Min={spoof_scores.min():.4f}, Max={spoof_scores.max():.4f}")
    
    # Recommendations
    print(f"\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(f"Update config/config.yaml:")
    print(f"  final_threshold: {best_threshold:.3f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = 'results/optimal_threshold.txt'
    with open(results_path, 'w') as f:
        f.write("Optimal Threshold Analysis\n")
        f.write("="*60 + "\n\n")
        f.write(f"Optimal Threshold: {best_threshold:.4f}\n")
        f.write(f"F1-Score:          {best_f1:.4f}\n")
        f.write(f"Accuracy:          {acc:.4f}\n")
        f.write(f"Precision:         {precision:.4f}\n")
        f.write(f"Recall:            {recall:.4f}\n\n")
        f.write(f"Real scores:  Mean={real_scores.mean():.4f}, Std={real_scores.std():.4f}\n")
        f.write(f"Spoof scores: Mean={spoof_scores.mean():.4f}, Std={spoof_scores.std():.4f}\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()

