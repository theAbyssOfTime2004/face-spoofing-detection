"""
Evaluate Ensemble với cùng preprocessing như individual evaluation
Sử dụng dataset.py để đảm bảo consistency
"""
import os
import yaml
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import torch

from dataset import create_dataloader
from pipeline.liveness_ensemble import LivenessEnsemble


def load_config(config_path):
    """Load config từ YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['pipeline']


def evaluate_ensemble_with_dataloader(config, data_dir, skip_quality=True):
    """
    Evaluate ensemble sử dụng cùng dataloader như individual evaluation
    Đảm bảo preprocessing giống hệt nhau
    """
    # Load ensemble
    ensemble = LivenessEnsemble(config['liveness'])
    
    # Tạo dataloader giống hệt như evaluate.py (cùng config với training)
    test_loader = create_dataloader(
        data_dir, 'test',
        batch_size=32,
        image_size=(112, 112),  # Size cho ensemble
        augment=False,
        shuffle=False,
        context_expansion_scale=2.7,  # Match training config
        use_raw_crop=True,  # Match training config
        use_full_image_detection=True  # Match training config
    )
    
    print(f"\nEvaluating ensemble on {len(test_loader.dataset)} images...")
    print("  Using same preprocessing as individual evaluation (face crops from dataset)")
    
    all_predictions = []
    all_labels = []
    all_global_scores = []
    all_local_scores = []
    all_final_scores = []
    
    # Process từng batch
    for batch_idx, batch in enumerate(test_loader):
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processing batch {batch_idx + 1}/{len(test_loader)}...")
        
        images = batch['image']  # Shape: [B, C, H, W]
        labels = batch['label']   # Shape: [B]
        paths = batch['path']
        
        # Process từng image trong batch
        for i in range(images.shape[0]):
            # Convert từ tensor về numpy
            img_tensor = images[i]  # [C, H, W]
            img_np = img_tensor.numpy().transpose(1, 2, 0)  # [H, W, C]
            
            # Denormalize: từ [-1, 1] về [0, 255]
            img_np = ((img_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
            
            # Convert RGB to BGR (vì ensemble expect BGR)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Resize về size phù hợp cho ensemble (112x112 đã đúng)
            # Global branch cần 80x80, Local branch cần 224x224
            # Nhưng ensemble sẽ tự resize trong predict()
            
            # Ensemble prediction
            try:
                results = ensemble.predict(img_bgr, frame_count=0)
                
                global_score = results.get('global_score', 0.0)
                local_score = results.get('local_score', 0.0)
                final_score = results.get('final_score', 0.0)
                is_real = results.get('is_real', False)
                
                all_global_scores.append(global_score)
                all_local_scores.append(local_score)
                all_final_scores.append(final_score)
                all_predictions.append(1 if is_real else 0)
                all_labels.append(int(labels[i].item()))
                
            except Exception as e:
                print(f"Error processing {paths[i]}: {e}")
                continue
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION RESULTS (with dataset preprocessing)")
    print("="*60)
    print(f"Total processed: {len(all_labels)}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Fake  Real")
    print(f"Actual Fake    {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Real    {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Score distributions
    real_mask = all_labels == 1
    spoof_mask = all_labels == 0
    
    print(f"\nScore Distributions:")
    print(f"  Global Branch:")
    print(f"    Real:  Mean={np.array(all_global_scores)[real_mask].mean():.4f}, Std={np.array(all_global_scores)[real_mask].std():.4f}")
    print(f"    Spoof: Mean={np.array(all_global_scores)[spoof_mask].mean():.4f}, Std={np.array(all_global_scores)[spoof_mask].std():.4f}")
    print(f"  Local Branch:")
    print(f"    Real:  Mean={np.array(all_local_scores)[real_mask].mean():.4f}, Std={np.array(all_local_scores)[real_mask].std():.4f}")
    print(f"    Spoof: Mean={np.array(all_local_scores)[spoof_mask].mean():.4f}, Std={np.array(all_local_scores)[spoof_mask].std():.4f}")
    print(f"  Final Score (Ensemble):")
    print(f"    Real:  Mean={np.array(all_final_scores)[real_mask].mean():.4f}, Std={np.array(all_final_scores)[real_mask].std():.4f}")
    print(f"    Spoof: Mean={np.array(all_final_scores)[spoof_mask].mean():.4f}, Std={np.array(all_final_scores)[spoof_mask].std():.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = 'results/evaluation_ensemble_fixed.txt'
    with open(results_path, 'w') as f:
        f.write("Ensemble Evaluation Results (with dataset preprocessing)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total processed: {len(all_labels)}\n\n")
        f.write(f"Metrics:\n")
        f.write(f"  Accuracy:  {acc:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"                Predicted\n")
        f.write(f"              Fake  Real\n")
        f.write(f"Actual Fake    {cm[0][0]:4d}  {cm[0][1]:4d}\n")
        f.write(f"       Real    {cm[1][0]:4d}  {cm[1][1]:4d}\n")
    
    print(f"\nResults saved to {results_path}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble with dataset preprocessing')
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
    print("Ensemble Evaluation với Dataset Preprocessing")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Skip quality gate: {args.skip_quality}")
    
    # Evaluate
    evaluate_ensemble_with_dataloader(config, args.data_dir, args.skip_quality)


if __name__ == '__main__':
    main()

