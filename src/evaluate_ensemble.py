"""
Evaluation script cho Ensemble (Fusion) Liveness Detection
Sử dụng dataset.py để đảm bảo preprocessing giống hệt training pipeline
Tối ưu: Output 224x224 để tránh resize 2 lần (Local Branch cần 224x224, Global sẽ resize về 80x80)
"""
import os
import yaml
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import create_dataloader
from pipeline.liveness_ensemble import LivenessEnsemble
from pipeline.detection import SCRFDDetector


def load_config(config_path):
    """Load config từ YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['pipeline']


def evaluate_ensemble_with_dataloader(config, data_dir, skip_quality=True):
    """
    Evaluate ensemble sử dụng cùng preprocessing như training
    Tối ưu: Lấy raw crop 224x224 trực tiếp từ bbox_cache để tránh resize 2 lần
    """
    # Load ensemble và detector
    ensemble = LivenessEnsemble(config['liveness'])
    detector = SCRFDDetector(config['detection'])
    
    # Load bbox cache
    bbox_cache_path = Path(data_dir).parent / 'bbox_cache.pkl'
    bbox_cache = {}
    if bbox_cache_path.exists():
        import pickle
        with open(bbox_cache_path, 'rb') as f:
            bbox_cache = pickle.load(f)
        print(f"  ✓ Loaded bbox cache: {len(bbox_cache)} images")
    else:
        print("  ⚠ No bbox cache found, will detect on-the-fly (slower)")
    
    # Tạo dataloader để lấy image paths và labels
    test_loader = create_dataloader(
        data_dir, 'test',
        batch_size=32,
        image_size=(112, 112),  # Không dùng cho ensemble, chỉ để lấy paths
        augment=False,
        shuffle=False,
        context_expansion_scale=2.7,  # Match training config
        use_raw_crop=True,  # Match training config
        use_full_image_detection=True  # Match training config
    )
    
    print(f"\nEvaluating ensemble on {len(test_loader.dataset)} images...")
    print("  Using raw crop 224x224 from bbox_cache (tránh resize 2 lần)")
    
    all_predictions = []
    all_labels = []
    all_global_scores = []
    all_local_scores = []
    all_final_scores = []
    processed = 0
    failed = 0
    
    # Process từng batch
    for batch_idx, batch in enumerate(test_loader):
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processing batch {batch_idx + 1}/{len(test_loader)}... (failed: {failed})")
        
        labels = batch['label']   # Shape: [B]
        paths = batch['path']
        
        # Process từng image trong batch
        for i in range(len(paths)):
            image_path = paths[i]
            label = int(labels[i].item())
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    failed += 1
                    continue
                
                # Lấy raw crop 224x224 từ bbox_cache (tối ưu nhất)
                raw_face = None
                if image_path in bbox_cache:
                    cached_data = bbox_cache[image_path]
                    bbox = cached_data['bbox']
                    is_rotated = cached_data.get('rotated', False)
                    
                    # Rotate nếu cần
                    if is_rotated:
                        image = cv2.rotate(image, cv2.ROTATE_180)
                    
                    # Extract raw face crop 224x224 (match Local Branch input)
                    raw_face = detector.extract_raw_face(
                        image,  # BGR format
                        bbox,
                        output_size=(224, 224)  # Match Local Branch, Global sẽ resize về 80x80
                    )
                else:
                    # Fallback: Detect on-the-fly
                    faces = detector.detect(image)
                    if len(faces) == 0:
                        failed += 1
                        continue
                    
                    raw_face = detector.extract_raw_face(
                        image,
                        faces[0]['bbox'],
                        output_size=(224, 224)
                    )
                
                if raw_face is None:
                    failed += 1
                    continue
                
                # Ensemble prediction với 224x224 (tránh resize 2 lần)
                results = ensemble.predict(raw_face, frame_count=0)
                
                global_score = results.get('global_score', 0.0)
                local_score = results.get('local_score', 0.0)
                final_score = results.get('final_score', 0.0)
                is_real = results.get('is_real', False)
                
                all_global_scores.append(global_score)
                all_local_scores.append(local_score)
                all_final_scores.append(final_score)
                all_predictions.append(1 if is_real else 0)
                all_labels.append(label)
                processed += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                failed += 1
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
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'labels': all_labels,
        'predictions': all_predictions,
        'global_scores': np.array(all_global_scores),
        'local_scores': np.array(all_local_scores),
        'final_scores': np.array(all_final_scores),
        'processed': processed,
        'failed': failed
    }


def save_results(results, output_dir='results', config_path='config/config.yaml', data_dir='data'):
    """Save evaluation results to file"""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'evaluation_ensemble.txt')
    
    with open(results_path, 'w') as f:
        f.write("Ensemble Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total processed: {results['processed']}\n")
        f.write(f"Failed: {results['failed']}\n\n")
        f.write("Metrics:\n")
        f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall:    {results['recall']:.4f}\n")
        f.write(f"  F1-Score:  {results['f1']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"                Predicted\n")
        f.write(f"              Fake  Real\n")
        cm = results['confusion_matrix']
        f.write(f"Actual Fake    {cm[0][0]:4d}  {cm[0][1]:4d}\n")
        f.write(f"       Real    {cm[1][0]:4d}  {cm[1][1]:4d}\n")
    
    print(f"\nResults saved to {results_path}")


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Ensemble Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
    plt.close()


def plot_score_distributions(results, save_path=None):
    """Plot score distributions cho real và spoof"""
    labels = results['labels']
    global_scores = results['global_scores']
    local_scores = results['local_scores']
    final_scores = results['final_scores']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Global scores
    axes[0].hist(global_scores[labels == 0], bins=50, alpha=0.5, label='Spoof', color='red')
    axes[0].hist(global_scores[labels == 1], bins=50, alpha=0.5, label='Real', color='green')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Global Branch Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Local scores
    axes[1].hist(local_scores[labels == 0], bins=50, alpha=0.5, label='Spoof', color='red')
    axes[1].hist(local_scores[labels == 1], bins=50, alpha=0.5, label='Real', color='green')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Local Branch Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Final scores
    axes[2].hist(final_scores[labels == 0], bins=50, alpha=0.5, label='Spoof', color='red')
    axes[2].hist(final_scores[labels == 1], bins=50, alpha=0.5, label='Real', color='green')
    axes[2].set_xlabel('Score')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Final Ensemble Scores')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Score distributions saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble with dataset preprocessing')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--skip-quality', action='store_true', default=True,
                       help='Skip quality gate check')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                       help='Plot confusion matrix and score distributions')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("="*70)
    print("Ensemble Evaluation với Dataset Preprocessing")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Skip quality gate: {args.skip_quality}")
    print(f"Output directory: {args.output_dir}")
    print(f"Plotting: {'Enabled' if args.plot else 'Disabled'}")
    
    # Evaluate
    results = evaluate_ensemble_with_dataloader(config, args.data_dir, args.skip_quality)
    
    # Save results
    save_results(results, args.output_dir, args.config, args.data_dir)
    
    # Plot
    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, 'ensemble_confusion_matrix.png')
        plot_confusion_matrix(results['confusion_matrix'], cm_path)
        
        # Score distributions
        dist_path = os.path.join(args.output_dir, 'ensemble_score_distributions.png')
        plot_score_distributions(results, dist_path)
        
        print(f"Plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

