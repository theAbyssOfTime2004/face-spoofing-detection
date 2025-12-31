"""
Evaluation script cho Ensemble (Fusion) Liveness Detection
Đánh giá performance của cả Global + Local branches kết hợp
"""
import os
import yaml
import numpy as np
import cv2
from pathlib import Path
import argparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Chạy inference ensemble trên tất cả ảnh và thu thập scores"""
    all_scores = []
    all_labels = []
    all_global_scores = []
    all_local_scores = []
    all_final_scores = []
    all_predictions = []
    
    print(f"\nProcessing {len(images)} images...")
    processed = 0
    failed = 0
    
    for i, img_path in enumerate(images):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(images)} images... (failed: {failed})")
        
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                failed += 1
                continue
            
            # Quality gate (skip nếu cần)
            if not skip_quality:
                quality_result = quality_gate.validate(image)
                if not quality_result['passed']:
                    failed += 1
                    continue  # Skip ảnh không đạt quality
            
            # Detect face
            faces = detector.detect(image)
            if len(faces) == 0:
                failed += 1
                continue  # Skip nếu không detect được face
            
            # Extract raw face crop (không alignment)
            # QUAN TRỌNG: Dùng 224x224 để match với Local Branch input size
            # Không dùng 112x112 rồi resize lại (làm mất thông tin)
            face_data = faces[0]
            raw_face = detector.extract_raw_face(
                image,
                face_data['bbox'],
                output_size=(224, 224)  # Match với Local Branch input size
            )
            
            if raw_face is None:
                failed += 1
                continue
            
            # Ensemble prediction
            results = ensemble.predict(raw_face, frame_count=0)
            
            # Collect scores
            global_score = results.get('global_score', 0.0)
            local_score = results.get('local_score', 0.0)
            final_score = results.get('final_score', 0.0)
            is_real = results.get('is_real', False)
            
            all_global_scores.append(global_score)
            all_local_scores.append(local_score)
            all_final_scores.append(final_score)
            all_predictions.append(1 if is_real else 0)
            all_labels.append(labels[i])
            processed += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            failed += 1
            continue
    
    print(f"\nSuccessfully processed {processed} images (failed: {failed})")
    
    return {
        'labels': np.array(all_labels),
        'predictions': np.array(all_predictions),
        'global_scores': np.array(all_global_scores),
        'local_scores': np.array(all_local_scores),
        'final_scores': np.array(all_final_scores),
        'processed': processed,
        'failed': failed
    }


def print_metrics(results):
    """Print evaluation metrics"""
    labels = results['labels']
    predictions = results['predictions']
    global_scores = results['global_scores']
    local_scores = results['local_scores']
    final_scores = results['final_scores']
    
    # Calculate metrics
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION RESULTS")
    print("="*70)
    print(f"Total images processed: {results['processed']}")
    print(f"Failed: {results['failed']}")
    print(f"\nOverall Metrics:")
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
    real_mask = labels == 1
    spoof_mask = labels == 0
    
    print(f"\nScore Distributions:")
    print(f"  Global Branch:")
    print(f"    Real:  Mean={global_scores[real_mask].mean():.4f}, Std={global_scores[real_mask].std():.4f}")
    print(f"    Spoof: Mean={global_scores[spoof_mask].mean():.4f}, Std={global_scores[spoof_mask].std():.4f}")
    
    print(f"  Local Branch:")
    print(f"    Real:  Mean={local_scores[real_mask].mean():.4f}, Std={local_scores[real_mask].std():.4f}")
    print(f"    Spoof: Mean={local_scores[spoof_mask].mean():.4f}, Std={local_scores[spoof_mask].std():.4f}")
    
    print(f"  Final Score (Ensemble):")
    print(f"    Real:  Mean={final_scores[real_mask].mean():.4f}, Std={final_scores[real_mask].std():.4f}")
    print(f"    Spoof: Mean={final_scores[spoof_mask].mean():.4f}, Std={final_scores[spoof_mask].std():.4f}")
    print(f"    Range: Min={final_scores.min():.4f}, Max={final_scores.max():.4f}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'results': results
    }


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
    parser = argparse.ArgumentParser(description='Evaluate Ensemble Liveness Detection')
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
    print("Ensemble Liveness Detection - Evaluation")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Skip quality gate: {args.skip_quality}")
    
    # Initialize components
    print("\nInitializing components...")
    quality_gate = QualityGate(config['quality_gate']) if not args.skip_quality else None
    detector = SCRFDDetector(config['detection'])
    ensemble = LivenessEnsemble(config['liveness'])
    print("  ✓ All components initialized")
    
    # Load test images
    print(f"\nLoading test images from {args.data_dir}...")
    images, labels = load_test_images(args.data_dir)
    print(f"  Found {len(images)} images ({np.sum(labels)} real, {np.sum(1-labels)} spoof)")
    
    # Evaluate ensemble
    results = evaluate_ensemble(
        images, labels, quality_gate, detector, ensemble,
        skip_quality=args.skip_quality
    )
    
    # Print metrics
    metrics = print_metrics(results)
    
    # Plot
    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, 'ensemble_confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
        
        # Score distributions
        dist_path = os.path.join(args.output_dir, 'ensemble_score_distributions.png')
        plot_score_distributions(results, dist_path)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'evaluation_ensemble.txt')
    with open(results_path, 'w') as f:
        f.write("Ensemble Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Total processed: {results['processed']}\n")
        f.write(f"Failed: {results['failed']}\n\n")
        f.write("Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"                Predicted\n")
        f.write(f"              Fake  Real\n")
        f.write(f"Actual Fake    {metrics['confusion_matrix'][0][0]:4d}  {metrics['confusion_matrix'][0][1]:4d}\n")
        f.write(f"       Real    {metrics['confusion_matrix'][1][0]:4d}  {metrics['confusion_matrix'][1][1]:4d}\n")
    
    print(f"\nResults saved to {results_path}")
    if args.plot:
        print(f"Plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

