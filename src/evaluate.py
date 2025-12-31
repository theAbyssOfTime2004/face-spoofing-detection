"""
Evaluation script cho trained models
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import create_dataloader
from train_global import MiniFASNetV2
from train_local import DeepPixBiS


def evaluate_model(model, dataloader, device, model_type='global'):
    """Evaluate model trên test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            if model_type == 'global':
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
            else:  # local
                pixel_map, binary_logits = model(images)
                probs = torch.softmax(binary_logits, dim=1)
                _, preds = torch.max(binary_logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_metrics(y_true, y_pred, y_probs):
    """Print evaluation metrics"""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Fake  Real")
    print(f"Actual Fake    {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Real    {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # ROC curve metrics (simplified)
    real_probs = y_probs[:, 1]  # Probability of being real
    print(f"\nReal class probability:")
    print(f"  Mean: {real_probs.mean():.4f}")
    print(f"  Std:  {real_probs.std():.4f}")
    print(f"  Min:  {real_probs.min():.4f}")
    print(f"  Max:  {real_probs.max():.4f}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        print(f"\nConfusion matrix saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model-type', type=str, choices=['global', 'local'], 
                       required=True, help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--image-size', type=int, default=80, help='Input image size')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Plot confusion matrix')
    
    args = parser.parse_args()
    
    # Adjust image size based on model type
    if args.model_type == 'local':
        args.image_size = 224
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if args.model_type == 'global':
        model = MiniFASNetV2(num_classes=2)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    else:
        model = DeepPixBiS()
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {args.model_type} model from {args.checkpoint}")
    
    # Create test dataloader với Raw Crop từ Full Images
    test_loader = create_dataloader(
        args.data_dir, 'test',
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        augment=False,
        shuffle=False,
        context_expansion_scale=2.7,  # Match training config
        use_raw_crop=True,
        use_full_image_detection=True  # Detect từ full image (giống training)
    )
    
    print(f"\nEvaluating on test set ({len(test_loader.dataset)} images)...")
    
    # Evaluate
    preds, labels, probs = evaluate_model(model, test_loader, device, args.model_type)
    
    # Print metrics
    metrics = print_metrics(labels, preds, probs)
    
    # Plot confusion matrix
    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_path = os.path.join(args.output_dir, f'confusion_matrix_{args.model_type}.png')
        plot_confusion_matrix(metrics['confusion_matrix'], plot_path)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f'evaluation_{args.model_type}.txt')
    with open(results_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()

