"""
Phân tích chi tiết Local Branch scores từ individual evaluation
So sánh với ensemble để hiểu tại sao mean thấp
"""
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from train_local import DeepPixBiS
from dataset import create_dataloader

def analyze_local_scores():
    """Phân tích Local Branch scores từ individual evaluation"""
    print("="*60)
    print("Phân tích Local Branch Scores (Individual Evaluation)")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = DeepPixBiS()
    checkpoint = torch.load('checkpoints/best_local.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test dataloader (giống individual evaluation)
    print("Loading test dataloader...")
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
    
    print(f"Test set: {len(test_loader.dataset)} images\n")
    
    # Evaluate
    all_scores = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_loader)}...")
            
            images = batch['image']
            labels = batch['label']
            
            pixel_map, binary_logits = model(images)
            probs = torch.softmax(binary_logits, dim=1)
            real_probs = probs[:, 1].cpu().numpy()
            
            all_scores.extend(real_probs)
            all_labels.extend(labels.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    real_mask = all_labels == 1
    spoof_mask = all_labels == 0
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total images: {len(all_scores)}")
    print(f"  Real: {real_mask.sum()}")
    print(f"  Spoof: {spoof_mask.sum()}")
    
    print(f"\nReal Images Statistics:")
    print(f"  Mean: {all_scores[real_mask].mean():.4f}")
    print(f"  Median: {np.median(all_scores[real_mask]):.4f}")
    print(f"  Std: {all_scores[real_mask].std():.4f}")
    print(f"  Min: {all_scores[real_mask].min():.4f}")
    print(f"  Max: {all_scores[real_mask].max():.4f}")
    
    print(f"\nSpoof Images Statistics:")
    print(f"  Mean: {all_scores[spoof_mask].mean():.4f}")
    print(f"  Median: {np.median(all_scores[spoof_mask]):.4f}")
    print(f"  Std: {all_scores[spoof_mask].std():.4f}")
    print(f"  Min: {all_scores[spoof_mask].min():.4f}")
    print(f"  Max: {all_scores[spoof_mask].max():.4f}")
    
    # Distribution analysis
    print(f"\nReal Images Distribution:")
    print(f"  Score < 0.1: {(all_scores[real_mask] < 0.1).sum()} ({(all_scores[real_mask] < 0.1).sum() / real_mask.sum() * 100:.1f}%)")
    print(f"  Score < 0.3: {(all_scores[real_mask] < 0.3).sum()} ({(all_scores[real_mask] < 0.3).sum() / real_mask.sum() * 100:.1f}%)")
    print(f"  Score < 0.5: {(all_scores[real_mask] < 0.5).sum()} ({(all_scores[real_mask] < 0.5).sum() / real_mask.sum() * 100:.1f}%)")
    print(f"  Score > 0.7: {(all_scores[real_mask] > 0.7).sum()} ({(all_scores[real_mask] > 0.7).sum() / real_mask.sum() * 100:.1f}%)")
    print(f"  Score > 0.9: {(all_scores[real_mask] > 0.9).sum()} ({(all_scores[real_mask] > 0.9).sum() / real_mask.sum() * 100:.1f}%)")
    
    print("\n" + "="*60)
    print("PHÂN TÍCH")
    print("="*60)
    print("Nếu mean thấp (~0.25) nhưng accuracy cao (85.88%):")
    print("  → Model có thể đang dùng threshold thấp trong evaluation")
    print("  → Hoặc có nhiều easy cases (scores rất cao) và hard cases (scores thấp)")
    print("  → Mean bị kéo xuống bởi hard cases")
    print("\nNếu có nhiều real images với score < 0.3:")
    print("  → Model không tự tin với một số loại real images")
    print("  → Có thể cần retrain hoặc data augmentation")

if __name__ == '__main__':
    analyze_local_scores()

