"""
Script để phân tích dataset
"""
import argparse
from dataset import analyze_dataset, FaceLivenessDataset
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def visualize_samples(data_dir, split='train', num_samples=8):
    """Visualize một số samples từ dataset"""
    dataset = FaceLivenessDataset(
        data_dir=data_dir,
        split=split,
        image_size=(112, 112),
        augment=False,
        normalize=False
    )
    
    # Lấy samples
    normal_samples = []
    spoof_samples = []
    
    for i, item in enumerate(dataset):
        if item['label'] == 1 and len(normal_samples) < num_samples // 2:
            normal_samples.append(item)
        elif item['label'] == 0 and len(spoof_samples) < num_samples // 2:
            spoof_samples.append(item)
        
        if len(normal_samples) + len(spoof_samples) >= num_samples:
            break
    
    # Plot
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))
    fig.suptitle(f'Sample Images from {split} set', fontsize=16)
    
    # Normal samples
    for i, sample in enumerate(normal_samples):
        ax = axes[0, i]
        image = sample['image'].transpose(1, 2, 0)  # CHW -> HWC
        image = (image * 255).astype(np.uint8)
        ax.imshow(image)
        ax.set_title('Normal/Real')
        ax.axis('off')
    
    # Spoof samples
    for i, sample in enumerate(spoof_samples):
        ax = axes[1, i]
        image = sample['image'].transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        ax.imshow(image)
        ax.set_title('Spoof/Fake')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'data_samples_{split}.png', dpi=150)
    print(f"Saved visualization to data_samples_{split}.png")
    plt.close()


def analyze_image_sizes(data_dir):
    """Phân tích kích thước ảnh trong dataset"""
    data_dir = Path(data_dir)
    sizes = []
    
    for split in ['train', 'test', 'dev']:
        normal_dir = data_dir / split / 'normal'
        spoof_dir = data_dir / split / 'spoof'
        
        for img_path in list(normal_dir.glob('*.jpg'))[:100] + list(spoof_dir.glob('*.jpg'))[:100]:
            img = cv2.imread(str(img_path))
            if img is not None:
                sizes.append(img.shape[:2])  # (height, width)
    
    sizes = np.array(sizes)
    print("\nImage Size Statistics:")
    print(f"  Height: min={sizes[:, 0].min()}, max={sizes[:, 0].max()}, mean={sizes[:, 0].mean():.1f}")
    print(f"  Width:  min={sizes[:, 1].min()}, max={sizes[:, 1].max()}, mean={sizes[:, 1].mean():.1f}")
    print(f"  Aspect Ratio: mean={sizes[:, 1].mean()/sizes[:, 0].mean():.2f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'dev'],
                       help='Split to analyze')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample images')
    parser.add_argument('--num-samples', type=int, default=8, help='Number of samples to visualize')
    parser.add_argument('--image-sizes', action='store_true', help='Analyze image sizes')
    
    args = parser.parse_args()
    
    # Basic analysis
    analyze_dataset(args.data_dir)
    
    # Image size analysis
    if args.image_sizes:
        analyze_image_sizes(args.data_dir)
    
    # Visualization
    if args.visualize:
        visualize_samples(args.data_dir, args.split, args.num_samples)


if __name__ == '__main__':
    main()

