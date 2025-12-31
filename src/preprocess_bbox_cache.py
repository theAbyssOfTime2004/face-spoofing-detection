"""
Script để pre-process và cache bbox detection results
Giúp tăng tốc training bằng cách detect một lần và lưu kết quả
"""
import cv2
import pickle
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse
from pipeline.detection import SCRFDDetector


def load_config(config_path):
    """Load config từ YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['pipeline']


def preprocess_bbox_cache(data_dir, config_path='config/config.yaml', output_cache='bbox_cache.pkl'):
    """
    Pre-process tất cả images và cache bbox detection results
    
    Args:
        data_dir: Đường dẫn đến thư mục data
        config_path: Đường dẫn đến config file
        output_cache: Đường dẫn để lưu cache file
    """
    print("="*60)
    print("Pre-processing BBox Cache for Training Speedup")
    print("="*60)
    
    # Load config
    config = load_config(config_path)
    detector_config = config['detection']
    
    # Initialize detector
    print("\nInitializing SCRFD detector...")
    detector = SCRFDDetector(detector_config)
    print("  ✓ Detector initialized")
    
    # Load all image paths
    data_dir = Path(data_dir)
    splits = ['train', 'test', 'dev']
    
    all_images = []
    for split in splits:
        normal_dir = data_dir / split / 'normal'
        spoof_dir = data_dir / split / 'spoof'
        
        normal_images = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png'))
        spoof_images = list(spoof_dir.glob('*.jpg')) + list(spoof_dir.glob('*.png'))
        
        for img_path in normal_images:
            all_images.append((str(img_path), split, 1))  # 1 = real
        for img_path in spoof_images:
            all_images.append((str(img_path), split, 0))  # 0 = spoof
    
    print(f"\nFound {len(all_images)} images to process")
    
    # Process và cache bbox
    bbox_cache = {}
    failed_images = []
    
    print("\nProcessing images...")
    for img_path, split, label in tqdm(all_images, desc="Detecting faces"):
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                failed_images.append(img_path)
                continue
            
            # Detect face (thử ảnh gốc trước)
            faces = detector.detect(image)
            
            # Nếu không detect được, thử rotate 180 độ (cho ảnh vertically flipped)
            if len(faces) == 0:
                image_rotated = cv2.rotate(image, cv2.ROTATE_180)
                faces = detector.detect(image_rotated)
                
                if len(faces) > 0:
                    # Detect được sau khi rotate → ảnh bị lật ngược
                    # Lưu bbox và landmarks, đánh dấu là đã rotate
                    face_data = faces[0]
                    bbox_cache[img_path] = {
                        'bbox': face_data['bbox'],
                        'landmarks': face_data.get('landmarks'),
                        'confidence': face_data.get('confidence', 1.0),
                        'split': split,
                        'label': label,
                        'rotated': True  # Đánh dấu đã rotate
                    }
                else:
                    # Vẫn không detect được sau khi rotate
                    failed_images.append(img_path)
                    print(f"  Warning: No face detected in {img_path} (even after rotation)")
            else:
                # Detect được ở ảnh gốc
                face_data = faces[0]
                bbox_cache[img_path] = {
                    'bbox': face_data['bbox'],
                    'landmarks': face_data.get('landmarks'),
                    'confidence': face_data.get('confidence', 1.0),
                    'split': split,
                    'label': label,
                    'rotated': False  # Không cần rotate
                }
        
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            failed_images.append(img_path)
            continue
    
    # Save cache
    cache_path = Path(output_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(bbox_cache, f)
    
    print("\n" + "="*60)
    print("Pre-processing Complete")
    print("="*60)
    print(f"Total images: {len(all_images)}")
    print(f"Successfully cached: {len(bbox_cache)}")
    print(f"Failed: {len(failed_images)}")
    print(f"\nCache saved to: {cache_path}")
    
    if failed_images:
        print(f"\nFailed images ({len(failed_images)}):")
        for img_path in failed_images[:10]:  # Show first 10
            print(f"  - {img_path}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    # Statistics by split
    print("\nStatistics by split:")
    for split in splits:
        split_count = sum(1 for v in bbox_cache.values() if v['split'] == split)
        print(f"  {split}: {split_count} images cached")
    
    return bbox_cache


def main():
    parser = argparse.ArgumentParser(description='Pre-process and cache bbox detection results')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='bbox_cache.pkl',
                       help='Output cache file path')
    
    args = parser.parse_args()
    
    preprocess_bbox_cache(args.data_dir, args.config, args.output)


if __name__ == '__main__':
    main()

