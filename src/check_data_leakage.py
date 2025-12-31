"""
Kiểm tra Data Leakage: Xem train/val/test có chung identity không
"""
import os
from pathlib import Path
from collections import defaultdict
import re

def extract_identity_from_path(path):
    """
    Extract identity từ file path
    Giả sử format: data/train/normal/person_001_image_001.jpg
    hoặc: data/train/normal/001_001.jpg
    hoặc: data/train/normal/id001_001.jpg
    """
    filename = Path(path).stem
    # Thử nhiều pattern
    patterns = [
        r'person_(\d+)',  # person_001
        r'^(\d+)_',       # 001_
        r'_(\d+)_',       # _001_
        r'id(\d+)',       # id001
        r'(\d{3,})',      # 3+ chữ số liên tiếp
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Nếu không match, dùng toàn bộ filename (cảnh báo)
    return filename

def check_data_leakage(data_dir):
    """Kiểm tra identity overlap giữa train/val/test"""
    data_dir = Path(data_dir)
    
    # Collect identities từ mỗi split
    identities = {
        'train': defaultdict(list),
        'dev': defaultdict(list),
        'test': defaultdict(list)
    }
    
    for split in ['train', 'dev', 'test']:
        for label_type in ['normal', 'spoof']:
            split_dir = data_dir / split / label_type
            if not split_dir.exists():
                print(f"   {split}/{label_type} không tồn tại, bỏ qua")
                continue
            
            for img_path in split_dir.glob('*.jpg'):
                identity = extract_identity_from_path(str(img_path))
                identities[split][identity].append(str(img_path))
            
            for img_path in split_dir.glob('*.png'):
                identity = extract_identity_from_path(str(img_path))
                identities[split][identity].append(str(img_path))
    
    # Kiểm tra overlap
    print("="*60)
    print("Data Leakage Check")
    print("="*60)
    
    train_ids = set(identities['train'].keys())
    dev_ids = set(identities['dev'].keys())
    test_ids = set(identities['test'].keys())
    
    train_dev_overlap = train_ids & dev_ids
    train_test_overlap = train_ids & test_ids
    dev_test_overlap = dev_ids & test_ids
    
    print(f"\n Statistics:")
    print(f"  Train identities: {len(train_ids)}")
    print(f"  Dev identities: {len(dev_ids)}")
    print(f"  Test identities: {len(test_ids)}")
    
    # Count images
    train_images = sum(len(v) for v in identities['train'].values())
    dev_images = sum(len(v) for v in identities['dev'].values())
    test_images = sum(len(v) for v in identities['test'].values())
    print(f"\n  Train images: {train_images}")
    print(f"  Dev images: {dev_images}")
    print(f"  Test images: {test_images}")
    
    print(f"\n Overlap Analysis:")
    has_leakage = False
    
    if train_dev_overlap:
        print(f"  Train-Dev overlap: {len(train_dev_overlap)} identities")
        print(f"     Examples: {list(train_dev_overlap)[:5]}")
        has_leakage = True
    else:
        print(f"  Train-Dev: No overlap")
    
    if train_test_overlap:
        print(f"  Train-Test overlap: {len(train_test_overlap)} identities")
        print(f"     Examples: {list(train_test_overlap)[:5]}")
        has_leakage = True
    else:
        print(f"  Train-Test: No overlap")
    
    if dev_test_overlap:
        print(f"  Dev-Test overlap: {len(dev_test_overlap)} identities")
        print(f"     Examples: {list(dev_test_overlap)[:5]}")
        has_leakage = True
    else:
        print(f"  Dev-Test: No overlap")
    
    # Thống kê số ảnh per identity
    print(f"\n Images per identity (sample - first 10):")
    all_ids = train_ids | dev_ids | test_ids
    sample_ids = sorted(list(all_ids))[:10]
    for identity in sample_ids:
        train_count = len(identities['train'][identity])
        dev_count = len(identities['dev'][identity])
        test_count = len(identities['test'][identity])
        if train_count > 0 or dev_count > 0 or test_count > 0:
            print(f"  {identity}: Train={train_count}, Dev={dev_count}, Test={test_count}")
    
    # Warning nếu có leakage
    if has_leakage:
        print(f"\nData Leakage detected!")
        print(f"   Model có thể học nhận diện identity thay vì liveness.")
        print(f"   Khuyến nghị: Chia lại data theo identity (không random).")
    else:
        print(f"\n No data leakage detected. Data split looks good!")
    
    return {
        'train_dev_overlap': train_dev_overlap,
        'train_test_overlap': train_test_overlap,
        'dev_test_overlap': dev_test_overlap,
        'has_leakage': has_leakage
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check for data leakage between train/val/test splits')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    args = parser.parse_args()
    
    check_data_leakage(args.data_dir)


