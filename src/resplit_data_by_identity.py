"""
Chia lại data theo Identity để tránh Data Leakage
Mỗi identity chỉ xuất hiện ở 1 split (train/val/test)
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict
import re
import random

def extract_identity_from_path(path):
    """Extract identity từ file path"""
    filename = Path(path).stem
    patterns = [
        r'person_(\d+)',
        r'^(\d+)_',
        r'_(\d+)_',
        r'id(\d+)',
        r'(\d{3,})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return filename

def collect_identity_images(data_dir):
    """Collect tất cả images theo identity"""
    data_dir = Path(data_dir)
    identity_images = defaultdict(lambda: {'normal': [], 'spoof': []})
    
    for split in ['train', 'dev', 'test']:
        for label_type in ['normal', 'spoof']:
            split_dir = data_dir / split / label_type
            if not split_dir.exists():
                continue
            
            for img_path in split_dir.glob('*.jpg'):
                identity = extract_identity_from_path(str(img_path))
                identity_images[identity][label_type].append(img_path)
            
            for img_path in split_dir.glob('*.png'):
                identity = extract_identity_from_path(str(img_path))
                identity_images[identity][label_type].append(img_path)
    
    return identity_images

def resplit_by_identity(data_dir, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, backup=True, auto_confirm=False):
    """
    Chia lại data theo identity
    
    Args:
        data_dir: Thư mục data gốc
        train_ratio: Tỷ lệ identities cho train (default: 0.7)
        dev_ratio: Tỷ lệ identities cho dev (default: 0.15)
        test_ratio: Tỷ lệ identities cho test (default: 0.15)
        backup: Có backup data cũ không (default: True)
    """
    data_dir = Path(data_dir)
    
    # Validate ratios
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + dev_ratio + test_ratio}")
    
    print("="*60)
    print("Resplit Data by Identity")
    print("="*60)
    
    # Backup nếu cần
    backup_dir = data_dir.parent / f"{data_dir.name}_backup_before_resplit"
    if backup:
        if backup_dir.exists():
            print(f"⚠️  Backup directory exists: {backup_dir}")
            if not auto_confirm:
                response = input("  Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print("  Cancelled.")
                    return
            else:
                print("  Auto-overwriting backup (--yes flag)")
        
        print(f"\n📦 Creating backup to {backup_dir}...")
        shutil.copytree(data_dir, backup_dir, dirs_exist_ok=True)
        print("  ✅ Backup created")
    
    # Collect images theo identity từ backup (nếu có) hoặc data_dir
    print("\n📊 Collecting images by identity...")
    source_dir = backup_dir if backup and backup_dir.exists() else data_dir
    identity_images_raw = collect_identity_images(source_dir)
    
    # Convert to absolute paths
    identity_images = defaultdict(lambda: {'normal': [], 'spoof': []})
    for identity, images in identity_images_raw.items():
        identity_images[identity]['normal'] = [Path(img).resolve() for img in images['normal']]
        identity_images[identity]['spoof'] = [Path(img).resolve() for img in images['spoof']]
    
    # Kiểm tra test split có overlap không
    test_identities = set()
    for img_path in (data_dir / 'test' / 'normal').glob('*.jpg'):
        test_identities.add(extract_identity_from_path(str(img_path)))
    for img_path in (data_dir / 'test' / 'normal').glob('*.png'):
        test_identities.add(extract_identity_from_path(str(img_path)))
    
    train_dev_identities = set()
    for img_path in (data_dir / 'train' / 'normal').glob('*.jpg'):
        train_dev_identities.add(extract_identity_from_path(str(img_path)))
    for img_path in (data_dir / 'dev' / 'normal').glob('*.jpg'):
        train_dev_identities.add(extract_identity_from_path(str(img_path)))
    
    test_overlap = test_identities & train_dev_identities
    
    if len(test_overlap) == 0:
        print(f"  ✅ Test split không có overlap, sẽ giữ nguyên test split")
        preserve_test = True
    else:
        print(f"  ⚠️  Test split có {len(test_overlap)} identities overlap, sẽ chia lại toàn bộ")
        preserve_test = False
    
    # Collect tất cả identities (không yêu cầu cả normal và spoof)
    # Vì dataset có thể có identities chỉ có normal hoặc chỉ có spoof
    # Balance sẽ được đảm bảo ở mức split (train/dev/test), không phải ở mức identity
    all_identities = []
    for identity, images in identity_images.items():
        # Lấy tất cả identities có ít nhất 1 image (normal hoặc spoof)
        if len(images['normal']) > 0 or len(images['spoof']) > 0:
            if preserve_test and identity in test_identities:
                continue  # Bỏ qua test identities
            all_identities.append(identity)
    
    # Phân loại identities
    normal_only_ids = []
    spoof_only_ids = []
    both_ids = []
    
    for identity in all_identities:
        images = identity_images[identity]
        has_normal = len(images['normal']) > 0
        has_spoof = len(images['spoof']) > 0
        
        if has_normal and has_spoof:
            both_ids.append(identity)
        elif has_normal:
            normal_only_ids.append(identity)
        elif has_spoof:
            spoof_only_ids.append(identity)
    
    print(f"\n  Phân loại identities:")
    print(f"    Có cả normal và spoof: {len(both_ids)}")
    print(f"    Chỉ có normal: {len(normal_only_ids)}")
    print(f"    Chỉ có spoof: {len(spoof_only_ids)}")
    
    # Chia riêng normal và spoof identities để đảm bảo balance
    # Strategy: Chia normal và spoof identities riêng, sau đó gộp lại
    valid_identities = all_identities
    
    print(f"  Total identities: {len(identity_images)}")
    print(f"  Valid identities (sẽ chia lại): {len(valid_identities)}")
    if preserve_test:
        print(f"  Test identities (giữ nguyên): {len(test_identities)}")
    
    # Shuffle identities (riêng normal và spoof để đảm bảo balance)
    random.seed(42)  # Reproducible
    random.shuffle(normal_only_ids)
    random.shuffle(spoof_only_ids)
    random.shuffle(both_ids)
    
    # Chia identities (chia riêng từng loại để đảm bảo balance)
    if preserve_test:
        # Chỉ chia train và dev, giữ nguyên test
        # Tính lại ratio cho train và dev (không tính test)
        train_dev_ratio = train_ratio / (train_ratio + dev_ratio)
        
        # Chia riêng từng loại identities
        n_normal_train = int(len(normal_only_ids) * train_dev_ratio)
        n_spoof_train = int(len(spoof_only_ids) * train_dev_ratio)
        n_both_train = int(len(both_ids) * train_dev_ratio)
        
        train_normal_ids = set(normal_only_ids[:n_normal_train])
        train_spoof_ids = set(spoof_only_ids[:n_spoof_train])
        train_both_ids = set(both_ids[:n_both_train])
        train_ids = train_normal_ids | train_spoof_ids | train_both_ids
        
        dev_normal_ids = set(normal_only_ids[n_normal_train:])
        dev_spoof_ids = set(spoof_only_ids[n_spoof_train:])
        dev_both_ids = set(both_ids[n_both_train:])
        dev_ids = dev_normal_ids | dev_spoof_ids | dev_both_ids
        
        test_ids = test_identities  # Giữ nguyên test
        
        n_total = len(normal_only_ids) + len(spoof_only_ids) + len(both_ids)
        print(f"\n📈 Split plan (Test giữ nguyên):")
        print(f"  Train: {len(train_ids)} identities ({100*len(train_ids)/n_total:.1f}%)")
        print(f"  Dev: {len(dev_ids)} identities ({100*len(dev_ids)/n_total:.1f}%)")
        print(f"  Test: {len(test_ids)} identities (giữ nguyên)")
    else:
        # Chia lại toàn bộ (bao gồm test)
        n_normal_train = int(len(normal_only_ids) * train_ratio)
        n_normal_dev = int(len(normal_only_ids) * dev_ratio)
        
        n_spoof_train = int(len(spoof_only_ids) * train_ratio)
        n_spoof_dev = int(len(spoof_only_ids) * dev_ratio)
        
        n_both_train = int(len(both_ids) * train_ratio)
        n_both_dev = int(len(both_ids) * dev_ratio)
        
        train_normal_ids = set(normal_only_ids[:n_normal_train])
        train_spoof_ids = set(spoof_only_ids[:n_spoof_train])
        train_both_ids = set(both_ids[:n_both_train])
        train_ids = train_normal_ids | train_spoof_ids | train_both_ids
        
        dev_normal_ids = set(normal_only_ids[n_normal_train:n_normal_train+n_normal_dev])
        dev_spoof_ids = set(spoof_only_ids[n_spoof_train:n_spoof_train+n_spoof_dev])
        dev_both_ids = set(both_ids[n_both_train:n_both_train+n_both_dev])
        dev_ids = dev_normal_ids | dev_spoof_ids | dev_both_ids
        
        test_normal_ids = set(normal_only_ids[n_normal_train+n_normal_dev:])
        test_spoof_ids = set(spoof_only_ids[n_spoof_train+n_spoof_dev:])
        test_both_ids = set(both_ids[n_both_train+n_both_dev:])
        test_ids = test_normal_ids | test_spoof_ids | test_both_ids
        
        n_total = len(normal_only_ids) + len(spoof_only_ids) + len(both_ids)
        print(f"\n📈 Split plan:")
        print(f"  Train: {len(train_ids)} identities ({100*len(train_ids)/n_total:.1f}%)")
        print(f"  Dev: {len(dev_ids)} identities ({100*len(dev_ids)/n_total:.1f}%)")
        print(f"  Test: {len(test_ids)} identities ({100*len(test_ids)/n_total:.1f}%)")
    
    # Xác nhận
    if preserve_test:
        print(f"\n⚠️  This will DELETE and recreate train/dev directories!")
        print(f"    Test directory will be PRESERVED.")
    else:
        print(f"\n⚠️  This will DELETE and recreate train/dev/test directories!")
    
    if not auto_confirm:
        response = input("  Continue? (y/n): ")
        if response.lower() != 'y':
            print("  Cancelled.")
            return
    else:
        print("  Auto-confirmed (--yes flag)")
    
    # Xóa thư mục cũ
    print("\n🗑️  Removing old splits...")
    splits_to_process = ['train', 'dev'] if preserve_test else ['train', 'dev', 'test']
    for split in splits_to_process:
        for label_type in ['normal', 'spoof']:
            split_dir = data_dir / split / label_type
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images theo identity
    print("\n📋 Copying images to new splits...")
    
    splits = {
        'train': train_ids,
        'dev': dev_ids,
    }
    
    if not preserve_test:
        splits['test'] = test_ids
    
    for split_name, identity_set in splits.items():
        for identity in identity_set:
            images = identity_images[identity]
            
            # Copy normal images
            for img_path in images['normal']:
                dst = data_dir / split_name / 'normal' / img_path.name
                shutil.copy2(img_path, dst)
            
            # Copy spoof images
            for img_path in images['spoof']:
                dst = data_dir / split_name / 'spoof' / img_path.name
                shutil.copy2(img_path, dst)
        
        # Count images
        n_normal = len(list((data_dir / split_name / 'normal').glob('*.jpg'))) + \
                   len(list((data_dir / split_name / 'normal').glob('*.png')))
        n_spoof = len(list((data_dir / split_name / 'spoof').glob('*.jpg'))) + \
                  len(list((data_dir / split_name / 'spoof').glob('*.png')))
        print(f"  {split_name}: {n_normal} normal, {n_spoof} spoof")
    
    print("\n✅ Resplit completed!")
    print("\n🔍 Verifying no overlap...")
    
    # Verify
    final_train_ids = set()
    final_dev_ids = set()
    final_test_ids = set()
    
    for img_path in (data_dir / 'train' / 'normal').glob('*.jpg'):
        final_train_ids.add(extract_identity_from_path(str(img_path)))
    for img_path in (data_dir / 'train' / 'normal').glob('*.png'):
        final_train_ids.add(extract_identity_from_path(str(img_path)))
    
    for img_path in (data_dir / 'dev' / 'normal').glob('*.jpg'):
        final_dev_ids.add(extract_identity_from_path(str(img_path)))
    for img_path in (data_dir / 'dev' / 'normal').glob('*.png'):
        final_dev_ids.add(extract_identity_from_path(str(img_path)))
    
    for img_path in (data_dir / 'test' / 'normal').glob('*.jpg'):
        final_test_ids.add(extract_identity_from_path(str(img_path)))
    for img_path in (data_dir / 'test' / 'normal').glob('*.png'):
        final_test_ids.add(extract_identity_from_path(str(img_path)))
    
    overlap_train_dev = final_train_ids & final_dev_ids
    overlap_train_test = final_train_ids & final_test_ids
    overlap_dev_test = final_dev_ids & final_test_ids
    
    has_error = False
    if overlap_train_dev:
        print(f"  ❌ Train-Dev overlap: {len(overlap_train_dev)} identities")
        has_error = True
    else:
        print(f"  ✅ Train-Dev: No overlap")
    
    if preserve_test:
        if overlap_train_test:
            print(f"  ⚠️  Train-Test overlap: {len(overlap_train_test)} identities (test was preserved)")
            has_error = True
        else:
            print(f"  ✅ Train-Test: No overlap (test preserved)")
        
        if overlap_dev_test:
            print(f"  ⚠️  Dev-Test overlap: {len(overlap_dev_test)} identities (test was preserved)")
            has_error = True
        else:
            print(f"  ✅ Dev-Test: No overlap (test preserved)")
    else:
        if overlap_train_test:
            print(f"  ❌ Train-Test overlap: {len(overlap_train_test)} identities")
            has_error = True
        else:
            print(f"  ✅ Train-Test: No overlap")
        
        if overlap_dev_test:
            print(f"  ❌ Dev-Test overlap: {len(overlap_dev_test)} identities")
            has_error = True
        else:
            print(f"  ✅ Dev-Test: No overlap")
    
    if not has_error:
        print("\n  ✅ No overlap! Data split is correct.")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Resplit data by identity to avoid data leakage')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train ratio (default: 0.7)')
    parser.add_argument('--dev-ratio', type=float, default=0.15, help='Dev ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test ratio (default: 0.15)')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup (not recommended)')
    parser.add_argument('--yes', '-y', action='store_true', help='Auto-confirm (skip prompt)')
    args = parser.parse_args()
    
    resplit_by_identity(
        args.data_dir,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        backup=not args.no_backup,
        auto_confirm=args.yes
    )

