"""
Script để xóa các ảnh không detect được face
"""
import pickle
from pathlib import Path
import argparse


def remove_failed_images(data_dir, cache_path='bbox_cache.pkl', dry_run=False, auto_confirm=False):
    """
    Xóa các ảnh không có trong bbox cache (không detect được face)
    
    Args:
        data_dir: Thư mục data
        cache_path: Đường dẫn đến bbox_cache.pkl
        dry_run: Chỉ hiển thị, không xóa thực sự
    """
    print("="*60)
    print("Remove Failed Images (No Face Detected)")
    print("="*60)
    
    # Load cache
    if not Path(cache_path).exists():
        print(f"❌ Cache file not found: {cache_path}")
        return
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    print(f"\n📊 Cache info:")
    print(f"   Total cached images: {len(cache)}")
    
    # Collect tất cả ảnh trong data
    all_images = []
    for split in ['train', 'dev', 'test']:
        for label_type in ['normal', 'spoof']:
            split_dir = Path(data_dir) / split / label_type
            if split_dir.exists():
                for img_path in split_dir.glob('*.jpg'):
                    all_images.append(img_path)
                for img_path in split_dir.glob('*.png'):
                    all_images.append(img_path)
    
    # Tìm ảnh không có trong cache
    failed_images = [img for img in all_images if str(img) not in cache]
    
    print(f"\n📈 Analysis:")
    print(f"   Total images in data: {len(all_images)}")
    print(f"   Cached images: {len(cache)}")
    print(f"   Failed images (not in cache): {len(failed_images)}")
    
    if len(failed_images) == 0:
        print("\n✅ No failed images to remove!")
        return
    
    # Hiển thị danh sách
    print(f"\n🗑️  Failed images to remove ({len(failed_images)}):")
    for img in failed_images[:20]:
        print(f"   {img}")
    if len(failed_images) > 20:
        print(f"   ... and {len(failed_images) - 20} more")
    
    # Xác nhận
    if not dry_run:
        print(f"\n⚠️  This will DELETE {len(failed_images)} images!")
        if not auto_confirm:
            response = input("  Continue? (y/n): ")
            if response.lower() != 'y':
                print("  Cancelled.")
                return
        else:
            print("  Auto-confirmed (--yes flag)")
        
        # Xóa ảnh
        print(f"\n🗑️  Removing {len(failed_images)} images...")
        removed_count = 0
        for img_path in failed_images:
            try:
                img_path.unlink()
                removed_count += 1
            except Exception as e:
                print(f"  Error removing {img_path}: {e}")
        
        print(f"\n✅ Removed {removed_count} images")
        print(f"\n📊 Remaining images:")
        remaining = len(all_images) - removed_count
        print(f"   {remaining} images remaining")
    else:
        print(f"\n🔍 DRY RUN: Would remove {len(failed_images)} images")
        print("   Run without --dry-run to actually remove")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove images that failed face detection')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--cache-path', type=str, default='bbox_cache.pkl', help='Path to bbox_cache.pkl')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (show what would be removed)')
    parser.add_argument('--yes', '-y', action='store_true', help='Auto-confirm (skip prompt)')
    args = parser.parse_args()
    
    remove_failed_images(args.data_dir, args.cache_path, args.dry_run, args.yes)

