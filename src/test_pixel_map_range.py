"""
Script để kiểm tra range của pixel_map từ Local Branch
Xác định xem ONNX model đã có Sigmoid hay chưa
"""
import cv2
import numpy as np
import yaml
from pathlib import Path
from pipeline.detection import SCRFDDetector
from pipeline.liveness_ensemble import LivenessEnsemble

def test_pixel_map_range(config_path='config/config.yaml', num_samples=10):
    """Kiểm tra range của pixel_map với một vài ảnh"""
    print("="*60)
    print("Testing Pixel Map Range from Local Branch")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pipeline_config = config['pipeline']
    
    # Initialize detector
    detector = SCRFDDetector(pipeline_config['detection'])
    
    # Initialize ensemble
    ensemble = LivenessEnsemble(pipeline_config['liveness'])
    
    # Load test images
    data_dir = Path('data/test')
    real_images = list((data_dir / 'normal').glob('*.jpg'))[:num_samples]
    spoof_images = list((data_dir / 'spoof').glob('*.jpg'))[:num_samples]
    
    all_images = [(str(img), 1) for img in real_images] + [(str(img), 0) for img in spoof_images]
    
    print(f"\nTesting {len(all_images)} images...\n")
    
    pixel_map_ranges = []
    
    for img_path, label in all_images:
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Detect face
            faces = detector.detect(image)
            if len(faces) == 0:
                continue
            
            # Extract raw face
            face_data = faces[0]
            raw_face = detector.extract_raw_face(
                image,
                face_data['bbox'],
                output_size=(112, 112)
            )
            
            if raw_face is None:
                continue
            
            # Get pixel map từ Local Branch
            local_score, pixel_map = ensemble.local_branch.predict(raw_face)
            
            if pixel_map is not None:
                min_val = float(np.min(pixel_map))
                max_val = float(np.max(pixel_map))
                mean_val = float(np.mean(pixel_map))
                
                pixel_map_ranges.append({
                    'path': img_path,
                    'label': 'real' if label == 1 else 'spoof',
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'score': local_score
                })
                
                print(f"{'Real' if label == 1 else 'Spoof':5s} | "
                      f"Min: {min_val:7.3f} | "
                      f"Max: {max_val:7.3f} | "
                      f"Mean: {mean_val:7.3f} | "
                      f"Score: {local_score:.3f}")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Summary
    if pixel_map_ranges:
        all_mins = [r['min'] for r in pixel_map_ranges]
        all_maxs = [r['max'] for r in pixel_map_ranges]
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Pixel Map Range:")
        print(f"  Min (overall): {min(all_mins):.3f}")
        print(f"  Max (overall): {max(all_maxs):.3f}")
        print(f"  Mean (overall): {np.mean([r['mean'] for r in pixel_map_ranges]):.3f}")
        
        if min(all_mins) < 0:
            print(f"\n⚠️  NEGATIVE VALUES DETECTED!")
            print(f"   → Pixel map chưa có Sigmoid")
            print(f"   → Cần apply Sigmoid: 1 / (1 + exp(-x))")
        elif min(all_mins) >= 0 and max(all_maxs) <= 1.0:
            print(f"\n✓  Values trong [0, 1]")
            print(f"   → Pixel map đã có Sigmoid")
            print(f"   → Giữ nguyên, không cần apply thêm")
        else:
            print(f"\n⚠️  Values ngoài [0, 1]")
            print(f"   → Cần kiểm tra lại model output")

if __name__ == '__main__':
    test_pixel_map_range()

