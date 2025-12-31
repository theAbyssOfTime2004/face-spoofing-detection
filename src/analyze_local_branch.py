"""
Phân tích Local Branch output để hiểu tại sao mean score thấp
"""
import cv2
import numpy as np
import yaml
from pathlib import Path
from pipeline.detection import SCRFDDetector
from pipeline.liveness_ensemble import LivenessEnsemble

def analyze_local_branch(config_path='config/config.yaml', num_samples=20):
    """Phân tích chi tiết Local Branch output"""
    print("="*60)
    print("Phân tích Local Branch Output")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pipeline_config = config['pipeline']
    
    # Initialize
    detector = SCRFDDetector(pipeline_config['detection'])
    ensemble = LivenessEnsemble(pipeline_config['liveness'])
    
    # Load test images
    data_dir = Path('data/test')
    real_images = list((data_dir / 'normal').glob('*.jpg'))[:num_samples]
    spoof_images = list((data_dir / 'spoof').glob('*.jpg'))[:num_samples]
    
    all_images = [(str(img), 1) for img in real_images] + [(str(img), 0) for img in spoof_images]
    
    print(f"\nPhân tích {len(all_images)} images...\n")
    
    real_pixel_stats = []
    spoof_pixel_stats = []
    
    for img_path, label in all_images:
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            faces = detector.detect(image)
            if len(faces) == 0:
                continue
            
            face_data = faces[0]
            raw_face = detector.extract_raw_face(
                image,
                face_data['bbox'],
                output_size=(112, 112)
            )
            
            if raw_face is None:
                continue
            
            # Get pixel map
            local_score, pixel_map = ensemble.local_branch.predict(raw_face)
            
            if pixel_map is not None:
                pixel_map_flat = pixel_map.flatten()
                
                stats = {
                    'mean': float(np.mean(pixel_map)),
                    'min': float(np.min(pixel_map)),
                    'max': float(np.max(pixel_map)),
                    'std': float(np.std(pixel_map)),
                    'median': float(np.median(pixel_map)),
                    'pixels_above_0.5': int(np.sum(pixel_map > 0.5)),
                    'pixels_above_0.7': int(np.sum(pixel_map > 0.7)),
                    'pixels_above_0.9': int(np.sum(pixel_map > 0.9)),
                    'total_pixels': pixel_map.size,
                    'score': local_score
                }
                
                if label == 1:
                    real_pixel_stats.append(stats)
                else:
                    spoof_pixel_stats.append(stats)
        
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Summary
    print("="*60)
    print("REAL IMAGES - Pixel Map Statistics")
    print("="*60)
    if real_pixel_stats:
        print(f"Mean score: {np.mean([s['score'] for s in real_pixel_stats]):.4f}")
        print(f"Pixel map mean: {np.mean([s['mean'] for s in real_pixel_stats]):.4f}")
        print(f"Pixel map min: {np.mean([s['min'] for s in real_pixel_stats]):.4f}")
        print(f"Pixel map max: {np.mean([s['max'] for s in real_pixel_stats]):.4f}")
        print(f"Pixel map median: {np.mean([s['median'] for s in real_pixel_stats]):.4f}")
        print(f"Pixels > 0.5: {np.mean([s['pixels_above_0.5'] for s in real_pixel_stats]):.1f} / {real_pixel_stats[0]['total_pixels']}")
        print(f"Pixels > 0.7: {np.mean([s['pixels_above_0.7'] for s in real_pixel_stats]):.1f} / {real_pixel_stats[0]['total_pixels']}")
        print(f"Pixels > 0.9: {np.mean([s['pixels_above_0.9'] for s in real_pixel_stats]):.1f} / {real_pixel_stats[0]['total_pixels']}")
    
    print("\n" + "="*60)
    print("SPOOF IMAGES - Pixel Map Statistics")
    print("="*60)
    if spoof_pixel_stats:
        print(f"Mean score: {np.mean([s['score'] for s in spoof_pixel_stats]):.4f}")
        print(f"Pixel map mean: {np.mean([s['mean'] for s in spoof_pixel_stats]):.4f}")
        print(f"Pixel map min: {np.mean([s['min'] for s in spoof_pixel_stats]):.4f}")
        print(f"Pixel map max: {np.mean([s['max'] for s in spoof_pixel_stats]):.4f}")
        print(f"Pixel map median: {np.mean([s['median'] for s in spoof_pixel_stats]):.4f}")
        print(f"Pixels > 0.5: {np.mean([s['pixels_above_0.5'] for s in spoof_pixel_stats]):.1f} / {spoof_pixel_stats[0]['total_pixels']}")
        print(f"Pixels > 0.7: {np.mean([s['pixels_above_0.7'] for s in spoof_pixel_stats]):.1f} / {spoof_pixel_stats[0]['total_pixels']}")
        print(f"Pixels > 0.9: {np.mean([s['pixels_above_0.9'] for s in spoof_pixel_stats]):.1f} / {spoof_pixel_stats[0]['total_pixels']}")
    
    print("\n" + "="*60)
    print("PHÂN TÍCH")
    print("="*60)
    if real_pixel_stats and spoof_pixel_stats:
        real_mean = np.mean([s['mean'] for s in real_pixel_stats])
        spoof_mean = np.mean([s['mean'] for s in spoof_pixel_stats])
        
        print(f"Real mean: {real_mean:.4f}")
        print(f"Spoof mean: {spoof_mean:.4f}")
        print(f"Separation: {real_mean - spoof_mean:.4f}")
        
        print("\n⚠️  VẤN ĐỀ:")
        print("  Pixel map mean thấp (~0.23) vì:")
        print("  1. Pixel map không phải classification probability")
        print("  2. Một số vùng (edge, background) có giá trị thấp")
        print("  3. Mean không phản ánh đúng confidence của model")
        print("\n💡 GIẢI PHÁP:")
        print("  - Dùng binary_logits thay vì pixel_map mean")
        print("  - Hoặc dùng max/median thay vì mean")
        print("  - Hoặc dùng weighted mean (tập trung vào center)")

if __name__ == '__main__':
    analyze_local_branch()

