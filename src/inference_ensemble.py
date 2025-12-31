"""
Inference với Ensemble Learning
Sử dụng cả 3 nhánh: Global + Local + Temporal
"""
import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.detection import SCRFDDetector
from pipeline.liveness_ensemble import LivenessEnsemble
from pipeline.quality_gate import QualityGate


def visualize_result(image_path, result, save_path=None):
    """
    Visualize kết quả trên ảnh với thông tin ensemble
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return
    
    # Draw bounding box nếu có
    if 'detection' in result and 'bbox' in result['detection']:
        bbox = result['detection']['bbox']
        color = (0, 255, 0) if result['liveness']['is_real'] else (0, 0, 255)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    # Draw result
    prediction = "REAL" if result['liveness']['is_real'] else "SPOOF"
    confidence = result['liveness']['final_score']
    
    # Color: green for REAL, red for SPOOF
    color = (0, 255, 0) if prediction == "REAL" else (0, 0, 255)
    
    # Main prediction text
    text = f"{prediction} ({confidence*100:.1f}%)"
    cv2.putText(image, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Ensemble scores
    liveness = result['liveness']
    y_offset = 70
    cv2.putText(image, f"Global: {liveness.get('global_score', 0):.3f}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(image, f"Local: {liveness.get('local_score', 0):.3f}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if 'temporal_score' in liveness:
        y_offset += 25
        cv2.putText(image, f"Temporal: {liveness.get('temporal_score', 0):.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save or show
    if save_path:
        cv2.imwrite(str(save_path), image)
        print(f"Saved result to {save_path}")
    else:
        # Try to display, fallback to auto-save if GUI not available
        try:
            cv2.imshow('Ensemble Prediction Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            # GUI not available, auto-save to file
            auto_save_path = str(image_path.parent / f"{image_path.stem}_ensemble_result{image_path.suffix}")
            cv2.imwrite(auto_save_path, image)
            print(f"\nNote: GUI not available. Result saved to: {auto_save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Inference với Ensemble Learning - Sử dụng cả 3 nhánh (Global + Local + Temporal)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference với ensemble (sử dụng config.yaml)
  python src/inference_ensemble.py --image test.jpg --config config/config.yaml
  
  # Inference và save kết quả
  python src/inference_ensemble.py --image test.jpg --config config/config.yaml --output result.jpg
  
  # Inference với config tùy chỉnh
  python src/inference_ensemble.py --image test.jpg --config custom_config.yaml
        """
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Đường dẫn đến ảnh cần predict')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Đường dẫn đến config file')
    parser.add_argument('--output', type=str, default=None,
                       help='Đường dẫn để lưu ảnh kết quả (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Không hiển thị ảnh (chỉ print kết quả)')
    parser.add_argument('--skip-quality', action='store_true',
                       help='Bỏ qua Quality Gate (nếu ảnh đã được preprocess)')
    
    args = parser.parse_args()
    
    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Load config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        pipeline_config = config['pipeline']
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    print("="*60)
    print("Face Liveness Detection - Ensemble Inference")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Config: {args.config}")
    print("-"*60)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Cannot load image from {args.image}")
        return
    
    print(f"Image size: {image.shape}")
    
    # Initialize components
    print("\nInitializing components...")
    
    # 1. Quality Gate
    quality_gate = None
    if not args.skip_quality:
        quality_gate = QualityGate(pipeline_config['quality_gate'])
        print("  ✓ Quality Gate initialized")
    
    # 2. Face Detection
    detector = SCRFDDetector(pipeline_config['detection'])
    print("  ✓ Face Detector initialized")
    
    # 3. Liveness Ensemble
    ensemble = LivenessEnsemble(pipeline_config['liveness'])
    print("  ✓ Liveness Ensemble initialized")
    print(f"    - Global Branch: {pipeline_config['liveness']['global_branch']['model_path']}")
    print(f"    - Local Branch: {pipeline_config['liveness']['local_branch']['model_path']}")
    print(f"    - Temporal Branch: {'Enabled' if pipeline_config['liveness']['temporal_branch']['enabled'] else 'Disabled'}")
    print(f"    - Fusion Method: {pipeline_config['liveness']['fusion_method']}")
    
    # Check if models exist
    global_model_path = Path(pipeline_config['liveness']['global_branch']['model_path'])
    local_model_path = Path(pipeline_config['liveness']['local_branch']['model_path'])
    
    if not global_model_path.exists() or not local_model_path.exists():
        print("\n⚠ WARNING: ONNX models not found!")
        print("  The ensemble will use dummy predictions (not accurate).")
        print("  To use trained models:")
        print("  1. Convert PyTorch checkpoints to ONNX:")
        print("     python src/convert_to_onnx.py --model-type global --checkpoint checkpoints/best_global.pth --output models/minifasnet_v2.onnx")
        print("     python src/convert_to_onnx.py --model-type local --checkpoint checkpoints/best_local.pth --output models/deeppixbis.onnx")
        print("  2. Or use single model inference:")
        print("     python src/inference.py --image <image> --model-type global --checkpoint checkpoints/best_global.pth")
    
    print("\n" + "-"*60)
    print("Processing...")
    print("-"*60)
    
    result = {
        'image_path': str(image_path),
        'status': 'rejected',
        'message': ''
    }
    
    try:
        # Step 1: Quality Gate
        if quality_gate is not None:
            is_valid, quality_info = quality_gate.validate(image)
            result['quality_info'] = quality_info
            
            if not is_valid:
                failed_reason = quality_info.get('failed_reason', 'unknown')
                result['message'] = f"Frame không đạt chất lượng: {failed_reason}"
                print(f"⚠ Quality Gate failed: {result['message']}")
                if failed_reason == 'blur':
                    blur_score = quality_info.get('blur_score', 0)
                    print(f"  Blur score: {blur_score:.2f} (threshold: {pipeline_config['quality_gate']['blur_threshold']})")
                    print(f"  Tip: Use --skip-quality to bypass quality check")
                # Continue anyway if user wants (but warn)
                if not args.skip_quality:
                    print("  Continuing anyway... (use --skip-quality to suppress this warning)")
                else:
                    print("  Continuing with --skip-quality flag...")
            
            if is_valid:
                print("✓ Quality Gate passed")
        
        # Step 2: Face Detection
        detections = detector.detect(image)
        if len(detections) == 0:
            result['message'] = 'Không phát hiện khuôn mặt'
            print("✗ No face detected")
            return result
        
        print(f"✓ Detected {len(detections)} face(s)")
        
        # Lấy face đầu tiên
        face_data = detections[0]
        result['detection'] = {
            'bbox': face_data['bbox'],
            'confidence': face_data['confidence'],
            'num_faces': len(detections)
        }
        
        # Extract raw face crop (KHÔNG alignment) cho Liveness Model
        # Quan trọng: Giữ nguyên pixel gốc để bảo toàn high-frequency patterns (Moiré)
        raw_face = detector.extract_raw_face(
            image,
            face_data['bbox'],
            output_size=(224, 224)  # Match với Local Branch input size (tránh resize 2 lần)
        )
        
        print(f"✓ Face extracted (raw crop, no alignment): {raw_face.shape}")
        print("  Note: Raw crop preserves high-frequency patterns for Moiré detection")
        
        # Step 3: Liveness Detection với Ensemble (dùng raw crop)
        print("\nRunning Ensemble Prediction...")
        liveness_result = ensemble.predict(raw_face, frame_count=0)
        result['liveness'] = liveness_result
        
        # Step 4: Final decision
        if liveness_result['is_real']:
            result['status'] = 'accepted'
            result['message'] = f"Liveness check passed - Face is REAL"
        else:
            result['status'] = 'rejected'
            result['message'] = f"Liveness check failed - Face is SPOOF"
        
        # Print results
        print("\n" + "="*60)
        print("ENSEMBLE PREDICTION RESULT")
        print("="*60)
        print(f"Image: {args.image}")
        print(f"Status: {result['status'].upper()}")
        print(f"Message: {result['message']}")
        print(f"\nEnsemble Scores:")
        print(f"  Global Branch:   {liveness_result['global_score']:.4f} {'✓' if liveness_result['global_passed'] else '✗'}")
        print(f"  Local Branch:    {liveness_result['local_score']:.4f} {'✓' if liveness_result['local_passed'] else '✗'}")
        
        if 'temporal_score' in liveness_result:
            print(f"  Temporal Branch: {liveness_result['temporal_score']:.4f}")
            if 'blink_count' in liveness_result:
                print(f"    Blink count: {liveness_result['blink_count']}")
        
        print(f"\nFinal Score (Ensemble): {liveness_result['final_score']:.4f}")
        print(f"Prediction: {'REAL' if liveness_result['is_real'] else 'SPOOF'}")
        print("="*60)
        
        # Visualize
        if not args.no_display or args.output:
            visualize_result(image_path, result, args.output)
        
        return result
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        result['message'] = f"Error: {str(e)}"
        return result


if __name__ == '__main__':
    main()

