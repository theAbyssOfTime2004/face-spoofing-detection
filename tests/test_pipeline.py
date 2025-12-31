"""
Simple test script cho Face Liveness Detection Pipeline
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml
from src.pipeline.pipeline import FaceLivenessPipeline


def test_quality_gate():
    """Test Quality Gate module"""
    print("Testing Quality Gate...")
    from src.pipeline.quality_gate import QualityGate
    
    config = {
        'max_yaw': 20.0,
        'max_pitch': 20.0,
        'max_roll': 20.0,
        'blur_threshold': 100.0
    }
    
    quality_gate = QualityGate(config)
    
    # Tạo test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    is_valid, info = quality_gate.validate(test_image)
    print(f"  Quality Gate result: {is_valid}")
    print(f"  Blur score: {info.get('blur_score', 0):.2f}")
    print("  ✓ Quality Gate test passed\n")


def test_detection():
    """Test Detection module"""
    print("Testing Detection...")
    from src.pipeline.detection import SCRFDDetector
    
    config = {
        'model_name': 'buffalo_l',
        'input_size': [640, 640],
        'conf_threshold': 0.5,
        'nms_threshold': 0.4
    }
    
    detector = SCRFDDetector(config)
    
    # Tạo test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    detections = detector.detect(test_image)
    print(f"  Detected {len(detections)} faces")
    print("  ✓ Detection test passed\n")


def test_liveness_ensemble():
    """Test Liveness Ensemble module"""
    print("Testing Liveness Ensemble...")
    from src.pipeline.liveness_ensemble import LivenessEnsemble
    
    config = {
        'global_branch': {
            'model_path': 'models/minifasnet_v2.onnx',
            'threshold': 0.9,
            'weight': 0.4
        },
        'local_branch': {
            'model_path': 'models/deeppixbis.onnx',
            'threshold': 0.8,
            'weight': 0.4
        },
        'temporal_branch': {
            'enabled': False,  # Tắt để test nhanh
            'blink_frames': 5,
            'min_blinks': 1,
            'ear_threshold': 0.25,
            'weight': 0.2
        },
        'fusion_method': 'weighted_sum'
    }
    
    ensemble = LivenessEnsemble(config)
    
    # Tạo test face image
    test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    result = ensemble.predict(test_face)
    print(f"  Global score: {result.get('global_score', 0):.3f}")
    print(f"  Local score: {result.get('local_score', 0):.3f}")
    print(f"  Final score: {result.get('final_score', 0):.3f}")
    print(f"  Is real: {result.get('is_real', False)}")
    print("  ✓ Liveness Ensemble test passed\n")


def test_pipeline():
    """Test toàn bộ pipeline"""
    print("Testing Full Pipeline...")
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    if not config_path.exists():
        print(f"  ✗ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = FaceLivenessPipeline(config['pipeline'])
    
    # Tạo test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = pipeline.process_frame(test_image)
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")
    print("  ✓ Pipeline test passed\n")


def main():
    """Run all tests"""
    print("="*50)
    print("Face Liveness Detection Pipeline - Tests")
    print("="*50)
    print()
    
    try:
        test_quality_gate()
        test_detection()
        test_liveness_ensemble()
        test_pipeline()
        
        print("="*50)
        print("All tests completed!")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


