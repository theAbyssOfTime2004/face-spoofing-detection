"""
Main entry point cho Face Liveness Detection Pipeline
"""
import yaml
import cv2
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.pipeline import FaceLivenessPipeline


def load_config(config_path: str) -> dict:
    """Load configuration từ YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['pipeline']


def visualize_result(frame, result, show_details: bool = False):
    """Visualize kết quả trên frame"""
    status = result['status']
    status_color = (0, 255, 0) if status == 'accepted' else (0, 0, 255)
    
    # Vẽ status text
    cv2.putText(frame, status.upper(), (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Vẽ score nếu có
    if 'liveness' in result:
        score = result['liveness']['final_score']
        cv2.putText(frame, f"Score: {score:.3f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Vẽ bounding box nếu có detection
    if 'detection' in result and 'bbox' in result['detection']:
        bbox = result['detection']['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), status_color, 2)
    
    # Vẽ chi tiết nếu cần
    if show_details and 'liveness' in result:
        liveness = result['liveness']
        y_offset = 100
        cv2.putText(frame, f"Global: {liveness.get('global_score', 0):.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"Local: {liveness.get('local_score', 0):.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if 'temporal_score' in liveness:
            y_offset += 20
            cv2.putText(frame, f"Temporal: {liveness.get('temporal_score', 0):.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(
        description='Face Liveness Detection Pipeline - SOTA 2025 Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process từ camera
  python src/main.py --camera
  
  # Process từ video file
  python src/main.py --input video.mp4
  
  # Process với config tùy chỉnh
  python src/main.py --input video.mp4 --config config/custom_config.yaml
        """
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--input', type=str, help='Video file path')
    parser.add_argument('--camera', action='store_true', help='Use camera (index 0)')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--output', type=str, help='Output video path (optional)')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to process')
    parser.add_argument('--show-details', action='store_true', help='Show detailed scores on frame')
    parser.add_argument('--no-display', action='store_true', help='Do not display video window')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.camera and args.input is None:
        parser.error("Either --camera or --input must be specified")
    
    # Check config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = FaceLivenessPipeline(config)
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
    # Setup video writer nếu có output
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Sẽ set size sau khi đọc frame đầu tiên
    
    if args.camera or args.input is None:
        # Process từ camera
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {args.camera_id}")
            sys.exit(1)
        
        print(f"Bắt đầu xử lý từ camera {args.camera_id}. Nhấn 'q' để thoát.")
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Không thể đọc frame từ camera")
                    break
                
                frame_count += 1
                result = pipeline.process_frame(frame)
                
                # Visualize
                if not args.no_display:
                    visualize_result(frame, result, args.show_details)
                    cv2.imshow('Face Liveness Detection', frame)
                
                # Log mỗi 10 frames hoặc khi có kết quả quan trọng
                if frame_count % 10 == 0 or result['status'] == 'accepted':
                    print(f"Frame {frame_count}: {result['status']} - {result['message']}")
                
                # Write to output video nếu có
                if video_writer is not None:
                    video_writer.write(frame)
                
                if not args.no_display and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if args.max_frames and frame_count >= args.max_frames:
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Lấy quyết định cuối cùng
            final_decision = pipeline.get_final_decision()
            print(f"\n{'='*50}")
            print("=== KẾT QUẢ CUỐI CÙNG ===")
            print(f"{'='*50}")
            print(f"Status: {final_decision['status']}")
            print(f"Message: {final_decision['message']}")
            if 'confidence' in final_decision:
                print(f"Confidence: {final_decision['confidence']:.3f}")
            if 'pass_rate' in final_decision:
                print(f"Pass Rate: {final_decision['pass_rate']:.2%}")
            
            # Statistics
            stats = pipeline.get_statistics()
            print(f"\n=== THỐNG KÊ ===")
            print(f"Total frames: {stats['total_frames']}")
            print(f"Quality passed: {stats['quality_passed']} ({stats.get('quality_pass_rate', 0):.2%})")
            print(f"Detection passed: {stats['detection_passed']} ({stats.get('detection_pass_rate', 0):.2%})")
            print(f"Liveness passed: {stats['liveness_passed']} ({stats.get('liveness_pass_rate', 0):.2%})")
            print(f"Final accepted: {stats['final_accepted']} ({stats.get('acceptance_rate', 0):.2%})")
            
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if not args.no_display:
                cv2.destroyAllWindows()
    
    else:
        # Process từ video file
        if not os.path.exists(args.input):
            print(f"Error: Video file not found: {args.input}")
            sys.exit(1)
        
        print(f"Đang xử lý video: {args.input}")
        
        try:
            results = pipeline.process_video(args.input, args.max_frames)
            
            # Tổng kết
            accepted_count = sum(1 for r in results if r['status'] == 'accepted')
            rejected_count = sum(1 for r in results if r['status'] == 'rejected')
            
            print(f"\n{'='*50}")
            print("=== TỔNG KẾT ===")
            print(f"{'='*50}")
            print(f"Tổng số frames: {len(results)}")
            print(f"Accepted: {accepted_count} ({accepted_count/len(results)*100:.1f}%)")
            print(f"Rejected: {rejected_count} ({rejected_count/len(results)*100:.1f}%)")
            
            # Quyết định cuối cùng
            final_decision = pipeline.get_final_decision()
            print(f"\n=== KẾT QUẢ CUỐI CÙNG ===")
            print(f"Status: {final_decision['status']}")
            print(f"Message: {final_decision['message']}")
            if 'confidence' in final_decision:
                print(f"Confidence: {final_decision['confidence']:.3f}")
            if 'pass_rate' in final_decision:
                print(f"Pass Rate: {final_decision['pass_rate']:.2%}")
            
            # Statistics
            stats = pipeline.get_statistics()
            print(f"\n=== THỐNG KÊ ===")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2%}" if 'rate' in key else f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
        
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()


