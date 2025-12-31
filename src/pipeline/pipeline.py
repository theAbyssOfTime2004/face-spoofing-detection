"""
Main Face Liveness Detection Pipeline
Kết hợp tất cả các module: Quality Gate -> Detection -> Liveness -> Recognition (optional)
"""
import cv2
import numpy as np
from typing import Dict, Optional, List
from .quality_gate import QualityGate
from .detection import SCRFDDetector
from .liveness_ensemble import LivenessEnsemble
from .recognition import FaceRecognizer


class FaceLivenessPipeline:
    """SOTA Face Liveness Detection Pipeline 2025 - Chỉ tập trung Liveness"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize components
        self.quality_gate = QualityGate(config['quality_gate'])
        self.detector = SCRFDDetector(config['detection'])
        self.liveness = LivenessEnsemble(config['liveness'])
        
        # Recognition chỉ dùng nếu enabled (cho 1-1 matching)
        self.recognizer = None
        if config.get('recognition', {}).get('enabled', False):
            self.recognizer = FaceRecognizer(config['recognition'])
        
        # Frame buffer cho temporal analysis
        self.frame_buffer = []
        self.max_buffer_size = 10
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'quality_passed': 0,
            'detection_passed': 0,
            'liveness_passed': 0,
            'final_accepted': 0
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process một frame từ video stream
        Args:
            frame: Input frame (BGR format)
        Returns:
            {
                'status': 'rejected' | 'processing' | 'accepted',
                'quality_info': dict,
                'detection': dict,
                'liveness': dict,
                'recognition': dict (optional),
                'message': str
            }
        """
        self.stats['total_frames'] += 1
        
        result = {
            'status': 'rejected',
            'message': '',
            'frame_number': self.stats['total_frames']
        }
        
        # 1. Quality Gate
        is_valid, quality_info = self.quality_gate.validate(frame)
        result['quality_info'] = quality_info
        
        if not is_valid:
            result['message'] = f"Frame không đạt chất lượng: {quality_info.get('failed_reason', 'unknown')}"
            return result
        
        self.stats['quality_passed'] += 1
        
        # 2. Detection & Alignment
        detections = self.detector.detect(frame)
        if len(detections) == 0:
            result['message'] = 'Không phát hiện khuôn mặt'
            return result
        
        # Lấy face đầu tiên (có thể cải thiện để chọn face tốt nhất)
        face_data = detections[0]
        result['detection'] = {
            'bbox': face_data['bbox'],
            'confidence': face_data['confidence'],
            'num_faces': len(detections)
        }
        
        self.stats['detection_passed'] += 1
        
        # Extract và align face
        x1, y1, x2, y2 = face_data['bbox']
        face_roi = frame[y1:y2, x1:x2]
        landmarks = np.array(face_data['landmarks']) if face_data['landmarks'] else None
        aligned_face = self.detector.align_face(face_roi, landmarks) if landmarks is not None else cv2.resize(face_roi, (112, 112))
        
        # 3. Liveness Detection (PHẦN CHÍNH)
        frame_count = len(self.frame_buffer)
        liveness_result = self.liveness.predict(aligned_face, frame_count)
        result['liveness'] = liveness_result
        
        # 4. Final decision dựa trên Liveness
        if liveness_result['is_real']:
            self.stats['liveness_passed'] += 1
            result['status'] = 'accepted'
            result['message'] = f"Liveness check passed - Face is REAL (Score: {liveness_result['final_score']:.3f})"
            self.stats['final_accepted'] += 1
        else:
            result['status'] = 'rejected'
            result['message'] = f"Liveness check failed - Score: {liveness_result['final_score']:.3f}"
        
        # 5. Optional: 1-1 Face Recognition (nếu cần so sánh với 1 face cụ thể)
        if self.recognizer is not None:
            embedding = self.recognizer.extract_embedding(aligned_face)
            if embedding is not None:
                result['recognition'] = {
                    'embedding_extracted': True,
                    'embedding_size': len(embedding)
                }
                # Có thể so sánh với 1 embedding cụ thể ở đây
                # if reference_embedding is not None:
                #     similarity = self.recognizer.compare_faces(embedding, reference_embedding)
                #     result['recognition']['similarity'] = similarity
                #     result['recognition']['is_match'] = similarity > self.config['recognition']['threshold']
            else:
                result['recognition'] = {
                    'embedding_extracted': False
                }
        
        # Add to buffer cho temporal analysis
        self.frame_buffer.append({
            'frame': aligned_face,
            'liveness': liveness_result,
            'frame_number': self.stats['total_frames']
        })
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        return result
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> List[Dict]:
        """
        Process toàn bộ video
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None = all)
        Returns:
            List of results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames is not None and frame_count >= max_frames:
                break
            
            result = self.process_frame(frame)
            results.append(result)
            frame_count += 1
            
            # Log kết quả
            if frame_count % 10 == 0 or result['status'] == 'accepted':
                print(f"Frame {frame_count}: {result['status']} - {result['message']}")
        
        cap.release()
        return results
    
    def get_final_decision(self) -> Dict:
        """
        Lấy quyết định cuối cùng dựa trên toàn bộ frame buffer
        (Dùng khi xử lý xong video stream)
        Returns:
            Final decision với statistics
        """
        if len(self.frame_buffer) == 0:
            return {
                'status': 'no_data',
                'message': 'Chưa có frame nào được xử lý',
                'stats': self.stats
            }
        
        # Tính toán dựa trên tất cả frames
        all_scores = [f['liveness']['final_score'] for f in self.frame_buffer]
        avg_score = np.mean(all_scores)
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        
        # Kiểm tra temporal branch
        temporal_passed = True
        if self.liveness.temporal_branch:
            temporal_passed = self.liveness.temporal_branch.validate()
        
        # Quyết định cuối cùng
        # Cần có ít nhất 70% frames pass liveness check
        passed_frames = sum(1 for f in self.frame_buffer if f['liveness']['is_real'])
        pass_rate = passed_frames / len(self.frame_buffer)
        
        if avg_score > 0.7 and pass_rate > 0.7 and temporal_passed:
            return {
                'status': 'accepted',
                'message': 'Face is REAL',
                'confidence': float(avg_score),
                'min_score': float(min_score),
                'max_score': float(max_score),
                'pass_rate': float(pass_rate),
                'frames_processed': len(self.frame_buffer),
                'temporal_passed': temporal_passed,
                'stats': self.stats.copy()
            }
        else:
            return {
                'status': 'rejected',
                'message': 'Face is FAKE or uncertain',
                'confidence': float(avg_score),
                'min_score': float(min_score),
                'max_score': float(max_score),
                'pass_rate': float(pass_rate),
                'frames_processed': len(self.frame_buffer),
                'temporal_passed': temporal_passed,
                'stats': self.stats.copy()
            }
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê xử lý"""
        stats = self.stats.copy()
        if stats['total_frames'] > 0:
            stats['quality_pass_rate'] = stats['quality_passed'] / stats['total_frames']
            stats['detection_pass_rate'] = stats['detection_passed'] / stats['total_frames']
            stats['liveness_pass_rate'] = stats['liveness_passed'] / stats['total_frames']
            stats['acceptance_rate'] = stats['final_accepted'] / stats['total_frames']
        return stats
    
    def reset(self):
        """Reset pipeline state"""
        self.frame_buffer.clear()
        if self.liveness.temporal_branch:
            self.liveness.temporal_branch.reset()
        self.stats = {
            'total_frames': 0,
            'quality_passed': 0,
            'detection_passed': 0,
            'liveness_passed': 0,
            'final_accepted': 0
        }


