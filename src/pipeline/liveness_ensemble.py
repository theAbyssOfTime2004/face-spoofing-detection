"""
Liveness Detection Ensemble Module
Multi-stage Ensemble với 3 nhánh: Global, Local, Temporal
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from collections import deque
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Please install: pip install onnxruntime")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Temporal branch will be limited.")


class GlobalBranch:
    """MiniFASNetV2 - Global Analysis"""
    
    def __init__(self, model_path: str, threshold: float):
        self.threshold = threshold
        self.session = None
        self.input_name = None
        
        if not ONNX_AVAILABLE:
            print(f"Warning: ONNX Runtime not available. Global branch will use dummy predictions.")
            return
        
        if os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(model_path)
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:
                print(f"Warning: Could not load Global branch model from {model_path}: {e}")
                print("Will use dummy predictions.")
        else:
            print(f"Warning: Global branch model not found at {model_path}")
            print("Will use dummy predictions.")
    
    def predict(self, face_image: np.ndarray) -> float:
        """
        Predict liveness score (0=fake, 1=real)
        Args:
            face_image: Raw cropped face image (BGR format, KHÔNG alignment)
                       Quan trọng: Raw crop giữ nguyên high-frequency patterns (Moiré)
        Returns:
            Liveness score (0-1)
        """
        if self.session is None:
            # Dummy prediction (giả định real)
            return 0.95
        
        try:
            # Preprocess cho MiniFASNetV2
            # Thường input size là 80x80 hoặc 112x112
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (80, 80))
            
            # Normalize: (pixel / 255.0 - 0.5) / 0.5
            face_normalized = (face_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            
            # Convert to NCHW format: [batch, channels, height, width]
            face_input = np.expand_dims(face_normalized.transpose(2, 0, 1), axis=0)
            
            # Inference
            outputs = self.session.run(None, {self.input_name: face_input})
            
            # Output format: [batch, 2] với [fake_logit, real_logit]
            # Model output là LOGITS, cần apply softmax để convert thành probabilities
            if len(outputs[0].shape) == 2:
                logits = outputs[0][0]  # [fake_logit, real_logit]
                
                # Apply softmax để convert logits → probabilities
                # Softmax: exp(x_i) / sum(exp(x_j))
                # Numerical stability: subtract max để tránh overflow
                max_logit = np.max(logits)
                logits_shifted = logits - max_logit
                exp_logits = np.exp(logits_shifted)
                probs = exp_logits / np.sum(exp_logits)
                
                # Lấy probability của class "real" (index 1)
                score = float(probs[1])  # real_score (probability, range [0, 1])
            else:
                # Fallback cho format khác
                score = float(outputs[0].flatten()[0])
                # Nếu output là probability của fake, đảo ngược
                if score < 0.5:
                    score = 1.0 - score
                # Apply softmax nếu cần (nếu là logit)
                if score < 0 or score > 1:
                    score = 1.0 / (1.0 + np.exp(-score))  # Sigmoid
            
            return score  # Đã là probability [0, 1], không cần clamp
            
        except Exception as e:
            print(f"Error in Global branch prediction: {e}")
            return 0.5  # Neutral score on error


class LocalBranch:
    """DeepPixBiS - Local Patch Analysis"""
    
    def __init__(self, model_path: str, threshold: float):
        self.threshold = threshold
        self.session = None
        self.input_name = None
        
        if not ONNX_AVAILABLE:
            print(f"Warning: ONNX Runtime not available. Local branch will use dummy predictions.")
            return
        
        if os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(model_path)
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:
                print(f"Warning: Could not load Local branch model from {model_path}: {e}")
                print("Will use dummy predictions.")
        else:
            print(f"Warning: Local branch model not found at {model_path}")
            print("Will use dummy predictions.")
    
    def predict(self, face_image: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Predict liveness score và pixel-wise map
        Args:
            face_image: Raw cropped face image (BGR format, KHÔNG alignment)
                       Quan trọng: Raw crop giữ nguyên texture patterns
        Returns:
            (score, pixel_map)
        """
        if self.session is None:
            # Dummy prediction
            return 0.9, None
        
        try:
            # Preprocess cho DeepPixBiS
            # Thường input size là 224x224
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            
            # Normalize
            face_normalized = (face_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            
            # Convert to NCHW format
            face_input = np.expand_dims(face_normalized.transpose(2, 0, 1), axis=0)
            
            # Inference
            outputs = self.session.run(None, {self.input_name: face_input})
            
            # DeepPixBiS output: [batch, 2] binary_logits (sau khi sửa convert_to_onnx.py)
            # Hoặc [batch, 1, H, W] pixel_map (nếu dùng model cũ)
            
            # Kiểm tra output format
            if len(outputs[0].shape) == 2:
                # Output là binary_logits: [batch, 2] với [fake_logit, real_logit]
                logits = outputs[0][0]  # [fake_logit, real_logit]
                
                # Apply softmax để convert logits → probabilities
                max_logit = np.max(logits)
                logits_shifted = logits - max_logit
                exp_logits = np.exp(logits_shifted)
                probs = exp_logits / np.sum(exp_logits)
                
                # Lấy probability của class "real" (index 1)
                score = float(probs[1])  # real_score (probability, range [0, 1])
                
                # Pixel map không có, return None
                pixel_map = None
                
            elif len(outputs[0].shape) == 4:
                # Fallback: Output là pixel_map (model cũ)
                pixel_map = outputs[0][0, 0]  # Shape: (H, W)
                raw_score = float(np.mean(pixel_map))
                
                # Kiểm tra nếu có giá trị âm (chưa có Sigmoid)
                if np.min(pixel_map) < 0:
                    pixel_map_sigmoid = 1.0 / (1.0 + np.exp(-pixel_map))
                    score = float(np.mean(pixel_map_sigmoid))
                else:
                    score = raw_score
                    
            else:
                # Fallback cho format khác
                pixel_map = outputs[0].flatten().reshape(14, 14) if outputs[0].size >= 196 else None
                raw_score = float(np.mean(pixel_map)) if pixel_map is not None else 0.5
                score = max(0.0, min(1.0, raw_score))
            
            return score, pixel_map
            
        except Exception as e:
            print(f"Error in Local branch prediction: {e}")
            return 0.5, None  # Neutral score on error


class TemporalBranch:
    """Blink Detection & Micro-motion Analysis"""
    
    def __init__(self, config: dict):
        self.blink_frames = config.get('blink_frames', 5)
        self.min_blinks = config.get('min_blinks', 1)
        self.ear_threshold = config.get('ear_threshold', 0.25)
        
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize MediaPipe Face Mesh: {e}")
        
        # Lưu EAR history
        self.ear_history = deque(maxlen=self.blink_frames)
        self.blink_count = 0
        self.consecutive_low_ear = 0
    
    def calculate_ear(self, landmarks) -> float:
        """
        Calculate Eye Aspect Ratio từ MediaPipe landmarks
        """
        # MediaPipe face mesh indices cho mắt
        # Left eye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        try:
            # Tính EAR cho mắt trái (6 điểm chính)
            left_eye_points = np.array([
                [landmarks[i].x, landmarks[i].y] 
                for i in [LEFT_EYE_INDICES[0], LEFT_EYE_INDICES[1], LEFT_EYE_INDICES[2], 
                          LEFT_EYE_INDICES[3], LEFT_EYE_INDICES[4], LEFT_EYE_INDICES[5]]
            ])
            left_ear = self._calculate_ear_for_eye(left_eye_points)
            
            # Tính EAR cho mắt phải
            right_eye_points = np.array([
                [landmarks[i].x, landmarks[i].y] 
                for i in [RIGHT_EYE_INDICES[0], RIGHT_EYE_INDICES[1], RIGHT_EYE_INDICES[2],
                          RIGHT_EYE_INDICES[3], RIGHT_EYE_INDICES[4], RIGHT_EYE_INDICES[5]]
            ])
            right_ear = self._calculate_ear_for_eye(right_eye_points)
            
            return (left_ear + right_ear) / 2.0
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.3  # Default value
    
    def _calculate_ear_for_eye(self, eye_points: np.ndarray) -> float:
        """
        Calculate EAR for one eye
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        if len(eye_points) < 6:
            return 0.3
        
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if h == 0:
            return 0.3
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def detect_blink(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect blink trong frame hiện tại
        Returns: (has_blink, ear_value)
        """
        if self.face_mesh is None:
            # Không có MediaPipe, giả định có blink
            return True, 0.3
        
        try:
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                return False, 0.0
            
            landmarks = results.multi_face_landmarks[0].landmark
            ear = self.calculate_ear(landmarks)
            self.ear_history.append(ear)
            
            # Phát hiện blink: EAR giảm đột ngột rồi tăng lại
            # Blink pattern: high -> low -> high
            has_blink = False
            if len(self.ear_history) >= 3:
                # Kiểm tra pattern blink
                if (self.ear_history[-3] > self.ear_threshold and
                    self.ear_history[-2] < self.ear_threshold and
                    self.ear_history[-1] > self.ear_threshold):
                    self.blink_count += 1
                    has_blink = True
            
            return has_blink, ear
            
        except Exception as e:
            print(f"Error in blink detection: {e}")
            return False, 0.0
    
    def validate(self) -> bool:
        """Kiểm tra đã có đủ số lần chớp mắt chưa"""
        return self.blink_count >= self.min_blinks
    
    def reset(self):
        """Reset state cho video stream mới"""
        self.ear_history.clear()
        self.blink_count = 0
        self.consecutive_low_ear = 0


class LivenessEnsemble:
    """Multi-stage Ensemble Liveness Detection"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize branches
        global_config = config['global_branch']
        self.global_branch = GlobalBranch(
            global_config['model_path'],
            global_config['threshold']
        )
        
        local_config = config['local_branch']
        self.local_branch = LocalBranch(
            local_config['model_path'],
            local_config['threshold']
        )
        
        self.temporal_branch = None
        if config.get('temporal_branch', {}).get('enabled', False):
            self.temporal_branch = TemporalBranch(config['temporal_branch'])
        
        self.fusion_method = config.get('fusion_method', 'weighted_sum')
        self.weights = {
            'global': global_config.get('weight', 0.4),
            'local': local_config.get('weight', 0.4),
            'temporal': config.get('temporal_branch', {}).get('weight', 0.2) if self.temporal_branch else 0.0
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight
    
    def predict(self, face_image: np.ndarray, frame_count: int = 0) -> Dict:
        """
        Predict liveness từ ensemble
        Args:
            face_image: Raw cropped face image (BGR format, KHÔNG alignment)
                       Quan trọng: Raw crop giữ nguyên high-frequency patterns cho Moiré detection
            frame_count: Current frame number (for temporal analysis)
        Returns:
            {
                'is_real': bool,
                'global_score': float,
                'local_score': float,
                'temporal_score': float,
                'final_score': float,
                'global_passed': bool,
                'local_passed': bool,
                'pixel_map': np.ndarray (optional)
            }
        """
        results = {}
        
        # Global branch
        global_score = self.global_branch.predict(face_image)
        results['global_score'] = global_score
        results['global_passed'] = global_score > self.config['global_branch']['threshold']
        
        # Local branch
        local_score, pixel_map = self.local_branch.predict(face_image)
        results['local_score'] = local_score
        results['local_passed'] = local_score > self.config['local_branch']['threshold']
        if pixel_map is not None:
            results['pixel_map'] = pixel_map
        
        # Temporal branch (nếu enabled)
        temporal_score = 0.0  # Mặc định 0.0 khi disabled
        if self.temporal_branch:
            has_blink, ear = self.temporal_branch.detect_blink(face_image)
            temporal_passed = self.temporal_branch.validate()
            temporal_score = 1.0 if temporal_passed else 0.5  # Partial score nếu chưa đủ blinks
            results['temporal_score'] = temporal_score
            results['has_blink'] = has_blink
            results['ear'] = ear
            results['blink_count'] = self.temporal_branch.blink_count
        # Nếu temporal branch disabled, không thêm temporal_score vào results
        
        # Fusion
        if self.fusion_method == 'weighted_sum':
            # Chỉ tính temporal nếu enabled (weight sẽ tự động = 0 nếu disabled)
            final_score = (
                self.weights['global'] * global_score +
                self.weights['local'] * local_score +
                self.weights['temporal'] * temporal_score
            )
        elif self.fusion_method == 'soft_voting':
            # Soft voting: Trung bình đơn giản của các scores (không dùng weights)
            scores = [global_score, local_score]
            # Chỉ thêm temporal score nếu enabled
            if self.temporal_branch:
                scores.append(temporal_score)
            final_score = sum(scores) / len(scores)  # Trung bình đơn giản
        elif self.fusion_method == 'voting':
            # Hard voting: Dựa trên passed/not passed
            votes = [
                results['global_passed'],
                results['local_passed']
            ]
            # Chỉ thêm temporal vote nếu enabled
            if self.temporal_branch:
                votes.append(temporal_score > 0.5)
            final_score = sum(votes) / len(votes)  # Chia cho số branches thực tế
        else:
            # AND logic (strict) - tất cả phải pass
            passed = results['global_passed'] and results['local_passed']
            if self.temporal_branch:
                passed = passed and (temporal_score > 0.5)
            final_score = 1.0 if passed else 0.0
        
        results['final_score'] = final_score
        # Lấy threshold từ config, mặc định 0.5 nếu không có
        final_threshold = self.config.get('final_threshold', 0.5)
        
        # Logic fusion: Chỉ cần final_score > threshold
        # Weighted sum đã kết hợp cả 2 branches, nên chỉ cần kiểm tra final_score
        # Nếu muốn nghiêm ngặt hơn, có thể yêu cầu cả 2 branch pass
        results['is_real'] = final_score > final_threshold
        
        # Optional: Nếu muốn logic nghiêm ngặt hơn (tránh false positive)
        # Uncomment dòng dưới để yêu cầu cả 2 branch pass:
        # both_branches_passed = results['global_passed'] and results['local_passed']
        # results['is_real'] = both_branches_passed and (final_score > final_threshold)
        
        return results


