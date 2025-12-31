"""
Quality Gate Module - Bộ lọc chất lượng ảnh đầu vào
Kiểm tra blur, pose, và chất lượng khuôn mặt
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Pose estimation will be limited.")


class QualityGate:
    """Bộ lọc chất lượng ảnh đầu vào"""
    
    def __init__(self, config: dict):
        self.max_yaw = config.get('max_yaw', 20.0)
        self.max_pitch = config.get('max_pitch', 20.0)
        self.max_roll = config.get('max_roll', 20.0)
        self.blur_threshold = config.get('blur_threshold', 100.0)
        self.min_face_quality = config.get('min_face_quality', 0.5)
        
        # MediaPipe Face Mesh cho pose estimation (optional)
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize MediaPipe Face Mesh: {e}")
                self.face_mesh = None
    
    def check_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Kiểm tra độ mờ bằng Laplacian variance
        Returns: (is_sharp, blur_score)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = laplacian_var > self.blur_threshold
        return is_sharp, laplacian_var
    
    def estimate_pose_from_landmarks(self, landmarks: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Ước tính góc quay mặt (Yaw, Pitch, Roll) từ 2D landmarks
        Sử dụng phương pháp đơn giản dựa trên vị trí landmarks
        """
        if landmarks is None or len(landmarks) < 5:
            return None, None, None
        
        landmarks = np.array(landmarks)
        
        # Giả sử landmarks có format: [[x1, y1], [x2, y2], ..., [x5, y5]]
        # Landmarks thường là: left_eye, right_eye, nose, left_mouth, right_mouth
        
        if landmarks.shape[0] >= 5:
            # Tính toán đơn giản dựa trên vị trí mắt và mũi
            # Yaw: dựa trên sự khác biệt giữa 2 mắt
            # Pitch: dựa trên vị trí mũi so với mắt
            # Roll: dựa trên góc nghiêng của đường nối 2 mắt
            
            # Giả sử landmarks[0] và landmarks[1] là 2 mắt
            if landmarks.shape[0] >= 2:
                eye_left = landmarks[0]
                eye_right = landmarks[1]
                
                # Tính Roll (góc nghiêng)
                dx = eye_right[0] - eye_left[0]
                dy = eye_right[1] - eye_left[1]
                roll = np.degrees(np.arctan2(dy, dx))
                
                # Tính Yaw (góc quay ngang) - đơn giản hóa
                eye_center_x = (eye_left[0] + eye_right[0]) / 2
                face_width = np.linalg.norm(eye_right - eye_left)
                if face_width > 0:
                    # Giả sử face center là trung tâm ảnh hoặc từ landmarks
                    yaw = 0.0  # Cần implement đầy đủ hơn
                else:
                    yaw = 0.0
                
                # Tính Pitch (góc quay dọc) - đơn giản hóa
                if landmarks.shape[0] >= 3:
                    nose = landmarks[2]
                    eye_center_y = (eye_left[1] + eye_right[1]) / 2
                    pitch = 0.0  # Cần implement đầy đủ hơn
                else:
                    pitch = 0.0
                
                return yaw, pitch, roll
        
        return None, None, None
    
    def estimate_pose_mediapipe(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Ước tính pose sử dụng MediaPipe Face Mesh (nếu available)
        """
        if self.face_mesh is None:
            return None, None, None
        
        try:
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None, None, None
            
            # MediaPipe có thể tính toán pose từ 3D landmarks
            # Đây là implementation đơn giản, có thể cải thiện
            landmarks = results.multi_face_landmarks[0]
            
            # Lấy một số landmarks quan trọng
            h, w = image.shape[:2]
            
            # Tính toán đơn giản từ landmarks
            # (Cần implement đầy đủ công thức tính yaw, pitch, roll từ 3D landmarks)
            # Tạm thời return None để dùng phương pháp khác
            return None, None, None
            
        except Exception as e:
            print(f"Error in MediaPipe pose estimation: {e}")
            return None, None, None
    
    def check_pose(self, yaw: Optional[float], pitch: Optional[float], roll: Optional[float]) -> bool:
        """Kiểm tra góc quay mặt có hợp lệ không"""
        if yaw is None or pitch is None or roll is None:
            # Nếu không thể estimate pose, cho phép qua (có thể cải thiện sau)
            return True
        
        return (abs(yaw) < self.max_yaw and 
                abs(pitch) < self.max_pitch and 
                abs(roll) < self.max_roll)
    
    def validate(self, image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Tuple[bool, Dict]:
        """
        Validate chất lượng ảnh đầu vào
        Args:
            image: Input image (BGR format)
            landmarks: Optional 2D landmarks array (shape: [N, 2])
        Returns:
            (is_valid, quality_info)
        """
        quality_info = {}
        
        # 1. Kiểm tra blur
        is_sharp, blur_score = self.check_blur(image)
        quality_info['blur_score'] = float(blur_score)
        quality_info['is_sharp'] = is_sharp
        
        if not is_sharp:
            quality_info['failed_reason'] = 'blur'
            return False, quality_info
        
        # 2. Kiểm tra pose (nếu có landmarks)
        yaw, pitch, roll = None, None, None
        if landmarks is not None:
            yaw, pitch, roll = self.estimate_pose_from_landmarks(landmarks)
        elif self.face_mesh is not None:
            # Thử dùng MediaPipe nếu không có landmarks
            yaw, pitch, roll = self.estimate_pose_mediapipe(image)
        
        if yaw is not None and pitch is not None and roll is not None:
            quality_info['yaw'] = float(yaw)
            quality_info['pitch'] = float(pitch)
            quality_info['roll'] = float(roll)
            
            if not self.check_pose(yaw, pitch, roll):
                quality_info['failed_reason'] = 'pose'
                return False, quality_info
        
        quality_info['passed'] = True
        return True, quality_info


