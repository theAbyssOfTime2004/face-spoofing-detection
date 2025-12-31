"""
SCRFD Face Detection & Alignment Module
Sử dụng InsightFace SCRFD model cho detection và alignment
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Please install: pip install insightface")


class SCRFDDetector:
    """SCRFD Face Detection với 5 keypoints"""
    
    def __init__(self, config: dict):
        self.config = config
        self.input_size = tuple(config.get('input_size', [640, 640]))
        self.conf_threshold = config.get('conf_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.4)
        self.model_name = config.get('model_name', 'buffalo_l')
        self.context_expansion_scale = config.get('context_expansion_scale', 1.0)  # Mở rộng bbox để thấy context
        
        self.model = None
        if INSIGHTFACE_AVAILABLE:
            try:
                if 'buffalo' in self.model_name.lower():
                    # Load FaceAnalysis với chỉ detection module (tiết kiệm RAM)
                    # allowed_modules=['detection'] chỉ load det_10g.onnx, bỏ qua w600k_r50.onnx
                    from insightface import app
                    face_app = app.FaceAnalysis(
                        name=self.model_name, 
                        allowed_modules=['detection'],  # Chỉ load detection, không load recognition
                        providers=['CPUExecutionProvider']
                    )
                    face_app.prepare(ctx_id=-1, det_size=self.input_size)
                    
                    # Lấy "Lõi" Detection ra (SCRFD object)
                    # Đây chính là object SCRFD có method .detect()
                    if hasattr(face_app, 'det_model'):
                        self.model = face_app.det_model
                    elif hasattr(face_app, 'detection_model'):
                        self.model = face_app.detection_model
                    else:
                        raise AttributeError("FaceAnalysis does not have det_model or detection_model")
                else:
                    # Load SCRFD trực tiếp (không phải buffalo pack)
                    self.model = insightface.model_zoo.get_model(self.model_name)
                    self.model.prepare(ctx_id=-1, input_size=self.input_size)
            except Exception as e:
                print(f"Warning: Could not load SCRFD model '{self.model_name}': {e}")
                print("Falling back to basic detection. Please ensure model is available.")
                self.model = None
        else:
            print("Warning: InsightFace not available. Detection will be limited.")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces và landmarks
        Luôn dùng detector.detect() trực tiếp để tối ưu hiệu năng
        Args:
            image: Input image (BGR format)
        Returns:
            List of {bbox, landmarks, confidence}
        """
        if self.model is None:
            # Fallback: sử dụng OpenCV Haar Cascade (basic)
            return self._detect_fallback(image)
        
        try:
            # SCRFD/RetinaFace detect API:
            # detect(image, max_num=0, metric='default')
            # Không có parameter 'threshold' - threshold được lọc sau khi detect
            bboxes, kpss = self.model.detect(image, max_num=0, metric='default')
            
            results = []
            if bboxes is not None and len(bboxes) > 0:
                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    # bbox format: [x1, y1, x2, y2, score]
                    x1, y1, x2, y2, conf = bbox
                    
                    # Lọc theo confidence threshold (sau khi detect)
                    if conf < self.conf_threshold:
                        continue  # Bỏ qua face có confidence thấp
                    
                    # Extract landmarks cho face này
                    # kpss là [N, 5, 2] - 5 landmarks với tọa độ [x, y]
                    if kpss is not None and i < len(kpss):
                        kps = kpss[i]  # Shape: [5, 2]
                        landmark_2d = kps.astype(int).tolist()
                    else:
                        landmark_2d = None
                    
                    results.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'landmarks': landmark_2d,
                        'confidence': float(conf)
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in SCRFD detection: {e}")
            import traceback
            traceback.print_exc()
            return self._detect_fallback(image)
    
    def _detect_fallback(self, image: np.ndarray) -> List[Dict]:
        """Fallback detection sử dụng OpenCV Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load Haar Cascade (có sẵn trong OpenCV)
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            results = []
            for (x, y, w, h) in faces:
                # Tạo landmarks giả (center của face)
                # Không chính xác nhưng đủ để test
                landmarks = [
                    [x + w * 0.3, y + h * 0.3],  # left eye
                    [x + w * 0.7, y + h * 0.3],  # right eye
                    [x + w * 0.5, y + h * 0.5],  # nose
                    [x + w * 0.3, y + h * 0.7],  # left mouth
                    [x + w * 0.7, y + h * 0.7],  # right mouth
                ]
                
                results.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'landmarks': landmarks,
                    'confidence': 0.8  # Giả định
                })
            
            return results
        except Exception as e:
            print(f"Error in fallback detection: {e}")
            return []
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray, 
                   output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        Căn chỉnh mặt dựa trên 5 landmarks sử dụng Similarity Transform
        Args:
            image: Input image (BGR format)
            landmarks: 5 landmarks array (shape: [5, 2])
            output_size: Output size (width, height)
        Returns:
            Aligned face image
        """
        if landmarks is None or len(landmarks) < 5:
            # Nếu không có landmarks, chỉ resize
            return cv2.resize(image, output_size)
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Landmarks chuẩn cho face alignment (theo InsightFace/ArcFace)
        # Format: [left_eye, right_eye, nose, left_mouth, right_mouth]
        dst_landmarks = np.array([
            [30.2946, 51.6963],  # left eye
            [65.5318, 51.5014],  # right eye
            [48.0252, 71.7366],  # nose
            [33.5493, 92.3655],  # left mouth
            [62.7299, 92.2041]   # right mouth
        ], dtype=np.float32)
        
        # Scale landmarks theo output size
        dst_landmarks[:, 0] *= (output_size[0] / 96.0)
        dst_landmarks[:, 1] *= (output_size[1] / 112.0)
        
        # Tính Similarity Transform matrix từ 2 mắt và mũi
        # Sử dụng 3 điểm đầu tiên (2 mắt + mũi)
        src_points = landmarks[:3].astype(np.float32)
        dst_points = dst_landmarks[:3].astype(np.float32)
        
        # Tính Affine Transform (có thể dùng getAffineTransform)
        transform_matrix = cv2.getAffineTransform(src_points, dst_points)
        
        # Áp dụng transform
        aligned_face = cv2.warpAffine(
            image, transform_matrix, output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return aligned_face
    
    def extract_face(self, image: np.ndarray, bbox: List[int], 
                    landmarks: Optional[np.ndarray] = None,
                    align: bool = True,
                    output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        Extract và align face từ image với Context Expansion
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            landmarks: Optional landmarks
            align: Whether to align face
            output_size: Output size
        Returns:
            Extracted/aligned face (có thể bao gồm context xung quanh)
        """
        x1, y1, x2, y2 = bbox
        
        # Context Expansion: Mở rộng bbox để thấy context (viền giấy, thiết bị, ngón tay)
        if self.context_expansion_scale > 1.0:
            # Tính center và size của bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            
            # Mở rộng theo scale
            new_width = width * self.context_expansion_scale
            new_height = height * self.context_expansion_scale
            
            # Tính bbox mới
            x1 = center_x - new_width / 2.0
            y1 = center_y - new_height / 2.0
            x2 = center_x + new_width / 2.0
            y2 = center_y + new_height / 2.0
        
        # Đảm bảo bbox trong bounds
        h, w = image.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        face_roi = image[y1:y2, x1:x2]
        
        if align and landmarks is not None:
            # Điều chỉnh landmarks về tọa độ trong face_roi (sau khi expand)
            adjusted_landmarks = np.array(landmarks) - np.array([x1, y1])
            return self.align_face(face_roi, adjusted_landmarks, output_size)
        else:
            return cv2.resize(face_roi, output_size)
    
    def extract_raw_face(self, image: np.ndarray, bbox: List[int],
                        output_size: Tuple[int, int] = (80, 80)) -> np.ndarray:
        """
        Extract raw face crop KHÔNG alignment (cho Liveness Model)
        Giữ nguyên pixel gốc để bảo toàn high-frequency patterns (Moiré)
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            output_size: Output size (thường 80x80 cho Global Branch)
        Returns:
            Raw cropped face (chỉ resize, không warp)
        """
        x1, y1, x2, y2 = bbox
        
        # Context Expansion: Mở rộng bbox để thấy context
        if self.context_expansion_scale > 1.0:
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            
            new_width = width * self.context_expansion_scale
            new_height = height * self.context_expansion_scale
            
            x1 = center_x - new_width / 2.0
            y1 = center_y - new_height / 2.0
            x2 = center_x + new_width / 2.0
            y2 = center_y + new_height / 2.0
        
        # Đảm bảo bbox trong bounds
        h, w = image.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        # Crop raw (KHÔNG align, KHÔNG warp)
        face_roi = image[y1:y2, x1:x2]
        
        # Chỉ resize (giữ nguyên pixel patterns)
        raw_face = cv2.resize(face_roi, output_size, interpolation=cv2.INTER_LINEAR)
        
        return raw_face


