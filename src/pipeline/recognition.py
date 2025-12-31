"""
Face Recognition Module - ArcFace
Optional module cho 1-1 face matching
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Recognition will be limited.")


class FaceRecognizer:
    """ArcFace Face Recognition"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get('model_name', 'arcface_r100_v1')
        self.threshold = config.get('threshold', 0.6)
        
        self.model = None
        if INSIGHTFACE_AVAILABLE:
            try:
                # Load ArcFace model từ InsightFace
                self.model = insightface.app.FaceAnalysis(
                    name=self.model_name,
                    providers=['CPUExecutionProvider']  # Có thể thêm 'CUDAExecutionProvider' nếu có GPU
                )
                self.model.prepare(ctx_id=-1, det_size=(640, 640))
            except Exception as e:
                print(f"Warning: Could not load ArcFace model '{self.model_name}': {e}")
                print("Recognition will be limited.")
                self.model = None
        else:
            print("Warning: InsightFace not available. Recognition disabled.")
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding vector
        Args:
            face_image: Aligned face image (BGR format, thường 112x112)
        Returns:
            embedding vector (512-dim) hoặc None nếu lỗi
        """
        if self.model is None:
            # Dummy embedding
            return np.random.rand(512).astype(np.float32)
        
        try:
            # ArcFace model có thể tự detect và align, nhưng ta đã có aligned face
            # Nên ta có thể dùng trực tiếp
            faces = self.model.get(face_image)
            
            if len(faces) == 0:
                # Thử detect lại nếu không tìm thấy
                # Có thể do face đã được align nhưng model vẫn cần detect
                return None
            
            # Lấy embedding từ face đầu tiên
            embedding = faces[0].embedding
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        So sánh 2 embeddings bằng cosine similarity
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize vectors
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        # Clamp to [0, 1] (cosine similarity thường trong [-1, 1])
        similarity = (similarity + 1.0) / 2.0
        
        return float(similarity)
    
    def is_match(self, embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
        """
        Kiểm tra 2 face có match không
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        Returns:
            True nếu match (similarity > threshold)
        """
        similarity = self.compare_faces(embedding1, embedding2)
        return similarity > self.threshold
    
    def get_embedding_size(self) -> int:
        """Trả về kích thước embedding vector"""
        return self.config.get('embedding_size', 512)


