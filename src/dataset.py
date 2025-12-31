"""
Dataset Loader cho Face Liveness Detection
Hỗ trợ format: data/train/normal/ và data/train/spoof/
Training: Full image → Detect → Crop rộng (tránh distribution shift)
"""
import os
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional, List, Dict
import random
import yaml
import pickle
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available. CoarseDropout will be disabled.")


class FaceLivenessDataset(Dataset):
    """Dataset cho Face Liveness Detection với Raw Crop (không alignment)"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (112, 112),
        augment: bool = True,
        normalize: bool = True,
        context_expansion_scale: float = 2.7,
        use_raw_crop: bool = True,
        use_full_image_detection: bool = True
    ):
        """
        Args:
            data_dir: Đường dẫn đến thư mục data (chứa train/test/dev)
            split: 'train', 'test', hoặc 'dev'
            image_size: Kích thước ảnh output
            augment: Có augment data không (chỉ cho train)
            normalize: Có normalize ảnh không
            context_expansion_scale: Scale để mở rộng context (default: 2.7)
            use_raw_crop: Sử dụng raw crop (default: True, để giữ high-frequency patterns)
            use_full_image_detection: Detect từ full image thay vì padding simulation (default: True)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.normalize = normalize
        self.context_expansion_scale = context_expansion_scale
        self.use_raw_crop = use_raw_crop
        self.use_full_image_detection = use_full_image_detection
        
        # Load bbox cache nếu có (để tăng tốc training)
        self.bbox_cache = {}
        cache_path = Path(data_dir).parent / 'bbox_cache.pkl'
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    self.bbox_cache = pickle.load(f)
                print(f"  ✓ Loaded bbox cache: {len(self.bbox_cache)} images")
            except Exception as e:
                print(f"  Warning: Could not load bbox cache: {e}")
                self.bbox_cache = {}
        else:
            if self.use_full_image_detection:
                print(f"  ⚠ BBox cache not found at {cache_path}")
                print(f"     Run: python src/preprocess_bbox_cache.py --data-dir {data_dir}")
                print(f"     Training will be slower (detecting faces on-the-fly)")
        
        # Initialize detector (cần cho cả cache và on-the-fly detection)
        self.detector = None
        if self.use_full_image_detection:
            try:
                from pipeline.detection import SCRFDDetector
                # Load config từ config.yaml
                config_path = Path(data_dir).parent / 'config' / 'config.yaml'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    detector_config = config['pipeline']['detection']
                else:
                    # Fallback config
                    detector_config = {
                        'model_name': 'buffalo_l',
                        'input_size': [640, 640],
                        'conf_threshold': 0.5,
                        'nms_threshold': 0.4,
                        'context_expansion_scale': context_expansion_scale
                    }
                self.detector = SCRFDDetector(detector_config)
                print(f"  ✓ Detector initialized for full image detection")
            except Exception as e:
                print(f"Warning: Could not initialize detector for full image detection: {e}")
                print("  Falling back to padding simulation")
                self.use_full_image_detection = False
        
        # Load paths
        self.normal_dir = self.data_dir / split / 'normal'
        self.spoof_dir = self.data_dir / split / 'spoof'
        
        # Get all image paths
        self.normal_paths = sorted(list(self.normal_dir.glob('*.jpg')) + 
                                   list(self.normal_dir.glob('*.png')))
        self.spoof_paths = sorted(list(self.spoof_dir.glob('*.jpg')) + 
                                 list(self.spoof_dir.glob('*.png')))
        
        # Combine và label
        self.data = []
        for path in self.normal_paths:
            self.data.append((str(path), 1))  # 1 = real/normal
        for path in self.spoof_paths:
            self.data.append((str(path), 0))  # 0 = fake/spoof
        
        # Shuffle
        if self.split == 'train':
            random.shuffle(self.data)
        
        print(f"Loaded {len(self.data)} images from {split} split")
        print(f"  Normal/Real: {len(self.normal_paths)}")
        print(f"  Spoof/Fake: {len(self.spoof_paths)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Nếu không load được, tạo ảnh dummy
            image = np.zeros((112, 112, 3), dtype=np.uint8)
        
        # Convert BGR to RGB (nếu cần)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Giả sử image là BGR từ cv2.imread
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Raw Crop với Context Expansion
        # Option 1: Dùng bbox cache (NHANH - khuyến nghị)
        # Option 2: Detect on-the-fly (CHẬM - fallback)
        # Option 3: Padding simulation (fallback nếu không có detector)
        if self.use_raw_crop and self.context_expansion_scale > 1.0:
            # Ưu tiên: Dùng bbox cache (nhanh nhất)
            if image_path in self.bbox_cache:
                try:
                    cached_data = self.bbox_cache[image_path]
                    bbox = cached_data['bbox']
                    is_rotated = cached_data.get('rotated', False)
                    
                    # Nếu ảnh đã bị rotate khi detect, cần rotate lại ảnh gốc
                    if is_rotated:
                        image = cv2.rotate(image, cv2.ROTATE_180)
                    
                    # Extract raw face crop với context expansion từ cache
                    if self.detector is None:
                        # Cần detector để extract_raw_face, khởi tạo nếu chưa có
                        from pipeline.detection import SCRFDDetector
                        config_path = Path(self.data_dir).parent / 'config' / 'config.yaml'
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                config = yaml.safe_load(f)
                            detector_config = config['pipeline']['detection']
                            self.detector = SCRFDDetector(detector_config)
                    
                    raw_face = self.detector.extract_raw_face(
                        image,  # BGR format (đã rotate nếu cần)
                        bbox,
                        output_size=self.image_size
                    )
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(raw_face, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    # Fallback: Nếu có lỗi với cache, dùng padding simulation
                    h, w = image_rgb.shape[:2]
                    pad_h = int(h * (self.context_expansion_scale - 1.0) / 2.0)
                    pad_w = int(w * (self.context_expansion_scale - 1.0) / 2.0)
                    image_rgb = cv2.copyMakeBorder(
                        image_rgb, pad_h, pad_h, pad_w, pad_w,
                        cv2.BORDER_REFLECT_101
                    )
                    image_rgb = cv2.resize(image_rgb, self.image_size, interpolation=cv2.INTER_LINEAR)
            elif self.use_full_image_detection and self.detector is not None:
                # Fallback: Detect on-the-fly (chậm hơn)
                try:
                    # Detect face từ full image (thử ảnh gốc trước)
                    faces = self.detector.detect(image)  # image là BGR format
                    
                    # Nếu không detect được, thử rotate 180 độ (cho ảnh vertically flipped)
                    if len(faces) == 0:
                        image_rotated = cv2.rotate(image, cv2.ROTATE_180)
                        faces = self.detector.detect(image_rotated)
                        if len(faces) > 0:
                            # Detect được sau khi rotate → dùng ảnh đã rotate
                            image = image_rotated
                    
                    if len(faces) > 0:
                        # Extract raw face crop với context expansion
                        face_data = faces[0]
                        raw_face = self.detector.extract_raw_face(
                            image,  # BGR format (đã rotate nếu cần)
                            face_data['bbox'],
                            output_size=self.image_size
                        )
                        # Convert BGR to RGB
                        image_rgb = cv2.cvtColor(raw_face, cv2.COLOR_BGR2RGB)
                    else:
                        # Fallback: Nếu không detect được, dùng padding simulation
                        h, w = image_rgb.shape[:2]
                        pad_h = int(h * (self.context_expansion_scale - 1.0) / 2.0)
                        pad_w = int(w * (self.context_expansion_scale - 1.0) / 2.0)
                        image_rgb = cv2.copyMakeBorder(
                            image_rgb, pad_h, pad_h, pad_w, pad_w,
                            cv2.BORDER_REFLECT_101
                        )
                        image_rgb = cv2.resize(image_rgb, self.image_size, interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    # Fallback: Nếu có lỗi, dùng padding simulation
                    h, w = image_rgb.shape[:2]
                    pad_h = int(h * (self.context_expansion_scale - 1.0) / 2.0)
                    pad_w = int(w * (self.context_expansion_scale - 1.0) / 2.0)
                    image_rgb = cv2.copyMakeBorder(
                        image_rgb, pad_h, pad_h, pad_w, pad_w,
                        cv2.BORDER_REFLECT_101
                    )
                    image_rgb = cv2.resize(image_rgb, self.image_size, interpolation=cv2.INTER_LINEAR)
            else:
                # Fallback: Padding simulation (nếu dataset đã là face crops)
                h, w = image_rgb.shape[:2]
                pad_h = int(h * (self.context_expansion_scale - 1.0) / 2.0)
                pad_w = int(w * (self.context_expansion_scale - 1.0) / 2.0)
                image_rgb = cv2.copyMakeBorder(
                    image_rgb, pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_REFLECT_101
                )
                image_rgb = cv2.resize(image_rgb, self.image_size, interpolation=cv2.INTER_LINEAR)
        else:
            # Chỉ resize nếu không có context expansion
            image_rgb = cv2.resize(image_rgb, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        image = image_rgb
        
        # Augmentation (chỉ cho train)
        if self.augment:
            image = self._augment(image)
        
        # Convert to float và normalize
        image = image.astype(np.float32) / 255.0
        
        if self.normalize:
            # Normalize về [-1, 1]
            image = (image - 0.5) / 0.5
        
        # Convert to CHW format
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        
        return {
            'image': image,
            'label': label,
            'path': image_path
        }
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Data augmentation với CoarseDropout để chống overfitting"""
        # Sử dụng albumentations nếu có (tốt hơn)
        if ALBUMENTATIONS_AVAILABLE:
            # Tạo augmentation pipeline
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                # CoarseDropout: Đục lỗ ngẫu nhiên để ép model học texture ở nhiều vùng
                # Quan trọng cho Face Liveness: tránh model chỉ nhìn mắt, phải nhìn cả má, trán, mũi
                # API mới của albumentations: sử dụng ranges thay vì min/max riêng lẻ
                # Lỗ 32x32: Ép model học texture thay vì nhớ mặt (chống overfitting mạnh)
                A.CoarseDropout(
                    num_holes_range=(2, 12),        # Số lỗ: từ 2 đến 12
                    hole_height_range=(32, 32),      # Chiều cao lỗ: 32 pixels (cố định)
                    hole_width_range=(32, 32),        # Chiều rộng lỗ: 32 pixels (cố định)
                    fill=0,                          # Điền màu đen (0)
                    fill_mask=None,                 # Không mask
                    p=0.6                            # 60% áp dụng
                ),
            ])
            # Áp dụng augmentation
            augmented = transform(image=image)
            return augmented['image']
        else:
            # Fallback: augmentation cơ bản nếu không có albumentations
            # Random horizontal flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
            
            # Random brightness
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            
            # Random contrast
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=random.randint(-10, 10))
            
            return image


def create_dataloader(
    data_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (112, 112),
    augment: bool = True,
    shuffle: bool = True,
    context_expansion_scale: float = 2.7,
    use_raw_crop: bool = True,
    use_full_image_detection: bool = True  # Mới: Detect từ full image
) -> DataLoader:
    """
    Tạo DataLoader cho dataset với Raw Crop
    Args:
        context_expansion_scale: Scale để mở rộng context (default: 2.7)
        use_raw_crop: Sử dụng raw crop (không alignment) để giữ high-frequency patterns
        use_full_image_detection: Detect từ full image thay vì padding simulation (tránh distribution shift)
    """
    dataset = FaceLivenessDataset(
        data_dir=data_dir,
        split=split,
        image_size=image_size,
        augment=augment and (split == 'train'),
        normalize=True,
        context_expansion_scale=context_expansion_scale,
        use_raw_crop=use_raw_crop,
        use_full_image_detection=use_full_image_detection
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and (split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def analyze_dataset(data_dir: str):
    """
    Phân tích dataset: số lượng, kích thước, distribution
    """
    data_dir = Path(data_dir)
    
    print("="*50)
    print("Dataset Analysis")
    print("="*50)
    
    for split in ['train', 'test', 'dev']:
        normal_dir = data_dir / split / 'normal'
        spoof_dir = data_dir / split / 'spoof'
        
        normal_files = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png'))
        spoof_files = list(spoof_dir.glob('*.jpg')) + list(spoof_dir.glob('*.png'))
        
        print(f"\n{split.upper()}:")
        print(f"  Normal/Real: {len(normal_files)}")
        print(f"  Spoof/Fake: {len(spoof_files)}")
        print(f"  Total: {len(normal_files) + len(spoof_files)}")
        print(f"  Ratio: {len(normal_files)/(len(normal_files)+len(spoof_files))*100:.1f}% normal")
        
        # Sample một vài ảnh để kiểm tra kích thước
        if normal_files:
            sample = cv2.imread(str(normal_files[0]))
            if sample is not None:
                print(f"  Sample size: {sample.shape}")
    
    print("\n" + "="*50)

