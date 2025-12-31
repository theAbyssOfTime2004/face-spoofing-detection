# Face Liveness Detection Pipeline - SOTA 2025 Architecture

Pipeline phát hiện giả mạo khuôn mặt (Face Liveness Detection) sử dụng kiến trúc State-of-the-Art năm 2025 với chiến lược **Multi-stage Ensemble** và **Quality Aware**.

## 🎯 Tính năng chính

- ✅ **Quality Gate**: Lọc ảnh mờ, góc quay không hợp lệ
- ✅ **SCRFD Detection**: Phát hiện và căn chỉnh khuôn mặt chính xác
- ✅ **Multi-stage Liveness Ensemble**: 3 nhánh kết hợp
  - **Global Branch**: MiniFASNetV2 - Phân tích toàn cục
  - **Local Branch**: DeepPixBiS - Phân tích pixel-wise
  - **Temporal Branch**: Blink detection - Phát hiện chớp mắt
- ✅ **Face Recognition** (Optional): ArcFace cho 1-1 matching

## 📋 Yêu cầu

- Python >= 3.8
- CUDA (optional, cho GPU acceleration)

## 🚀 Cài đặt

### 1. Clone repository

```bash
cd /home/maidang/projects/fld-cake-assignment
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Tải models hoặc Train từ dataset của bạn

#### Option A: Train từ dataset của bạn (Khuyến nghị)

Bạn có dataset trong `data/` với cấu trúc:
```
data/
├── train/normal/  (Real faces)
├── train/spoof/   (Fake faces)
├── test/normal/
├── test/spoof/
└── dev/normal/
    └── dev/spoof/
```

**Quick Start Training:**
```bash
# Chạy toàn bộ pipeline training
./quick_start_training.sh

# Hoặc train từng model
python src/train_global.py --data-dir data --epochs 50
python src/train_local.py --data-dir data --epochs 50
```

Xem chi tiết trong [TRAINING.md](TRAINING.md)

#### Option B: Sử dụng pre-trained models

Pipeline cần các model sau (sẽ tự động tải khi chạy lần đầu với InsightFace):

- **SCRFD**: Tự động tải từ InsightFace model zoo
- **MiniFASNetV2**: Cần tải và convert sang ONNX
- **DeepPixBiS**: Cần tải và convert sang ONNX
- **ArcFace** (optional): Tự động tải từ InsightFace model zoo

**Lưu ý**: Các model ONNX cần được đặt trong thư mục `models/`:
- `models/minifasnet_v2.onnx`
- `models/deeppixbis.onnx`

Nếu không có model files, pipeline sẽ sử dụng dummy predictions để test.

## 📁 Cấu trúc dự án

```
fld-cake-assignment/
├── config/
│   └── config.yaml          # Cấu hình pipeline
├── src/
│   ├── pipeline/
│   │   ├── quality_gate.py      # Quality Gate module
│   │   ├── detection.py         # SCRFD Detection & Alignment
│   │   ├── liveness_ensemble.py # Multi-stage Ensemble
│   │   ├── recognition.py       # ArcFace Recognition (optional)
│   │   └── pipeline.py          # Pipeline chính
│   └── main.py                  # Entry point
├── models/                      # Thư mục chứa model weights
├── requirements.txt
└── README.md
```

## ⚙️ Cấu hình

Chỉnh sửa `config/config.yaml` để tùy chỉnh:

```yaml
pipeline:
  quality_gate:
    max_yaw: 20.0        # Góc quay tối đa (degrees)
    blur_threshold: 100.0 # Ngưỡng blur detection
    
  liveness:
    global_branch:
      threshold: 0.9      # Ngưỡng cho Global branch
      weight: 0.4
    local_branch:
      threshold: 0.8      # Ngưỡng cho Local branch
      weight: 0.4
    temporal_branch:
      enabled: true       # Bật/tắt temporal analysis
      min_blinks: 1      # Số lần chớp mắt tối thiểu
```

## 🎮 Sử dụng

### Xử lý từ camera

```bash
python src/main.py --camera
```

### Xử lý từ video file

```bash
python src/main.py --input video.mp4
```

### Xử lý với options

```bash
# Hiển thị chi tiết scores
python src/main.py --input video.mp4 --show-details

# Giới hạn số frames
python src/main.py --input video.mp4 --max-frames 100

# Lưu output video
python src/main.py --input video.mp4 --output output.mp4

# Không hiển thị window (headless mode)
python src/main.py --input video.mp4 --no-display
```

### Sử dụng config tùy chỉnh

```bash
python src/main.py --input video.mp4 --config config/custom_config.yaml
```

## 📊 Kết quả

Pipeline trả về:

- **Status**: `accepted` (real) hoặc `rejected` (fake)
- **Confidence score**: Điểm tin cậy (0-1)
- **Detailed scores**: Global, Local, Temporal scores
- **Statistics**: Thống kê xử lý

### Ví dụ output:

```
=== KẾT QUẢ CUỐI CÙNG ===
Status: accepted
Message: Face is REAL
Confidence: 0.892
Pass Rate: 85.00%

=== THỐNG KÊ ===
Total frames: 100
Quality passed: 95 (95.00%)
Detection passed: 90 (90.00%)
Liveness passed: 85 (85.00%)
Final accepted: 85 (85.00%)
```

## 🔧 Kiến trúc Pipeline

```
Input Video Stream
    ↓
Quality Gate (Blur/Pose Check)
    ↓
SCRFD Detection & Alignment
    ↓
Liveness Ensemble
    ├── Global Branch (MiniFASNetV2)
    ├── Local Branch (DeepPixBiS)
    └── Temporal Branch (Blink Detection)
    ↓
Fusion & Decision
    ↓
Real/Fake Result
```

## 🛡️ Chống tấn công

Pipeline có khả năng chống:

- ✅ **Print Attack**: Nhờ Local Branch (DeepPixBiS) phân tích pixel
- ✅ **Replay Attack**: Nhờ Global Branch (MiniFASNet) phát hiện Moiré pattern
- ✅ **3D Mask Attack**: Nhờ Quality Gate và độ sâu ảnh
- ✅ **Static Image**: Nhờ Temporal Branch yêu cầu chớp mắt

## 📝 Lưu ý

1. **Models**: Cần tải và convert các model ONNX (MiniFASNetV2, DeepPixBiS) vào thư mục `models/`
2. **GPU**: Để tăng tốc, cài `onnxruntime-gpu` và có CUDA
3. **Temporal Branch**: Cần xử lý nhiều frames liên tiếp để phát hiện chớp mắt
4. **InsightFace**: SCRFD và ArcFace sẽ tự động tải model khi chạy lần đầu

## 🐛 Troubleshooting

### Lỗi: "InsightFace not available"
```bash
pip install insightface
```

### Lỗi: "ONNX Runtime not available"
```bash
pip install onnxruntime
# Hoặc với GPU:
pip install onnxruntime-gpu
```

### Lỗi: "Model not found"
- Đảm bảo model files được đặt đúng trong `models/`
- Hoặc pipeline sẽ dùng dummy predictions để test

### Lỗi: "Cannot open camera"
- Kiểm tra camera index: `--camera-id 1` (thử các index khác)
- Kiểm tra quyền truy cập camera

## 🎓 Training với Dataset của bạn

Bạn có thể train models từ dataset của riêng bạn! Xem hướng dẫn chi tiết:

- **[TRAINING.md](TRAINING.md)**: Hướng dẫn training đầy đủ
- **Quick Start**: `./quick_start_training.sh`

### Dataset Format

Dataset cần có cấu trúc:
```
data/
├── train/
│   ├── normal/  (Real faces - .jpg hoặc .png)
│   └── spoof/   (Fake faces - .jpg hoặc .png)
├── test/
│   ├── normal/
│   └── spoof/
└── dev/
    ├── normal/
    └── spoof/
```

### Training Commands

```bash
# Phân tích dataset
python src/analyze_data.py --data-dir data --visualize

# Train Global Branch
python src/train_global.py --data-dir data --epochs 50

# Train Local Branch
python src/train_local.py --data-dir data --epochs 50

# Evaluate models
python src/evaluate.py --model-type global --checkpoint checkpoints/best_global.pth

# Convert to ONNX
python src/convert_to_onnx.py --model-type global \
    --checkpoint checkpoints/best_global.pth \
    --output models/minifasnet_v2.onnx
```

## 📚 Tài liệu tham khảo

- [InsightFace](https://github.com/deepinsight/insightface)
- [SCRFD Paper](https://arxiv.org/abs/2105.04714)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)

## 📄 License

MIT License

## 👥 Contributors

Developed for eKYC applications with SOTA 2025 architecture.


