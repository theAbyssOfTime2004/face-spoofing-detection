# Hướng dẫn Training Models

Dataset của bạn đã sẵn sàng để train! Cấu trúc:
```
data/
├── train/
│   ├── normal/  (1432 images - Real faces)
│   └── spoof/   (1368 images - Fake faces)
├── test/
│   ├── normal/
│   └── spoof/
└── dev/
    ├── normal/
    └── spoof/
```

## 📊 Phân tích Dataset

Trước khi train, hãy phân tích dataset:

```bash
python src/analyze_data.py --data-dir data --visualize --image-sizes
```

## 🚀 Training Global Branch (MiniFASNetV2)

### Bước 1: Train model

```bash
python src/train_global.py \
    --data-dir data \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001 \
    --image-size 80 \
    --save-dir checkpoints \
    --analyze
```

### Bước 2: Evaluate

```bash
python src/evaluate.py \
    --model-type global \
    --checkpoint checkpoints/best_global.pth \
    --data-dir data \
    --plot
```

### Bước 3: Convert sang ONNX

```bash
python src/convert_to_onnx.py \
    --model-type global \
    --checkpoint checkpoints/best_global.pth \
    --output models/minifasnet_v2.onnx \
    --image-size 80
```

## 🎯 Training Local Branch (DeepPixBiS)

### Bước 1: Train model

```bash
python src/train_local.py \
    --data-dir data \
    --batch-size 16 \
    --epochs 50 \
    --lr 0.001 \
    --image-size 224 \
    --save-dir checkpoints \
    --analyze
```

### Bước 2: Evaluate

```bash
python src/evaluate.py \
    --model-type local \
    --checkpoint checkpoints/best_local.pth \
    --data-dir data \
    --plot
```

### Bước 3: Convert sang ONNX

```bash
python src/convert_to_onnx.py \
    --model-type local \
    --checkpoint checkpoints/best_local.pth \
    --output models/deeppixbis.onnx \
    --image-size 224
```

## 📈 Monitoring Training

Training sẽ hiển thị:
- Loss và accuracy mỗi epoch
- Validation metrics
- Best model được lưu tự động

Checkpoints được lưu trong `checkpoints/`:
- `best_global.pth` / `best_local.pth`: Model tốt nhất
- `latest_global.pth` / `latest_local.pth`: Model mới nhất

## 🔧 Tùy chỉnh Training

### Thay đổi hyperparameters

Chỉnh sửa arguments khi chạy training:

```bash
# Learning rate thấp hơn
python src/train_global.py --lr 0.0001

# Batch size lớn hơn (nếu có GPU mạnh)
python src/train_global.py --batch-size 64

# Nhiều epochs hơn
python src/train_global.py --epochs 100
```

### Resume từ checkpoint

```bash
python src/train_global.py \
    --resume checkpoints/latest_global.pth \
    --epochs 100
```

## 📊 Evaluation Metrics

Script `evaluate.py` sẽ hiển thị:
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Độ chính xác khi dự đoán Real
- **Recall**: Tỷ lệ phát hiện Real
- **F1-Score**: Harmonic mean của Precision và Recall
- **Confusion Matrix**: Ma trận nhầm lẫn

## 🎨 Visualization

### Xem sample images

```bash
python src/analyze_data.py \
    --data-dir data \
    --split train \
    --visualize \
    --num-samples 8
```

### Plot confusion matrix

```bash
python src/evaluate.py \
    --model-type global \
    --checkpoint checkpoints/best_global.pth \
    --plot
```

## 💡 Tips

1. **GPU**: Sử dụng GPU để train nhanh hơn (tự động detect)
2. **Data Augmentation**: Đã được bật mặc định cho training
3. **Early Stopping**: Monitor validation accuracy, dừng nếu không cải thiện
4. **Learning Rate**: Có thể giảm nếu loss không giảm
5. **Batch Size**: Tăng nếu có GPU memory lớn

## 🔄 Workflow hoàn chỉnh

```bash
# 1. Phân tích dataset
python src/analyze_data.py --data-dir data --visualize

# 2. Train Global Branch
python src/train_global.py --data-dir data --epochs 50

# 3. Evaluate Global
python src/evaluate.py --model-type global --checkpoint checkpoints/best_global.pth

# 4. Convert Global to ONNX
python src/convert_to_onnx.py --model-type global \
    --checkpoint checkpoints/best_global.pth \
    --output models/minifasnet_v2.onnx

# 5. Train Local Branch
python src/train_local.py --data-dir data --epochs 50

# 6. Evaluate Local
python src/evaluate.py --model-type local --checkpoint checkpoints/best_local.pth

# 7. Convert Local to ONNX
python src/convert_to_onnx.py --model-type local \
    --checkpoint checkpoints/best_local.pth \
    --output models/deeppixbis.onnx

# 8. Test pipeline với models đã train
python src/main.py --input test_video.mp4
```

## 📝 Lưu ý

- Models được train sẽ phù hợp với dataset của bạn
- Có thể fine-tune từ pre-trained models nếu có
- Temporal Branch (blink detection) không cần train, dùng MediaPipe
- Sau khi convert sang ONNX, cập nhật `config/config.yaml` với đường dẫn models mới

## 🐛 Troubleshooting

### Out of Memory
- Giảm `--batch-size`
- Giảm `--image-size`

### Loss không giảm
- Giảm learning rate: `--lr 0.0001`
- Kiểm tra data quality: `python src/analyze_data.py --visualize`

### Accuracy thấp
- Tăng số epochs
- Kiểm tra class imbalance
- Thử data augmentation mạnh hơn

