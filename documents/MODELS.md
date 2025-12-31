# Hướng dẫn tải Models

Pipeline cần các model sau để hoạt động đầy đủ:

## 1. SCRFD (Face Detection)

Model này sẽ **tự động tải** khi chạy lần đầu với InsightFace:

```python
import insightface
model = insightface.model_zoo.get_model('buffalo_l')
model.prepare(ctx_id=-1)
```

Model sẽ được tải về `~/.insightface/models/` tự động.

## 2. MiniFASNetV2 (Global Branch)

**Cần tải thủ công** và convert sang ONNX:

### Cách 1: Từ repo Silent-Face-Anti-Spoofing

```bash
# Clone repo
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
cd Silent-Face-Anti-Spoofing

# Tải weights (nếu có)
# Convert sang ONNX
python convert_to_onnx.py --model_path models/minifasnet_v2.pth --output models/minifasnet_v2.onnx
```

### Cách 2: Sử dụng pre-trained ONNX (nếu có)

Tìm và tải từ:
- GitHub releases của Silent-Face-Anti-Spoofing
- Model zoo của các công ty eKYC

Đặt file vào: `models/minifasnet_v2.onnx`

## 3. DeepPixBiS (Local Branch)

**Cần tải thủ công** và convert sang ONNX:

### Từ repo DeepPixBiS

```bash
# Clone repo
git clone https://github.com/tding1/DeepPixBiS.git
cd DeepPixBiS

# Tải weights
# Convert sang ONNX
python convert_to_onnx.py --model_path models/deeppixbis.pth --output models/deeppixbis.onnx
```

Đặt file vào: `models/deeppixbis.onnx`

## 4. ArcFace (Recognition - Optional)

Model này sẽ **tự động tải** khi chạy lần đầu:

```python
import insightface
model = insightface.app.FaceAnalysis(name='arcface_r100_v1')
model.prepare(ctx_id=-1)
```

## Lưu ý

1. **Nếu không có model files**: Pipeline sẽ sử dụng dummy predictions để test. Kết quả sẽ không chính xác nhưng có thể test flow.

2. **Model paths trong config**: Đảm bảo đường dẫn trong `config/config.yaml` đúng:
   ```yaml
   liveness:
     global_branch:
       model_path: "models/minifasnet_v2.onnx"
     local_branch:
       model_path: "models/deeppixbis.onnx"
   ```

3. **ONNX Runtime**: Cần cài `onnxruntime` hoặc `onnxruntime-gpu`:
   ```bash
   pip install onnxruntime
   # Hoặc với GPU:
   pip install onnxruntime-gpu
   ```

## Test không có models

Pipeline vẫn có thể chạy test mode mà không cần models:

```bash
python tests/test_pipeline.py
```

Các module sẽ sử dụng dummy predictions để test flow.


