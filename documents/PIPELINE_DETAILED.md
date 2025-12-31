# Mô Tả Chi Tiết Pipeline Face Liveness Detection

## 📋 Tổng Quan Pipeline

Pipeline hiện tại là một **Static Ensemble SOTA** cho Face Liveness Detection, tập trung vào việc phân tích ảnh tĩnh với 2 nhánh chính: **Global Branch** (Frequency Analysis) và **Local Branch** (Patch-based Analysis).

Pipeline bao gồm 4 giai đoạn chính:
1. **Quality Gate** (Tùy chọn)
2. **Face Detection & Alignment** với Context Expansion
3. **Liveness Detection Ensemble** (Global + Local)
4. **Final Decision** với logic nghiêm ngặt

---

## 🔍 Giai Đoạn 1: Quality Gate (Tùy Chọn)

### Mục Đích
Lọc ảnh đầu vào để đảm bảo chất lượng trước khi chạy liveness detection.

### Quy Trình Chi Tiết

#### 1.1. Blur Detection (Laplacian Variance)

**Cơ chế:**
```python
# Tính toán độ sắc nét của ảnh
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
is_sharp = laplacian_var > blur_threshold  # Default: 100.0
```

**Giải thích:**
- **Laplacian operator** phát hiện biến thiên cường độ pixel
- Ảnh sắc nét có **variance cao** (nhiều biến thiên)
- Ảnh mờ có **variance thấp** (ít biến thiên)
- **Threshold**: 100.0 (có thể điều chỉnh trong config)

**Kết quả:**
- `blur_score < 100.0` → Quality Gate **FAIL**
- `blur_score >= 100.0` → Quality Gate **PASS**

#### 1.2. Pose Estimation (Yaw, Pitch, Roll)

**Kiểm tra góc quay mặt:**
- **Yaw** < 20° (quay trái/phải)
- **Pitch** < 20° (ngẩng/cúi)
- **Roll** < 20° (nghiêng)

**Nếu vượt ngưỡng** → Quality Gate **FAIL**

### Kết Quả Quality Gate

- **PASS**: Tiếp tục pipeline
- **FAIL**: Có thể bỏ qua nếu dùng `--skip-quality` flag

---

## 🎯 Giai Đoạn 2: Face Detection & Alignment

### 2.1. Face Detection (SCRFD)

#### Model: SCRFD (Sample and Computation Redistribution for Face Detection)

**Thông tin:**
- Sử dụng từ InsightFace `buffalo_l` pack
- Chỉ load **detection module** (`det_10g.onnx`), không load recognition
- **Input size**: 640x640
- **Output**: Bounding boxes + 5 keypoints

#### Quy Trình Detection:

```python
# SCRFD detect API
bboxes, kpss = model.detect(image, max_num=0, metric='default')
# bboxes: [N, 5] - [x1, y1, x2, y2, confidence]
# kpss: [N, 5, 2] - 5 landmarks [x, y] cho mỗi face
```

**5 Landmarks:**
1. **Left eye** (mắt trái)
2. **Right eye** (mắt phải)
3. **Nose tip** (đầu mũi)
4. **Left mouth corner** (khóe miệng trái)
5. **Right mouth corner** (khóe miệng phải)

#### Confidence Filtering:
- Lọc theo `conf_threshold` (default: 0.5)
- Chỉ giữ faces có `confidence >= threshold`

### 2.2. Context Expansion ⭐

#### Mục Đích
Mở rộng bbox để bao gồm **context xung quanh** (viền giấy, thiết bị, ngón tay, background).

#### Quy Trình:

```python
# Tính center và size của bbox gốc
center_x = (x1 + x2) / 2.0
center_y = (y1 + y2) / 2.0
width = x2 - x1
height = y2 - y1

# Mở rộng theo scale (default: 2.0)
new_width = width * context_expansion_scale  # 2.0x
new_height = height * context_expansion_scale  # 2.0x

# Tính bbox mới
x1_new = center_x - new_width / 2.0
y1_new = center_y - new_height / 2.0
x2_new = center_x + new_width / 2.0
y2_new = center_y + new_height / 2.0
```

**Scale**: 2.0 (bbox gốc được mở rộng **2 lần**)

**Lợi ích:**
- ✅ Phát hiện **viền giấy in** (nếu kẻ gian cầm ảnh giơ lên)
- ✅ Phát hiện **viền điện thoại/tablet** (nếu kẻ gian giơ điện thoại)
- ✅ Thấy **ngón tay cầm thiết bị**
- ✅ Phát hiện **background bị biến dạng**

### 2.3. Face Alignment (Similarity Transform)

#### Mục Đích
Căn chỉnh khuôn mặt về **góc nhìn chuẩn** để model dễ phân tích.

#### Quy Trình:

```python
# Landmarks chuẩn (theo InsightFace/ArcFace)
dst_landmarks = [
    [30.2946, 51.6963],  # left eye
    [65.5318, 51.5014],  # right eye
    [48.0252, 71.7366],  # nose
    [33.5493, 92.3655],  # left mouth
    [62.7299, 92.2041]   # right mouth
]

# Tính Affine Transform từ 3 điểm (2 mắt + mũi)
transform_matrix = cv2.getAffineTransform(src_points, dst_points)

# Áp dụng transform
aligned_face = cv2.warpAffine(image, transform_matrix, (112, 112))
```

**Input**: Face ROI sau context expansion  
**Output**: 112x112 RGB, đã căn chỉnh  
**Phương pháp**: Affine Transform dùng 3 điểm (2 mắt + mũi)

---

## 🧠 Giai Đoạn 3: Liveness Detection Ensemble

### Tổng Quan

Ensemble gồm **2 nhánh chính**:
- **Global Branch** (MiniFASNetV2): Phân tích toàn cục
- **Local Branch** (DeepPixBiS): Phân tích pixel-wise

**Temporal Branch** đã tắt (chỉ dùng cho video/stream).

---

## 🔬 Model 1: Global Branch (MiniFASNetV2)

### Mục Tiêu

Phát hiện dấu hiệu spoof ở mức **toàn cục**:
- **Moiré patterns** (lưới tần số từ màn hình)
- **Geometric distortions** (biến dạng do camera)
- **Phone bezels** (viền điện thoại)
- **Screen reflections** (phản xạ màn hình)

### Kiến Trúc Chi Tiết

#### Input Preprocessing:

```python
# 1. Convert BGR -> RGB
face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

# 2. Resize về 80x80 (input size của MiniFASNetV2)
face_resized = cv2.resize(face_rgb, (80, 80))

# 3. Normalize: (pixel / 255.0 - 0.5) / 0.5
# Kết quả: pixel values trong range [-1, 1]
face_normalized = (face_resized.astype(np.float32) / 255.0 - 0.5) / 0.5

# 4. Convert to NCHW: [batch, channels, height, width]
face_input = np.expand_dims(face_normalized.transpose(2, 0, 1), axis=0)
# Shape: [1, 3, 80, 80]
```

#### Network Architecture:

```
Input: [1, 3, 80, 80]

┌─────────────────────────────────────┐
│ Conv Block 1                        │
│ - Conv2d(3→16, kernel=3, stride=2)  │ → [1, 16, 40, 40]
│ - BatchNorm2d(16)                   │
│ - ReLU                               │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Conv Block 2                        │
│ - Conv2d(16→32, kernel=3, stride=2) │ → [1, 32, 20, 20]
│ - BatchNorm2d(32)                   │
│ - ReLU                               │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Conv Block 3                        │
│ - Conv2d(32→64, kernel=3, stride=2) │ → [1, 64, 10, 10]
│ - BatchNorm2d(64)                   │
│ - ReLU                               │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Conv Block 4                        │
│ - Conv2d(64→128, kernel=3, stride=2)│ → [1, 128, 5, 5]
│ - BatchNorm2d(128)                  │
│ - ReLU                               │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Global Average Pooling               │
│ - AdaptiveAvgPool2d(1)               │ → [1, 128, 1, 1]
│ - Flatten                            │ → [1, 128]
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Classifier                           │
│ - Linear(128→64)                     │ → [1, 64]
│ - ReLU                               │
│ - Dropout(0.5)                       │
│ - Linear(64→2)                      │ → [1, 2]
└─────────────────────────────────────┘

Output: [1, 2] - [fake_score, real_score]
```

#### Chi Tiết Từng Layer:

1. **Conv Block 1** (3→16 channels):
   - Kernel: 3x3, stride=2, padding=1
   - Giảm kích thước: 80x80 → 40x40
   - Tăng channels: 3 → 16

2. **Conv Block 2** (16→32 channels):
   - 40x40 → 20x20
   - Channels: 16 → 32

3. **Conv Block 3** (32→64 channels):
   - 20x20 → 10x10
   - Channels: 32 → 64

4. **Conv Block 4** (64→128 channels):
   - 10x10 → 5x5
   - Channels: 64 → 128

5. **Global Average Pooling**:
   - 5x5 → 1x1
   - Tạo feature vector 128D

6. **Classifier**:
   - Linear(128→64) + ReLU + Dropout(0.5)
   - Linear(64→2) → [fake_score, real_score]

#### Output Processing:

```python
# Output từ model: [batch, 2]
if len(outputs[0].shape) == 2:
    score = float(outputs[0][0][1])  # Lấy real_score (index 1)
else:
    # Fallback: nếu output format khác
    score = float(outputs[0].flatten()[0])
    if score < 0.5:
        score = 1.0 - score  # Đảo ngược nếu là fake_score

# Clamp về [0, 1]
score = max(0.0, min(1.0, score))
```

#### Tại Sao Hiệu Quả?

- ✅ **Nhìn toàn cục**: Phát hiện patterns trên toàn ảnh (Moiré, viền thiết bị)
- ✅ **Frequency analysis**: Conv layers học các tần số đặc trưng
- ✅ **Lightweight**: 80x80 input, ít tham số, inference nhanh

---

## 🔍 Model 2: Local Branch (DeepPixBiS)

### Mục Tiêu

Phân tích **pixel-wise** để phát hiện:
- **Skin texture** (kết cấu da thật vs in/màn hình)
- **Screen pixels** (pixel grid của màn hình)
- **Reflections** (phản xạ trên bề mặt)
- **Print artifacts** (artifacts từ in ấn)

### Kiến Trúc Chi Tiết

#### Input Preprocessing:

```python
# 1. Convert BGR -> RGB
face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

# 2. Resize về 224x224 (input size của DeepPixBiS)
face_resized = cv2.resize(face_rgb, (224, 224))

# 3. Normalize: (pixel / 255.0 - 0.5) / 0.5
face_normalized = (face_resized.astype(np.float32) / 255.0 - 0.5) / 0.5

# 4. Convert to NCHW
face_input = np.expand_dims(face_normalized.transpose(2, 0, 1), axis=0)
# Shape: [1, 3, 224, 224]
```

#### Network Architecture:

```
Input: [1, 3, 224, 224]

┌─────────────────────────────────────┐
│ Conv Block 1                        │
│ - Conv2d(3→64, kernel=7, stride=2)   │ → [1, 64, 112, 112]
│ - BatchNorm2d(64)                    │
│ - ReLU                               │
│ - MaxPool2d(kernel=3, stride=2)      │ → [1, 64, 56, 56]
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Conv Block 2 (2 layers)              │
│ - Conv2d(64→128, stride=2)           │ → [1, 128, 28, 28]
│ - BatchNorm2d(128)                   │
│ - ReLU                               │
│ - Conv2d(128→128, stride=1)          │
│ - BatchNorm2d(128)                   │
│ - ReLU                               │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Conv Block 3 (2 layers)              │
│ - Conv2d(128→256, stride=2)          │ → [1, 256, 14, 14]
│ - BatchNorm2d(256)                  │
│ - ReLU                               │
│ - Conv2d(256→256, stride=1)          │
│ - BatchNorm2d(256)                   │
│ - ReLU                               │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Conv Block 4 (2 layers)              │
│ - Conv2d(256→512, stride=2)          │ → [1, 512, 7, 7]
│ - BatchNorm2d(512)                  │
│ - ReLU                               │
│ - Conv2d(512→512, stride=1)          │
│ - BatchNorm2d(512)                   │
│ - ReLU                               │
└─────────────────────────────────────┘
         ↓
    ┌─────────┴─────────┐
    ↓                   ↓
┌──────────┐      ┌──────────┐
│ Pixel    │      │ Binary   │
│ Head     │      │ Head     │
└──────────┘      └──────────┘
    ↓                   ↓
[1,1,14,14]      [1, 2]
```

#### Pixel Head (Pixel-wise Map):

```
Input: [1, 512, 7, 7]

Conv2d(512→256, kernel=3, padding=1) → [1, 256, 7, 7]
BatchNorm2d(256)
ReLU

Conv2d(256→128, kernel=3, padding=1) → [1, 128, 7, 7]
BatchNorm2d(128)
ReLU

Conv2d(128→1, kernel=1) → [1, 1, 7, 7]
Sigmoid() → [1, 1, 7, 7]

Upsample to 14x14 → [1, 1, 14, 14]
```

**Output**: Pixel map 14x14, mỗi pixel là xác suất spoof tại vùng đó  
**Giá trị cao** = spoof region, **thấp** = real region

#### Binary Head (Global Classification):

```
Input: [1, 512, 7, 7]

AdaptiveAvgPool2d(1) → [1, 512, 1, 1]
Flatten → [1, 512]

Linear(512→256) → [1, 256]
ReLU
Dropout(0.5)

Linear(256→2) → [1, 2]
```

**Output**: [fake_score, real_score] cho toàn ảnh

#### Output Processing:

```python
# DeepPixBiS output: [batch, 1, H, W] pixel map
if len(outputs[0].shape) == 4:
    pixel_map = outputs[0][0, 0]  # Shape: (H, W) - thường 14x14
else:
    pixel_map = outputs[0][0].reshape(14, 14)

# Tính average score từ pixel map
raw_score = float(np.mean(pixel_map))

# Normalize về [0, 1]
# Giả sử raw_score cao = real (tùy model training)
score = raw_score
score = max(0.0, min(1.0, score))  # Clamp

return score, pixel_map
```

#### Tại Sao Hiệu Quả?

- ✅ **Pixel-wise supervision**: Học từng vùng, phát hiện chi tiết
- ✅ **Dual output**: Pixel map + binary classification
- ✅ **Texture analysis**: Phát hiện kết cấu da thật vs in/màn hình
- ✅ **High resolution**: 224x224 input, chi tiết hơn Global Branch

---

## 🎯 Giai Đoạn 4: Ensemble Fusion & Final Decision

### 4.1. Weighted Sum Fusion

#### Công Thức:

```python
final_score = (
    weight_global * global_score +    # 0.5 * global_score
    weight_local * local_score +       # 0.5 * local_score
    weight_temporal * temporal_score   # 0.0 (disabled)
)
```

**Weights**: Global = 0.5, Local = 0.5 (temporal disabled)  
**Final score**: [0, 1], 1 = real, 0 = spoof

### 4.2. Logic Nghiêm Ngặt ⚠️

#### Yêu Cầu:

```python
# 1. Cả 2 branch phải pass threshold riêng
global_passed = global_score > global_threshold  # 0.5
local_passed = local_score > local_threshold     # 0.5

# 2. Final score phải > final_threshold
final_score_passed = final_score > final_threshold  # 0.12

# 3. CẢ 2 điều kiện phải đúng
is_real = (global_passed AND local_passed) AND final_score_passed
```

**Lý do**: Tránh false positive khi 1 branch bị lỗi  
**Kết quả**: Yêu cầu cả 2 branch đồng thuận

### 4.3. Thresholds

- **Global threshold**: 0.5
- **Local threshold**: 0.5
- **Final threshold**: 0.12 (tối ưu trên test set)

---

## 📊 So Sánh 2 Models

| Tiêu Chí | Global Branch (MiniFASNetV2) | Local Branch (DeepPixBiS) |
|----------|------------------------------|---------------------------|
| **Input size** | 80x80 | 224x224 |
| **Mục tiêu** | Frequency patterns, geometric distortions | Skin texture, pixel artifacts |
| **Output** | Binary classification [fake, real] | Pixel map (14x14) + binary |
| **Strengths** | Phát hiện Moiré, viền thiết bị | Phát hiện texture, reflections |
| **Weaknesses** | Kém chi tiết, dễ miss texture | Chậm hơn, cần input lớn |
| **Weight** | 0.5 | 0.5 |

---

## 🔄 Tổng Kết Pipeline Flow

```
Input Image (BGR)
    ↓
[Quality Gate] (Optional)
    ├─ Blur Detection (Laplacian)
    └─ Pose Estimation (Yaw/Pitch/Roll)
    ↓
[Face Detection] (SCRFD)
    ├─ Detect faces + 5 landmarks
    └─ Confidence filtering
    ↓
[Context Expansion] (Scale 2.0)
    ├─ Expand bbox 2x
    └─ Include context (viền, thiết bị, ngón tay)
    ↓
[Face Alignment] (Similarity Transform)
    ├─ Align using 3 points (2 mắt + mũi)
    └─ Output: 112x112 aligned face
    ↓
[Ensemble Prediction]
    ├─ Global Branch (MiniFASNetV2)
    │   ├─ Input: 80x80
    │   ├─ Output: global_score [0, 1]
    │   └─ Phát hiện: Moiré, viền, distortions
    │
    └─ Local Branch (DeepPixBiS)
        ├─ Input: 224x224
        ├─ Output: local_score [0, 1] + pixel_map [14x14]
        └─ Phát hiện: Texture, reflections, artifacts
    ↓
[Fusion]
    ├─ Weighted Sum: 0.5 * global + 0.5 * local
    └─ Final Score [0, 1]
    ↓
[Final Decision]
    ├─ Check: global_passed AND local_passed
    ├─ Check: final_score > threshold
    └─ Output: REAL or SPOOF
```

---

## 🎯 Kết Luận

Pipeline này kết hợp:
- **Global Branch**: Phát hiện patterns toàn cục (Moiré, viền)
- **Local Branch**: Phân tích chi tiết pixel-wise (texture, artifacts)
- **Context Expansion**: Bao gồm context xung quanh
- **Logic nghiêm ngặt**: Yêu cầu cả 2 branch đồng thuận

**Kết quả**: Độ chính xác cao, giảm false positive, phù hợp với ảnh tĩnh.

---

## 📝 Config Parameters

### Detection
- `context_expansion_scale: 2.0` - Mở rộng bbox 2x để thấy context

### Liveness Ensemble
- `global_branch.threshold: 0.5` - Threshold cho Global Branch
- `local_branch.threshold: 0.5` - Threshold cho Local Branch
- `global_branch.weight: 0.5` - Weight trong fusion
- `local_branch.weight: 0.5` - Weight trong fusion
- `final_threshold: 0.12` - Threshold cuối cùng (tối ưu trên test set)
- `fusion_method: weighted_sum` - Phương pháp fusion

### Quality Gate
- `blur_threshold: 100.0` - Ngưỡng blur detection
- `max_yaw: 20.0` - Góc quay ngang tối đa
- `max_pitch: 20.0` - Góc quay dọc tối đa
- `max_roll: 20.0` - Góc nghiêng tối đa

---

## 🚀 Usage

```bash
# Inference với ensemble
python src/inference_ensemble.py \
    --image data/test/normal/22_1.jpg \
    --config config/config.yaml \
    --skip-quality

# Output:
# - Global Branch score
# - Local Branch score
# - Final Score
# - Prediction: REAL or SPOOF
```

---

**Tài liệu này mô tả chi tiết toàn bộ pipeline Face Liveness Detection, từ input đến output cuối cùng.**

