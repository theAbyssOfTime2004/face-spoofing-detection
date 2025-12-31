# Face Liveness Detection Pipeline

A state-of-the-art face liveness detection system using multi-stage ensemble architecture with quality-aware processing. The pipeline combines Global (MiniFASNetV2) and Local (DeepPixBiS) branches to detect spoof attacks in biometric authentication systems.

## Features

- **Quality Gate**: Filters blurry images and invalid head poses
- **SCRFD Detection**: Accurate face detection and alignment with 5 keypoints
- **Multi-stage Liveness Ensemble**: Combines complementary detection strategies
  - **Global Branch**: MiniFASNetV2 for global facial feature analysis
  - **Local Branch**: DeepPixBiS for pixel-wise texture analysis
  - **Temporal Branch**: Blink detection for video streams (optional)
- **Face Recognition**: ArcFace-based 1-to-1 matching (optional)

## Requirements

- Python >= 3.8
- CUDA (optional, for GPU acceleration)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/theAbyssOfTime2004/face-spoofing-detection.git
cd face-spoofing-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models

The pipeline requires ONNX models in the `models/` directory:

- `models/global_branch.onnx` - Global Branch (MiniFASNetV2)
- `models/local_branch.onnx` - Local Branch (DeepPixBiS)

**Note**: SCRFD and ArcFace models are automatically downloaded from InsightFace model zoo on first run.

## Project Structure

```
fld-cake-assignment/
├── config/
│   └── config.yaml          # Pipeline configuration
├── src/
│   ├── pipeline/
│   │   ├── quality_gate.py      # Quality validation
│   │   ├── detection.py         # SCRFD face detection
│   │   ├── liveness_ensemble.py # Ensemble liveness detection
│   │   ├── recognition.py       # ArcFace recognition (optional)
│   │   └── pipeline.py          # Main pipeline
│   ├── train_global.py          # Global branch training
│   ├── train_local.py           # Local branch training
│   ├── evaluate_ensemble.py     # Model evaluation
│   └── convert_to_onnx.py       # ONNX conversion
├── models/                      # Model weights (ONNX)
├── checkpoints/                 # Training checkpoints
├── data/                        # Dataset directory
├── results/                     # Evaluation results
└── requirements.txt
```

## Configuration

Edit `config/config.yaml` to customize pipeline behavior:

```yaml
pipeline:
  quality_gate:
    max_yaw: 20.0
    max_pitch: 20.0
    max_roll: 20.0
    blur_threshold: 100.0
    
  detection:
    model_name: "buffalo_l"
    context_expansion_scale: 2.7
    
  liveness:
    global_branch:
      model_path: "models/global_branch.onnx"
      threshold: 0.5
      weight: 0.4
    local_branch:
      model_path: "models/local_branch.onnx"
      threshold: 0.5
      weight: 0.6
    fusion_method: "weighted_sum"
    final_threshold: 0.410
```

## Usage

### Process Video File

```bash
python src/main.py --input video.mp4
```

### Process Camera Stream

```bash
python src/main.py --camera
```

### Command Line Options

```bash
# Show detailed scores
python src/main.py --input video.mp4 --show-details

# Limit number of frames
python src/main.py --input video.mp4 --max-frames 100

# Save output video
python src/main.py --input video.mp4 --output output.mp4

# Use custom config
python src/main.py --input video.mp4 --config config/custom_config.yaml
```

## Training

### Dataset Format

Organize your dataset as follows:

```
data/
├── train/
│   ├── normal/  # Real faces
│   └── spoof/   # Fake faces
├── dev/
│   ├── normal/
│   └── spoof/
└── test/
    ├── normal/
    └── spoof/
```

### Preprocessing

```bash
# Preprocess and cache bounding boxes
python src/preprocess_bbox_cache.py --data-dir data

# Check for data leakage (identity overlap)
python src/check_data_leakage.py --data-dir data

# Resplit data by identity (if leakage detected)
python src/resplit_data_by_identity.py --data-dir data --yes
```

### Training Commands

```bash
# Train Global Branch
python src/train_global.py --data-dir data --epochs 70 \
    --weight-decay 1e-3 --label-smoothing 0.15 --early-stopping 10

# Train Local Branch
python src/train_local.py --data-dir data --epochs 70 \
    --weight-decay 1e-3 --label-smoothing 0.15 --early-stopping 10

# Convert to ONNX
python src/convert_to_onnx.py --model-type global \
    --checkpoint checkpoints/best_global.pth \
    --output models/global_branch.onnx

python src/convert_to_onnx.py --model-type local \
    --checkpoint checkpoints/best_local.pth \
    --output models/local_branch.onnx
```

### Evaluation

```bash
# Evaluate ensemble model
python src/evaluate_ensemble.py --data-dir data --split test \
    --plot --output-dir results
```

## Architecture

```
Input Video Stream
    ↓
Quality Gate (Blur/Pose Validation)
    ↓
SCRFD Detection & Alignment
    ↓
Raw Crop (2.7x context expansion)
    ↓
Liveness Ensemble
    ├── Global Branch (80×80, MiniFASNetV2)
    ├── Local Branch (224×224, DeepPixBiS)
    └── Temporal Branch (Blink Detection, optional)
    ↓
Weighted Fusion (0.4 × Global + 0.6 × Local)
    ↓
Final Decision (Threshold: 0.410)
```

## Model Performance

- **Accuracy**: 89.28%
- **Precision**: 87.28%
- **Recall**: 93.11%
- **F1-Score**: 90.10%

Detailed evaluation results are available in `results/evaluation_ensemble.txt`.

## Key Design Decisions

1. **Raw Crop (No Alignment)**: Preserves high-frequency patterns essential for Moiré detection
2. **Context Expansion (2.7x)**: Captures surrounding context (paper edges, device screens, fingers)
3. **Identity-Based Splitting**: Ensures zero identity overlap between train/dev/test sets
4. **Anti-Overfitting Measures**: Weight decay (1e-3), label smoothing (0.15), CoarseDropout augmentation
5. **Ensemble Fusion**: Weighted sum combining complementary global and local features

## Troubleshooting

### Missing Models

If models are not found, ensure ONNX files are in `models/` directory. The pipeline will use dummy predictions for testing if models are unavailable.

### ONNX Runtime

```bash
# CPU version
pip install onnxruntime

# GPU version (requires CUDA)
pip install onnxruntime-gpu
```

### InsightFace

```bash
pip install insightface
```

Models are automatically downloaded on first run.

## Documentation

- **Model Performance Report**: See `Model_Performance_Report.pdf` for detailed evaluation
- **Submission Notebook**: See `Liveness_Detection_Submission.ipynb` for rationale and examples

## License

MIT License

## References