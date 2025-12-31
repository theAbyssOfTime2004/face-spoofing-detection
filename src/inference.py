"""
Inference script cho một ảnh đơn lẻ
Predict xem ảnh là Real hay Spoof
"""
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from train_global import MiniFASNetV2
from train_local import DeepPixBiS


def preprocess_image(image_path, image_size=(80, 80), normalize=True):
    """
    Preprocess ảnh giống như trong training
    Args:
        image_path: Đường dẫn đến ảnh
        image_size: Kích thước output (width, height)
        normalize: Có normalize không
    Returns:
        Preprocessed image tensor [1, 3, H, W]
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, image_size)
    
    # Convert to float và normalize
    image = image.astype(np.float32) / 255.0
    
    if normalize:
        # Normalize về [-1, 1] (giống training)
        image = (image - 0.5) / 0.5
    
    # Convert to CHW format
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def predict_global(image_path, checkpoint_path, device='cuda', image_size=80):
    """
    Predict sử dụng Global Branch model
    """
    # Load model
    model = MiniFASNetV2(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Preprocess image
    image = preprocess_image(image_path, image_size=(image_size, image_size))
    image_tensor = torch.from_numpy(image).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Get results
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()
    prediction = "REAL" if predicted.item() == 1 else "SPOOF"
    
    return {
        'prediction': prediction,
        'real_prob': real_prob,
        'fake_prob': fake_prob,
        'confidence': max(real_prob, fake_prob)
    }


def predict_local(image_path, checkpoint_path, device='cuda', image_size=224):
    """
    Predict sử dụng Local Branch model
    """
    # Load model
    model = DeepPixBiS()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Preprocess image
    image = preprocess_image(image_path, image_size=(image_size, image_size))
    image_tensor = torch.from_numpy(image).to(device)
    
    # Predict
    with torch.no_grad():
        pixel_map, binary_logits = model(image_tensor)
        probs = torch.softmax(binary_logits, dim=1)
        _, predicted = torch.max(binary_logits, 1)
    
    # Get results
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()
    prediction = "REAL" if predicted.item() == 1 else "SPOOF"
    
    # Average pixel map score
    pixel_score = pixel_map[0, 0].mean().item()
    
    return {
        'prediction': prediction,
        'real_prob': real_prob,
        'fake_prob': fake_prob,
        'confidence': max(real_prob, fake_prob),
        'pixel_score': pixel_score
    }


def visualize_result(image_path, result, save_path=None):
    """
    Visualize kết quả trên ảnh
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return
    
    # Draw result
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Color: green for REAL, red for SPOOF
    color = (0, 255, 0) if prediction == "REAL" else (0, 0, 255)
    
    # Add text
    text = f"{prediction} ({confidence*100:.1f}%)"
    cv2.putText(image, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Add probability info
    info = f"Real: {result['real_prob']*100:.1f}% | Fake: {result['fake_prob']*100:.1f}%"
    cv2.putText(image, info, (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(str(save_path), image)
        print(f"Saved result to {save_path}")
    else:
        # Try to display, fallback to auto-save if GUI not available
        try:
            cv2.imshow('Prediction Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            # GUI not available, auto-save to file
            auto_save_path = str(image_path.parent / f"{image_path.stem}_result{image_path.suffix}")
            cv2.imwrite(auto_save_path, image)
            print(f"\nNote: GUI not available. Result saved to: {auto_save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Inference một ảnh đơn lẻ - Predict Real/Spoof',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict với Global model
  python src/inference.py --image test.jpg --model-type global --checkpoint checkpoints/best_global.pth
  
  # Predict với Local model
  python src/inference.py --image test.jpg --model-type local --checkpoint checkpoints/best_local.pth
  
  # Predict và save kết quả
  python src/inference.py --image test.jpg --model-type global --checkpoint checkpoints/best_global.pth --output result.jpg
        """
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Đường dẫn đến ảnh cần predict')
    parser.add_argument('--model-type', type=str, choices=['global', 'local'],
                       required=True, help='Loại model (global hoặc local)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Đường dẫn đến checkpoint (.pth file)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device để inference')
    parser.add_argument('--output', type=str, default=None,
                       help='Đường dẫn để lưu ảnh kết quả (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Không hiển thị ảnh (chỉ print kết quả)')
    
    args = parser.parse_args()
    
    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Predict
    print(f"\nPredicting image: {args.image}")
    print(f"Model: {args.model_type}")
    print("-" * 50)
    
    try:
        if args.model_type == 'global':
            result = predict_global(image_path, checkpoint_path, device)
        else:
            result = predict_local(image_path, checkpoint_path, device)
        
        # Print results
        print(f"\n{'='*50}")
        print("PREDICTION RESULT")
        print(f"{'='*50}")
        print(f"Image: {args.image}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"\nProbabilities:")
        print(f"  REAL: {result['real_prob']*100:.2f}%")
        print(f"  SPOOF: {result['fake_prob']*100:.2f}%")
        
        if 'pixel_score' in result:
            print(f"  Pixel Score: {result['pixel_score']:.4f}")
        
        print(f"{'='*50}\n")
        
        # Visualize
        if not args.no_display or args.output:
            visualize_result(image_path, result, args.output)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

