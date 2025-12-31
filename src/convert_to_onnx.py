"""
Convert trained PyTorch models sang ONNX format
Để sử dụng trong inference pipeline
"""
import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path

from train_global import MiniFASNetV2
from train_local import DeepPixBiS


def convert_global_to_onnx(checkpoint_path, output_path, image_size=80):
    """Convert Global Branch model sang ONNX"""
    print(f"Converting Global Branch model...")
    
    # Load model
    model = MiniFASNetV2(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Converted to {output_path}")


def convert_local_to_onnx(checkpoint_path, output_path, image_size=224):
    """Convert Local Branch model sang ONNX"""
    print(f"Converting Local Branch model...")
    
    # Load model
    model = DeepPixBiS()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Export to ONNX: Export cả pixel_map và binary_logits
    # Option 1: Export binary_logits (classification head) - khuyến nghị
    class DeepPixBiS_ONNX(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, x):
            pixel_map, binary_logits = self.base_model(x)
            # Return binary_logits để dùng cho classification
            # Pixel map vẫn có thể dùng cho visualization nếu cần
            return binary_logits
    
    onnx_model = DeepPixBiS_ONNX(model)
    
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['binary_logits'],  # Output là binary_logits
        dynamic_axes={
            'input': {0: 'batch_size'},
            'binary_logits': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Converted to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX')
    parser.add_argument('--model-type', type=str, choices=['global', 'local'], 
                       required=True, help='Model type to convert')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output ONNX file path')
    parser.add_argument('--image-size', type=int, default=None,
                       help='Input image size (default: 80 for global, 224 for local)')
    
    args = parser.parse_args()
    
    # Set default image size
    if args.image_size is None:
        args.image_size = 80 if args.model_type == 'global' else 224
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Convert
    if args.model_type == 'global':
        convert_global_to_onnx(args.checkpoint, args.output, args.image_size)
    else:
        convert_local_to_onnx(args.checkpoint, args.output, args.image_size)
    
    print(f"\n✓ Conversion completed!")
    print(f"  Model: {args.model_type}")
    print(f"  Input size: {args.image_size}x{args.image_size}")
    print(f"  Output: {args.output}")
    print(f"\nYou can now use this model in the pipeline by updating config.yaml:")
    print(f"  liveness:")
    print(f"    {args.model_type}_branch:")
    print(f"      model_path: \"{args.output}\"")


if __name__ == '__main__':
    main()

