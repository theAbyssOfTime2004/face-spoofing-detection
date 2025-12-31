"""
Training script cho Local Branch (DeepPixBiS style)
Pixel-wise binary supervision
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import os

from dataset import create_dataloader, analyze_dataset


class DeepPixBiS(nn.Module):
    """
    Simplified DeepPixBiS architecture
    Input: 224x224 RGB image
    Output: Pixel-wise map (14x14) + binary classification
    """
    def __init__(self):
        super(DeepPixBiS, self).__init__()
        
        # Backbone (simplified ResNet-like)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = self._make_layer(64, 128, 2)
        self.conv3 = self._make_layer(128, 256, 2)
        self.conv4 = self._make_layer(256, 512, 2)
        
        # Pixel-wise head (14x14 output)
        self.pixel_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),  # 14x14 output
            nn.Sigmoid()
        )
        
        # Binary classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Pixel-wise map
        pixel_map = self.pixel_head(x)  # [B, 1, 14, 14]
        
        # Binary classification
        pooled = self.global_pool(x)
        pooled = pooled.view(pooled.size(0), -1)
        binary_logits = self.classifier(pooled)  # [B, 2]
        
        return pixel_map, binary_logits


def train_epoch(model, dataloader, criterion_pixel, criterion_binary, optimizer, device):
    """Train một epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        optimizer.zero_grad()
        pixel_map, binary_logits = model(images)
        
        # Pixel-wise loss: tạo target map từ label
        # Nếu label=1 (real), tất cả pixels = 1; nếu label=0 (fake), tất cả pixels = 0
        target_map = labels.view(-1, 1, 1, 1).expand_as(pixel_map).float()
        pixel_loss = criterion_pixel(pixel_map, target_map)
        
        # Binary classification loss
        binary_loss = criterion_binary(binary_logits, labels)
        
        # Combined loss
        loss = pixel_loss + binary_loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(binary_logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pixel': f'{pixel_loss.item():.4f}',
            'binary': f'{binary_loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion_pixel, criterion_binary, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            pixel_map, binary_logits = model(images)
            
            # Losses
            target_map = labels.view(-1, 1, 1, 1).expand_as(pixel_map).float()
            pixel_loss = criterion_pixel(pixel_map, target_map)
            binary_loss = criterion_binary(binary_logits, labels)
            loss = pixel_loss + binary_loss
            
            total_loss += loss.item()
            _, predicted = torch.max(binary_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Local Branch (DeepPixBiS)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=70, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset first')
    parser.add_argument('--context-expansion-scale', type=float, default=2.7,
                       help='Context expansion scale for raw crop (default: 2.7)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay (L2 regularization) for optimizer (default: 1e-3, increased from 5e-4)')
    parser.add_argument('--label-smoothing', type=float, default=0.15,
                       help='Label smoothing factor for binary classification (default: 0.15, increased from 0.1)')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience (stop if no improvement for N epochs, default: 10, 0 = disabled)')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone, only train heads (for transfer learning, reduces overfitting)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Analyze dataset
    if args.analyze:
        analyze_dataset(args.data_dir)
    
    # Create dataloaders với Raw Crop từ Full Images
    print("\nLoading datasets with Raw Crop from Full Images (no alignment)...")
    print(f"  Context expansion scale: {args.context_expansion_scale}")
    print("  Note: Full image → Detect → Crop trực tiếp (tránh distribution shift)")
    print("        Raw crop preserves texture patterns for better spoof detection")
    train_loader = create_dataloader(
        args.data_dir, 'train',
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        augment=True,
        context_expansion_scale=args.context_expansion_scale,
        use_raw_crop=True,
        use_full_image_detection=True  # Detect từ full image
    )
    val_loader = create_dataloader(
        args.data_dir, 'dev',
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        augment=False,
        shuffle=False,
        context_expansion_scale=args.context_expansion_scale,
        use_raw_crop=True,
        use_full_image_detection=True  # Detect từ full image
    )
    
    # Create model
    model = DeepPixBiS()
    model = model.to(device)
    
    # Freeze backbone nếu được yêu cầu (Transfer Learning)
    if args.freeze_backbone:
        print("\n" + "="*50)
        print("Freezing Backbone (Transfer Learning Mode):")
        print("  - Freezing: conv1, conv2, conv3, conv4")
        print("  - Training: pixel_head, classifier")
        print("="*50)
        
        # Freeze backbone
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.conv2.parameters():
            param.requires_grad = False
        for param in model.conv3.parameters():
            param.requires_grad = False
        for param in model.conv4.parameters():
            param.requires_grad = False
        
        # Chỉ train heads
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        print("="*50 + "\n")
    
    # Losses với chống overfitting
    criterion_pixel = nn.BCELoss()  # Binary Cross Entropy cho pixel map (không hỗ trợ label_smoothing)
    # Label Smoothing: Giảm overconfidence, cải thiện generalization
    criterion_binary = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # Cho binary classification
    
    # Optimizer với Weight Decay (L2 Regularization): Phạt weights quá lớn, giảm overfitting
    # Chỉ optimize các parameters có requires_grad=True
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    # ReduceLROnPlateau: Giảm LR khi val_acc không cải thiện (quan trọng để tránh "sập" ở cuối)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 'max' vì theo dõi val_acc (muốn tăng)
        factor=0.5,  # Giảm LR còn 50% khi không cải thiện
        patience=5,  # Đợi 5 epochs không cải thiện thì giảm LR
        min_lr=1e-6  # LR tối thiểu
    )
    
    # Print anti-overfitting settings
    print("\n" + "="*50)
    print("Anti-Overfitting Settings:")
    print(f"  Weight Decay (L2): {args.weight_decay} (increased from 5e-4)")
    print(f"  Label Smoothing: {args.label_smoothing} (increased from 0.1)")
    print(f"  Dropout: 0.5 (in model architecture)")
    print(f"  CoarseDropout: Enabled in augmentation (if albumentations available)")
    print(f"  LR Scheduler: ReduceLROnPlateau (giảm LR khi val_acc không cải thiện)")
    if args.early_stopping > 0:
        print(f"  Early Stopping: Enabled (patience={args.early_stopping})")
    else:
        print(f"  Early Stopping: Disabled")
    if args.freeze_backbone:
        print(f"  Freeze Backbone: Enabled (Transfer Learning)")
    print("="*50)
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*50)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion_pixel, criterion_binary, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion_pixel, criterion_binary, device
        )
        
        # Update learning rate (ReduceLROnPlateau cần val_acc để quyết định)
        scheduler.step(val_acc)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0  # Reset patience counter
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_acc': val_acc,
                'train_acc': train_acc
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_local.pth'))
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Save latest
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'val_acc': val_acc,
            'train_acc': train_acc
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_local.pth'))
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\nEarly stopping triggered! No improvement for {args.early_stopping} epochs.")
            print(f"Best validation accuracy: {best_acc:.2f}%")
            break
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()

