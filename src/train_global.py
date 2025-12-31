"""
Training script cho Global Branch (MiniFASNetV2 style)
Model architecture đơn giản hóa cho binary classification
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


class MiniFASNetV2(nn.Module):
    """
    Simplified MiniFASNetV2 architecture cho binary classification
    Input: 80x80 RGB image
    Output: 2 classes (fake, real)
    """
    def __init__(self, num_classes=2):
        super(MiniFASNetV2, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Global Branch (MiniFASNetV2)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=70, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=80, help='Input image size')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset first')
    parser.add_argument('--context-expansion-scale', type=float, default=2.7, 
                       help='Context expansion scale for raw crop (default: 2.7)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay (L2 regularization) for optimizer (default: 1e-3, increased from 5e-4)')
    parser.add_argument('--label-smoothing', type=float, default=0.15,
                       help='Label smoothing factor (default: 0.15, increased from 0.1)')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience (stop if no improvement for N epochs, default: 10, 0 = disabled)')
    
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
    print("        Raw crop preserves high-frequency patterns (Moiré) for better spoof detection")
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
    model = MiniFASNetV2(num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer với chống overfitting
    # Label Smoothing: Giảm overconfidence, cải thiện generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # Weight Decay (L2 Regularization): Phạt weights quá lớn, giảm overfitting
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
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
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_global.pth'))
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
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_global.pth'))
        
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

