#!/usr/bin/env python3
"""
Food Vision Pro - Food Classifier Training Script

This script fine-tunes an EfficientNet-B0 model on the Food-101 dataset
for food classification. It includes data augmentation, early stopping,
and checkpointing.

Usage:
    python train_classifier.py --data_dir /path/to/food101 --epochs 50 --batch_size 32
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Food101Dataset(Dataset):
    """Custom dataset for Food-101 with data augmentation"""
    
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        
        # Get class names from directory structure
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
        
        logger.info(f"Dataset loaded: {len(self.images)} images, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(input_size=224):
    """Get data transforms for training and validation"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model(num_classes, model_name='efficientnet_b0'):
    """Create and configure the model"""
    
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in model.features[:5].parameters():
            param.requires_grad = False
        
        # Modify classifier for our number of classes
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, num_classes)
        )
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in model.layer1.parameters():
            param.requires_grad = False
        
        # Modify classifier
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_predictions, all_targets


def save_checkpoint(model, optimizer, epoch, best_acc, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Food Vision Pro Classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to Food-101 dataset directory')
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                       help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'resnet50'],
                       help='Model architecture to use')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transform, val_transform = get_transforms(args.input_size)
    
    # Load datasets
    train_dataset = Food101Dataset(
        os.path.join(args.data_dir, 'train'),
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = Food101Dataset(
        os.path.join(args.data_dir, 'val'),
        transform=val_transform,
        is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Create model
    model = create_model(len(train_dataset.classes), args.model)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, predictions, targets = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(args.output_dir, f'{args.model}_best.pth')
            save_checkpoint(model, optimizer, epoch, best_acc, best_model_path)
            
            # Save class index
            class_index = {
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "num_classes": len(train_dataset.classes)
            }
            
            with open(os.path.join(args.output_dir, 'class_index.json'), 'w') as f:
                json.dump(class_index, f, indent=2)
            
            logger.info(f"New best accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping after {args.patience} epochs without improvement")
            break
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'{args.model}_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path)
    
    # Final evaluation
    logger.info("\nFinal evaluation...")
    final_val_loss, final_val_acc, final_predictions, final_targets = validate(
        model, val_loader, criterion, device
    )
    
    # Classification report
    report = classification_report(
        final_targets, final_predictions,
        target_names=train_dataset.classes,
        output_dict=True
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_acc': best_acc,
        'final_val_acc': final_val_acc,
        'classification_report': report
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    
    logger.info(f"Training completed! Best accuracy: {best_acc:.2f}%")
    logger.info(f"Final validation accuracy: {final_val_acc:.2f}%")
    logger.info(f"Model and artifacts saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
