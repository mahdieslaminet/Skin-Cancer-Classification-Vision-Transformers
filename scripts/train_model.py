#!/usr/bin/env python3
"""
اسکریپت آموزش سریع با مدل سبک‌تر
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_loader import DataLoaderModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LightweightViT(nn.Module):
    """ViT سبک‌تر برای اجرا روی CPU"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = len(config.class_mapping)
        
        # استفاده از مدل ResNet18 سبک‌تر
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        
        # جایگزینی لایه‌های fully connected
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # فریز کردن لایه‌های اولیه
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

def train_fast():
    """آموزش سریع"""
    print("=" * 60)
    print("FAST TRAINING ON CPU")
    print("=" * 60)
    
    # پیکربندی سبک‌تر
    config = Config()
    config.image_size = (128, 128)  # کاهش سایز تصویر
    config.batch_size = 8  # کاهش batch size
    config.num_epochs = 10  # کاهش تعداد epochs
    
    # بارگذاری داده‌ها
    print("\n1. Loading data...")
    data_loader = DataLoaderModule(config)
    metadata_df = data_loader.load_metadata()
    
    # تقسیم داده‌ها (کوچکتر)
    print("\n2. Creating smaller dataset...")
    # فقط 20% از داده‌ها را استفاده کن
    small_df = metadata_df.sample(frac=0.2, random_state=42)
    train_df, val_test_df = train_test_split(
        small_df, 
        test_size=0.3,
        stratify=small_df['dx'],
        random_state=42
    )
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=0.5,
        stratify=val_test_df['dx'],
        random_state=42
    )
    
    print(f"Small dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # ایجاد دیتاست ساده
    from src.data_loader import HAM10000Dataset
    
    simple_transform = A.Compose([
        A.Resize(config.image_size[0], config.image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    train_dataset = HAM10000Dataset(
        train_df,
        config.images_part1,
        config.images_part2,
        transform=simple_transform,
        mode='train'
    )
    
    val_dataset = HAM10000Dataset(
        val_df,
        config.images_part1,
        config.images_part2,
        transform=simple_transform,
        mode='val'
    )
    
    # DataLoaderها
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # غیرفعال برای CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # مدل سبک
    print("\n3. Creating lightweight model...")
    device = torch.device('cpu')
    model = LightweightViT(config).to(device)
    
    # Loss و Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # آموزش
    print("\n4. Starting training...")
    best_val_acc = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # آموزش
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # اعتبارسنجی
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        
        # ذخیره بهترین مدل
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/fast_model_best.pth')
            print(f"✓ Best model saved! (Acc: {val_acc:.2f}%)")
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    
    # تست
    print("\n5. Testing...")
    test_dataset = HAM10000Dataset(
        test_df,
        config.images_part1,
        config.images_part2,
        transform=simple_transform,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    model.load_state_dict(torch.load('models/fast_model_best.pth', map_location=device))
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("FAST TRAINING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
    
    # تنظیمات برای جلوگیری از crash
    torch.set_num_threads(1)  # کاهش threads
    os.environ['OMP_NUM_THREADS'] = '1'
    
    train_fast()