# train_paper_model_fixed.py
#!/usr/bin/env python3
"""آموزش مدل با رفع خطاها"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_loader import DataLoaderModule
from src.vit_model import SkinCancerClassifier
from src.trainer import Trainer

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def check_dataset():
    """بررسی وجود دیتاست"""
    config = Config()
    
    print("Checking dataset structure...")
    
    # بررسی فایل metadata
    if not os.path.exists(config.metadata_path):
        print(f"❌ Metadata file not found: {config.metadata_path}")
        print("\nPlease download HAM10000 dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("\nAnd place:")
        print(f"1. HAM10000_metadata.csv in {config.data_dir}")
        print(f"2. Images in {config.images_part1} and {config.images_part2}")
        return False
    
    # بررسی پوشه‌های تصاویر
    if not os.path.exists(config.images_part1):
        print(f"⚠ Warning: {config.images_part1} not found")
        # ایجاد پوشه
        os.makedirs(config.images_part1, exist_ok=True)
    
    if not os.path.exists(config.images_part2):
        print(f"⚠ Warning: {config.images_part2} not found")
        # ایجاد پوشه
        os.makedirs(config.images_part2, exist_ok=True)
    
    print("✓ Dataset structure OK")
    return True

def prepare_data():
    """آماده‌سازی داده‌ها"""
    config = Config()
    data_loader = DataLoaderModule(config)
    
    print("\nLoading metadata...")
    df = data_loader.load_metadata()
    
    if df is None or len(df) == 0:
        print("❌ Failed to load metadata")
        return None, None, None
    
    print(f"✓ Loaded {len(df)} samples")
    
    # بررسی توزیع کلاس‌ها
    print("\nClass distribution:")
    class_counts = df['dx'].value_counts()
    for class_name, count in class_counts.items():
        percentage = count / len(df) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # ایجاد balanced dataset (برای تست سریع، تعداد کمتر)
    print("\nCreating balanced dataset (for quick test)...")
    balanced_data = []
    
    for class_name, group in df.groupby('dx'):
        # برای تست سریع، 200 نمونه در هر کلاس
        n_samples = min(200, len(group))
        sampled = group.sample(n=n_samples, random_state=42)
        balanced_data.append(sampled)
    
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    print(f"Balanced dataset: {len(balanced_df)} samples")
    
    # تقسیم داده‌ها
    print("\nSplitting dataset...")
    train_val_df, test_df = train_test_split(
        balanced_df,
        test_size=config.test_size,
        stratify=balanced_df['dx'],
        random_state=config.random_state
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.val_size/(1-config.test_size),
        stratify=train_val_df['dx'],
        random_state=config.random_state
    )
    
    print(f"\nDataset Split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    return train_df, val_df, test_df

def main():
    """تابع اصلی"""
    
    print("=" * 80)
    print("PAPER-ACCURATE SKIN CANCER CLASSIFICATION")
    print("Target: 92.14% accuracy (from paper)")
    print("=" * 80)
    
    # 1. بررسی دیتاست
    if not check_dataset():
        return
    
    # 2. آماده‌سازی داده‌ها
    train_df, val_df, test_df = prepare_data()
    
    if train_df is None:
        return
    
    # 3. ایجاد مدل
    print("\n" + "="*80)
    print("CREATING VISION TRANSFORMER MODEL")
    print("="*80)
    
    config = Config()
    config.freeze_backbone = True  # اضافه کردن این ویژگی
    config.learning_rate = 0.0001
    config.num_epochs = 15  # برای تست سریع
    
    try:
        classifier = SkinCancerClassifier(config, use_paper_vit=True)
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        print("Trying alternative approach...")
        
        # راه‌حل جایگزین: استفاده از ResNet
        from torchvision import models
        import torch.nn as nn
        
        class SimpleResNet(nn.Module):
            def __init__(self, num_classes=7):
                super().__init__()
                self.model = models.resnet18(pretrained=True)
                self.model.fc = nn.Linear(512, num_classes)
            
            def forward(self, x):
                return self.model(x)
        
        # ایجاد مدل ساده
        classifier = type('SkinCancerClassifier', (), {})()
        classifier.model = SimpleResNet(len(config.class_mapping))
        classifier.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier.model.to(classifier.device)
        classifier.criterion = nn.CrossEntropyLoss()
        
        # متدهای شبیه‌سازی
        classifier.get_optimizer = lambda lr=0.001: torch.optim.Adam(
            classifier.model.parameters(), lr=lr
        )
        classifier.get_scheduler = lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=5)
        classifier.save_model = lambda path: torch.save(classifier.model.state_dict(), path)
        
        print("✓ Created simple ResNet18 model")
    
    # 4. ایجاد Trainer
    trainer = Trainer(config)
    trainer.classifier = classifier
    
    # 5. ایجاد DataLoaderها
    print("\n" + "="*80)
    print("CREATING DATA LOADERS")
    print("="*80)
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
    from src.data_loader import HAM10000Dataset
    
    # Transformهای ساده برای تست سریع
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # ایجاد datasets
    train_dataset = HAM10000Dataset(
        train_df,
        config.images_part1,
        config.images_part2,
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = HAM10000Dataset(
        val_df,
        config.images_part1,
        config.images_part2,
        transform=val_transform,
        mode='val'
    )
    
    # DataLoaderها با batch size کوچک
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # کوچک برای حافظه کمتر
        shuffle=True,
        num_workers=0  # برای جلوگیری از crash
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    # 6. آموزش
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    try:
        history = trainer.train(train_loader, val_loader, num_epochs=config.num_epochs)
        
        # 7. تست
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        
        test_dataset = HAM10000Dataset(
            test_df,
            config.images_part1,
            config.images_part2,
            transform=val_transform,
            mode='test'
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0
        )
        
        # ارزیابی
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
        
        classifier.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(classifier.device)
                labels = labels.to(classifier.device)
                
                outputs = classifier.model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n📊 RESULTS:")
        print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Paper Target: 92.14%")
        print(f"  Difference: {abs(accuracy*100 - 92.14):.2f}%")
        
        # ذخیره نتایج
        results = {
            'test_accuracy': float(accuracy),
            'paper_accuracy': 92.14,
            'difference': float(abs(accuracy*100 - 92.14)),
            'num_samples': len(test_df)
        }
        
        import json
        os.makedirs('results', exist_ok=True)
        with open('results/test_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✓ Results saved to results/test_results.json")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("PROGRAM COMPLETED")
    print("="*80)

if __name__ == "__main__":
    # نصب کتابخانه‌های ضروری
    import subprocess
    import importlib
    
    required = ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn']
    
    for package in required:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    main()