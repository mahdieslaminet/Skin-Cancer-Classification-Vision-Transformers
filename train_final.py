#!/usr/bin/env python3
"""اسکریپت نهایی برای آموزش مدل"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.config import Config
from src.data_loader import DataLoaderModule
from src.trainer import Trainer

def setup_directories():
    """ایجاد دایرکتوری‌های لازم"""
    directories = ['models', 'results', 'results/plots', 'results/logs', 'data/processed']
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Directories created")

def load_and_prepare_data():
    """بارگذاری و آماده‌سازی داده‌ها"""
    print("\n" + "="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    config = Config()
    data_loader = DataLoaderModule(config)
    
    # بارگذاری metadata
    df = data_loader.load_metadata()
    if df is None:
        print("❌ Failed to load metadata")
        return None, None, None, None
    
    # بررسی وجود ستون ضروری
    if 'dx' not in df.columns:
        print("❌ Column 'dx' not found in metadata")
        return None, None, None, None
    
    # نمایش اطلاعات اولیه
    print(f"\nDataset Info:")
    print(f"Total samples: {len(df)}")
    print(f"Classes: {df['dx'].nunique()}")
    
    # تقسیم داده‌ها
    print("\nSplitting data...")
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        stratify=df['dx'],
        random_state=config.random_state
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.val_size/(1-config.test_size),
        stratify=train_val_df['dx'],
        random_state=config.random_state
    )
    
    print(f"\nDataset Split:")
    print(f"Training:   {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test:       {len(test_df)} samples")
    
    # ذخیره split‌ها
    train_df.to_csv('data/processed/train_metadata.csv', index=False)
    val_df.to_csv('data/processed/val_metadata.csv', index=False)
    test_df.to_csv('data/processed/test_metadata.csv', index=False)
    
    print("\n✓ Data preparation completed")
    return train_df, val_df, test_df, config

def train_model(train_df, val_df, test_df, config):
    """آموزش مدل"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    data_loader = DataLoaderModule(config)
    
    # ایجاد DataLoaderها
    print("\nCreating data loaders...")
    train_loader, val_loader = data_loader.create_data_loaders(train_df, val_df)
    
    # ایجاد و آموزش مدل
    trainer = Trainer(config)
    
    print(f"\nModel Configuration:")
    print(f"Type: {config.model_type}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    
    # آموزش
    history = trainer.train(train_loader, val_loader)
    
    # ایجاد DataLoader برای تست
    print("\nCreating test data loader...")
    _, _, test_loader = data_loader.create_data_loaders(train_df, val_df, test_df)
    
    # ارزیابی روی تست ست
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_loader)
    
    return history, results, trainer

def analyze_results(results, config):
    """تحلیل و نمایش نتایج"""
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    accuracy = results['accuracy']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1_score']
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\n🎯 Paper Target: 92.14%")
    print(f"📈 Difference: {abs(accuracy*100 - 92.14):.2f}%")
    
    # نمایش classification report
    print(f"\n📋 Classification Report:")
    print(results['classification_report'])
    
    # رسم confusion matrix
    cm = results['confusion_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(config.class_mapping.keys()),
                yticklabels=list(config.class_mapping.keys()))
    plt.title(f'Confusion Matrix - Accuracy: {accuracy*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300)
    plt.show()
    
    return accuracy

def save_final_report(history, results, accuracy, config):
    """ذخیره گزارش نهایی"""
    print("\n" + "="*60)
    print("SAVING FINAL REPORT")
    print("="*60)
    
    report = {
        'model_info': {
            'type': config.model_type,
            'num_classes': len(config.class_mapping),
            'image_size': config.image_size,
            'training_epochs': len(history['train_loss'])
        },
        'performance': {
            'accuracy': float(accuracy),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score'])
        },
        'comparison': {
            'paper_accuracy': 92.14,
            'difference': float(abs(accuracy*100 - 92.14))
        },
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_acc': [float(x) for x in history['val_acc']]
        }
    }
    
    # ذخیره به JSON
    with open('results/final_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # ذخیره به CSV برای خوانایی بهتر
    df_history = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc']
    })
    df_history.to_csv('results/training_history.csv', index=False)
    
    print(f"\n✓ Final report saved to results/final_report.json")
    print(f"✓ Training history saved to results/training_history.csv")
    print(f"✓ Confusion matrix saved to results/confusion_matrix.png")
    print(f"✓ Best model saved to models/best_model.pth")

def main():
    """تابع اصلی"""
    print("="*80)
    print("SKIN CANCER CLASSIFICATION - PAPER IMPLEMENTATION")
    print("Target Accuracy: 92.14%")
    print("="*80)
    
    # مرحله 1: تنظیم دایرکتوری‌ها
    setup_directories()
    
    # مرحله 2: بارگذاری و آماده‌سازی داده‌ها
    train_df, val_df, test_df, config = load_and_prepare_data()
    if train_df is None:
        print("\n❌ Failed to prepare data. Exiting...")
        return
    
    # مرحله 3: آموزش مدل
    history, results, trainer = train_model(train_df, val_df, test_df, config)
    
    # مرحله 4: تحلیل نتایج
    accuracy = analyze_results(results, config)
    
    # مرحله 5: ذخیره گزارش نهایی
    save_final_report(history, results, accuracy, config)
    
    # مرحله 6: نمایش خلاصه
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📈 Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"🎯 Paper Target Accuracy: 92.14%")
    print(f"✅ Difference: {abs(accuracy*100 - 92.14):.2f}%")
    
    if accuracy * 100 >= 85:
        print("\n🏆 EXCELLENT! Model achieved good accuracy!")
    elif accuracy * 100 >= 75:
        print("\n👍 GOOD! Model performed well!")
    else:
        print("\n⚠ Model needs improvement. Consider:")
        print("  - Increasing epochs")
        print("  - Using more augmentation")
        print("  - Trying different model architecture")
    
    print("\n📁 Results saved in 'results/' folder")
    print("💾 Model saved in 'models/best_model.pth'")
    print("="*80)

if __name__ == "__main__":
    # بررسی و نصب کتابخانه‌های ضروری
    try:
        import torch
        import torchvision
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib
        import seaborn as sns
        import albumentations as A
        import timm
    except ImportError as e:
        print(f"Missing library: {e}")
        print("\nPlease install required packages:")
        print("pip install torch torchvision pandas numpy scikit-learn")
        print("pip install matplotlib seaborn albumentations timm tqdm")
        sys.exit(1)
    
    main()