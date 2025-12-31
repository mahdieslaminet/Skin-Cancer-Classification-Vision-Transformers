#!/usr/bin/env python3
"""
اسکریپت برای ارزیابی مدل آموزش‌دیده
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_loader import DataLoaderModule
from src.evaluator import ModelEvaluator

def main():
    """تابع اصلی"""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # بارگذاری پیکربندی
    config = Config()
    
    # بارگذاری داده‌ها
    print("\n1. Loading data...")
    data_loader = DataLoaderModule(config)
    data_loader.load_metadata()
    
    # ایجاد dataloaders
    train_loader, val_loader, test_loader = data_loader.create_data_loaders()
    
    # پیدا کردن آخرین مدل ذخیره‌شده
    import glob
    model_files = glob.glob(os.path.join(config.model_save_dir, "*.pth"))
    
    if not model_files:
        print("No trained models found!")
        return
    
    # انتخاب آخرین مدل
    latest_model = max(model_files, key=os.path.getctime)
    print(f"\n2. Loading model: {latest_model}")
    
    # ارزیابی مدل
    print("\n3. Evaluating model on test set...")
    evaluator = ModelEvaluator(config, latest_model)
    results = evaluator.evaluate(test_loader)
    
    # تست روی یک تصویر نمونه
    print("\n4. Testing on a sample image...")
    # پیدا کردن یک تصویر نمونه
    sample_images = glob.glob("data/HAM10000_images_part_1/*.jpg")
    if sample_images:
        sample_image = sample_images[0]
        evaluator.predict_single_image(sample_image)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()