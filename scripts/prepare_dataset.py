#!/usr/bin/env python3
"""
اسکریپت برای آماده‌سازی و تحلیل اولیه داده‌ها
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_loader import DataLoaderModule
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_dataset():
    """تحلیل جامع دیتاست"""
    config = Config()
    
    print("=" * 60)
    print("DATASET ANALYSIS - HAM10000")
    print("=" * 60)
    
    # بارگذاری داده‌ها
    data_loader = DataLoaderModule(config)
    metadata_df = data_loader.load_metadata()
    
    if metadata_df is None:
        print("Failed to load metadata!")
        return
    
    print("\n1. Basic Information:")
    print(f"Total samples: {len(metadata_df)}")
    print(f"Columns: {list(metadata_df.columns)}")
    
    print("\n2. Class Distribution:")
    class_counts = metadata_df['dx'].value_counts()
    print(class_counts)
    
    # نمودار توزیع کلاس‌ها
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.index, class_counts.values)
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.title('Class Distribution in HAM10000 Dataset')
    plt.xticks(rotation=45)
    
    # افزودن مقادیر روی میله‌ها
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/plots/class_distribution.png', dpi=150)
    plt.show()
    
    print("\n3. Demographic Analysis:")
    if 'age' in metadata_df.columns and 'sex' in metadata_df.columns:
        print(f"Average age: {metadata_df['age'].mean():.1f} years")
        print(f"Age range: {metadata_df['age'].min()} - {metadata_df['age'].max()} years")
        print(f"Gender distribution:\n{metadata_df['sex'].value_counts()}")
    
    print("\n4. Localization Analysis:")
    if 'localization' in metadata_df.columns:
        print(f"Top 10 locations:")
        print(metadata_df['localization'].value_counts().head(10))
    
    print("\n5. Dataset Statistics:")
    print(metadata_df.describe())
    
    # ذخیره تحلیل
    analysis_report = f"""
    DATASET ANALYSIS REPORT
    ======================
    
    Dataset: HAM10000
    Total Samples: {len(metadata_df)}
    
    Class Distribution:
    {class_counts.to_string()}
    
    Percentage Distribution:
    {(class_counts / len(metadata_df) * 100).round(2).to_string()}
    """
    
    report_path = os.path.join(config.reports_dir, 'dataset_analysis.txt')
    with open(report_path, 'w') as f:
        f.write(analysis_report)
    
    print(f"\nAnalysis report saved to: {report_path}")

if __name__ == "__main__":
    analyze_dataset()