# data_preprocessor.py
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# اضافه کردن این import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

class DataPreprocessor:
    def __init__(self, config=None):
        if config is None:
            self.config = Config()
        else:
            self.config = config
        
        # ایجاد دایرکتوری‌ها
        self.create_directories()
        
        # مسیرهای فایل
        self.raw_path = self.config.data_dir
        self.metadata_path = os.path.join(self.raw_path, 'HAM10000_metadata.csv')
        self.images_dir = os.path.join(self.raw_path, 'HAM10000_images')
        
    def create_directories(self):
        """ایجاد دایرکتوری‌های مورد نیاز"""
        directories = [
            './data/raw',
            './data/processed',
            './data/augmented',
            './models',
            './results'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def check_dataset(self):
        """بررسی وجود دیتاست"""
        print("Checking dataset...")
        
        # بررسی وجود فایل metadata
        if not os.path.exists(self.metadata_path):
            print(f"Error: Metadata file not found at {self.metadata_path}")
            print("\nPlease download the dataset from:")
            print("https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
            print("\nAnd extract it to ./data/")
            return False
        
        # بررسی وجود پوشه تصاویر
        if not os.path.exists(self.images_dir):
            print(f"Warning: Images directory not found at {self.images_dir}")
            
            # بررسی مسیرهای جایگزین
            alternative_paths = [
                os.path.join(self.raw_path, 'images'),
                os.path.join(self.raw_path, 'HAM10000'),
                os.path.join(self.raw_path, 'HAM10000_images_part_1'),
                os.path.join(self.raw_path, 'HAM10000_images_part_2'),
                self.raw_path
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Found images at: {alt_path}")
                    self.images_dir = alt_path
                    break
            
            if not os.path.exists(self.images_dir):
                print("Could not find images directory.")
                return False
        
        print("Dataset found successfully!")
        return True
    
    def load_metadata(self):
        """بارگذاری metadata"""
        print(f"\nLoading metadata from {self.metadata_path}...")
        df = pd.read_csv(self.metadata_path)
        
        print(f"Loaded {len(df)} records")
        print("\nDataset columns:")
        print(df.columns.tolist())
        
        # بررسی وجود ستون‌های ضروری
        required_columns = ['image_id', 'dx']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        return df
    
    def locate_images(self, df):
        """یافتن مسیر تصاویر"""
        print("\nLocating images...")
        
        # لیست فرمت‌های ممکن
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # بررسی مسیرهای مختلف برای تصاویر
        possible_dirs = [
            self.images_dir,
            os.path.join(self.raw_path, 'HAM10000_images_part_1'),
            os.path.join(self.raw_path, 'HAM10000_images_part_2'),
            os.path.join(self.raw_path, 'images'),
            self.raw_path
        ]
        
        found_images = 0
        image_paths = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_id = row['image_id']
            image_found = False
            
            # جستجو در دایرکتوری‌های مختلف
            for img_dir in possible_dirs:
                if not os.path.exists(img_dir):
                    continue
                    
                # جستجو با فرمت‌های مختلف
                for ext in extensions:
                    possible_path = os.path.join(img_dir, f"{image_id}{ext}")
                    if os.path.exists(possible_path):
                        image_paths.append(possible_path)
                        found_images += 1
                        image_found = True
                        break
                
                if image_found:
                    break
            
            if not image_found:
                # اگر تصویر پیدا نشد، None قرار بده
                image_paths.append(None)
        
        df['image_path'] = image_paths
        
        print(f"Found {found_images} out of {len(df)} images")
        
        # حذف ردیف‌هایی که تصویر ندارند
        df = df[df['image_path'].notna()]
        print(f"Remaining images: {len(df)}")
        
        return df
    
    def analyze_class_distribution(self, df):
        """آنالیز توزیع کلاس‌ها"""
        print("\nClass distribution:")
        class_dist = df['dx'].value_counts()
        
        for class_name, count in class_dist.items():
            percentage = (count / len(df)) * 100
            print(f"{class_name}: {count} ({percentage:.2f}%)")
        
        return class_dist
    
    def create_class_mapping(self):
        """ایجاد mapping برای کلاس‌ها"""
        class_mapping = {
            'akiec': 0,  # Actinic Keratoses
            'bcc': 1,    # Basal Cell Carcinoma
            'bkl': 2,    # Benign Keratosis
            'df': 3,     # Dermatofibroma
            'mel': 4,    # Melanoma
            'nv': 5,     # Melanocytic Nevi
            'vasc': 6    # Vascular Lesions
        }
        
        reverse_mapping = {v: k for k, v in class_mapping.items()}
        
        return class_mapping, reverse_mapping
    
    def create_augmentation_pipeline(self):
        """ایجاد pipeline برای augmentation"""
        return A.Compose([
            A.Rotate(limit=self.config.rotation_range, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        ])
    
    def augment_balance_classes(self, df, target_samples=980):
        print("\nAugmenting and balancing classes...")
        augmentation_pipeline = self.create_augmentation_pipeline()
        grouped = df.groupby('dx')
        balanced_data = []
        
        for class_name, group in grouped:
            print(f"\nProcessing class: {class_name}")
            current_samples = len(group)
            
            if current_samples < target_samples:
                # Augment minority
                num_needed = target_samples - current_samples
                for i in tqdm(range(num_needed), desc=f"Augmenting {class_name}"):
                    random_sample = group.sample(1).iloc[0]
                    img_path = random_sample['image_path']
                    if not os.path.exists(img_path):
                        continue
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    augmented = augmentation_pipeline(image=img)
                    aug_img = augmented['image']
                    aug_filename = f"{class_name}_aug_{i}_{os.path.basename(img_path)}"
                    aug_path = os.path.join('./data/augmented', aug_filename)
                    cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    balanced_data.append({
                        'image_id': f"aug_{class_name}_{i}",
                        'dx': class_name,
                        'image_path': aug_path
                    })
                # Add original
                balanced_data.extend(group.to_dict('records'))
            
            else:
                # Undersample majority (random sampling as in article)
                undersampled = group.sample(n=target_samples, random_state=42)
                balanced_data.extend(undersampled.to_dict('records'))
        
        balanced_df = pd.DataFrame(balanced_data)
        print(f"\nOriginal samples: {len(df)}")
        print(f"Balanced samples: {len(balanced_df)} ({target_samples} per class)")
        return balanced_df
    
    def prepare_train_val_test_split(self, df, test_size=0.2, val_size=0.1):
        """ایجاد train/val/test split"""
        print("\nCreating train/validation/test splits...")
        
        # ایجاد mapping کلاس
        class_mapping, _ = self.create_class_mapping()
        
        # اضافه کردن label عددی
        df['label'] = df['dx'].map(class_mapping)
        
        # حذف ردیف‌هایی که label ندارند
        df = df[df['label'].notna()]
        
        # تقسیم اولیه به train+val و test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            stratify=df['label'], 
            random_state=42
        )
        
        # تقسیم train و validation
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_ratio,
            stratify=train_val_df['label'], 
            random_state=42
        )
        
        # ذخیره split‌ها
        train_df.to_csv('./data/processed/train_metadata.csv', index=False)
        val_df.to_csv('./data/processed/val_metadata.csv', index=False)
        test_df.to_csv('./data/processed/test_metadata.csv', index=False)
        
        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        # نمایش توزیع کلاس در هر split
        print("\nClass distribution in splits:")
        
        for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            print(f"\n{split_name}:")
            class_dist = split_df['dx'].value_counts()
            for class_name, count in class_dist.items():
                percentage = (count / len(split_df)) * 100
                print(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        return train_df, val_df, test_df

def main():
    print("=" * 60)
    print("SKIN CANCER DATASET PREPROCESSING")
    print("=" * 60)
    
    # ایجاد preprocessor
    preprocessor = DataPreprocessor()
    
    # بررسی وجود دیتاست
    if not preprocessor.check_dataset():
        print("\nPlease download the dataset and try again.")
        return
    
    # بارگذاری metadata
    df = preprocessor.load_metadata()
    
    # یافتن مسیر تصاویر
    df = preprocessor.locate_images(df)
    
    # آنالیز توزیع کلاس
    preprocessor.analyze_class_distribution(df)
    
    # ایجاد mapping کلاس
    class_mapping, reverse_mapping = preprocessor.create_class_mapping()
    print(f"\nClass mapping: {class_mapping}")
    
    # ذخیره mapping
    mapping_df = pd.DataFrame({
        'class_name': list(class_mapping.keys()),
        'class_id': list(class_mapping.values())
    })
    mapping_df.to_csv('./data/processed/class_mapping.csv', index=False)
    
    # Augment کردن و متوازن کردن کلاس‌ها
    # برای تست سریع، target_samples را کم کنید
    balanced_df = preprocessor.augment_balance_classes(df, target_samples=200)  # کاهش برای تست سریع
    
    # ایجاد splits
    train_df, val_df, test_df = preprocessor.prepare_train_val_test_split(
        balanced_df, 
        test_size=0.2, 
        val_size=0.1
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nResults saved in ./data/processed/")
    print(f"Class mapping saved as class_mapping.csv")
    print(f"Train metadata: {len(train_df)} samples")
    print(f"Validation metadata: {len(val_df)} samples")
    print(f"Test metadata: {len(test_df)} samples")

if __name__ == "__main__":
    main()