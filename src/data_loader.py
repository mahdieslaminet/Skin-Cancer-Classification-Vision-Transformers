import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HAM10000Dataset(Dataset):
    """Dataset کلاس برای HAM10000"""
    
    def __init__(self, metadata_df: pd.DataFrame, 
                 image_dir_part1: str, 
                 image_dir_part2: str,
                 transform=None,
                 mode: str = 'train'):
        
        self.metadata_df = metadata_df
        self.image_dir_part1 = image_dir_part1
        self.image_dir_part2 = image_dir_part2
        self.transform = transform
        self.mode = mode
        
        # تصحیح مسیر تصاویر
        self._prepare_image_paths()
        
    def _prepare_image_paths(self):
        """آماده‌سازی مسیر تصاویر"""
        self.image_paths = []
        self.labels = []
        
        for idx, row in self.metadata_df.iterrows():
            image_id = row['image_id']
            
            # پیدا کردن label
            if 'dx' in row:
                label_str = row['dx']
            elif 'label' in row:
                label_str = row['label']
            else:
                continue
            
            # اگر label عددی نیست، map کن
            if isinstance(label_str, str):
                from src.config import Config
                config = Config()
                if label_str in config.class_mapping:
                    label = config.class_mapping[label_str]
                else:
                    continue
            else:
                label = int(label_str)
            
            # جستجوی تصویر
            image_path = None
            
            # بررسی در پوشه اول
            possible_path1 = os.path.join(self.image_dir_part1, f"{image_id}.jpg")
            if os.path.exists(possible_path1):
                image_path = possible_path1
            
            # بررسی در پوشه دوم
            if not image_path:
                possible_path2 = os.path.join(self.image_dir_part2, f"{image_id}.jpg")
                if os.path.exists(possible_path2):
                    image_path = possible_path2
            
            # بررسی با پسوند‌های مختلف
            if not image_path:
                for ext in ['.jpg', '.jpeg', '.png']:
                    possible_path1 = os.path.join(self.image_dir_part1, f"{image_id}{ext}")
                    possible_path2 = os.path.join(self.image_dir_part2, f"{image_id}{ext}")
                    
                    if os.path.exists(possible_path1):
                        image_path = possible_path1
                        break
                    elif os.path.exists(possible_path2):
                        image_path = possible_path2
                        break
            
            if image_path:
                self.image_paths.append(image_path)
                self.labels.append(label)
            else:
                # اگر تصویر پیدا نشد، skip کن
                continue
        
        print(f"Loaded {len(self.image_paths)} images for {self.mode} set")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # بارگذاری تصویر
        try:
            image = cv2.imread(image_path)
            if image is None:
                # اگر تصویر load نشد، یک تصویر سیاه برگردان
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
            
            # اعمال transform
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # پیش‌پردازش پایه
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            # در صورت خطا، یک تصویر placeholder برگردان
            print(f"Error loading image {image_path}: {e}")
            placeholder = np.zeros((224, 224, 3), dtype=np.float32)
            placeholder = torch.from_numpy(placeholder).permute(2, 0, 1).float()
            return placeholder, torch.tensor(label, dtype=torch.long)

class DataLoaderModule:
    """مدیریت بارگذاری داده‌ها"""
    
    def __init__(self, config):
        self.config = config
        self.metadata_df = None
        
    def load_metadata(self):
        """بارگذاری متادیتای دیتاست"""
        try:
            self.metadata_df = pd.read_csv(self.config.metadata_path)
            print(f"✓ Loaded metadata with {len(self.metadata_df)} samples")
            
            # نمایش توزیع کلاس‌ها
            print("\nClass Distribution:")
            print(self.metadata_df['dx'].value_counts())
            
            return self.metadata_df
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None
    
    def get_train_transforms(self):
        """تبدیل‌های آموزشی مطابق مقاله"""
        return A.Compose([
            A.Resize(224, 224),
            A.Rotate(limit=8, p=0.5),  # مقاله: چرخش تا ۸ درجه
            A.HorizontalFlip(p=0.5),   # مقاله: flip افقی
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    def get_val_transforms(self):
        """تبدیل‌های اعتبارسنجی"""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    def create_data_loaders(self, train_df, val_df, test_df=None):
        """ایجاد DataLoaderها"""
        
        # ایجاد datasets
        train_dataset = HAM10000Dataset(
            train_df,
            self.config.images_part1,
            self.config.images_part2,
            transform=self.get_train_transforms(),
            mode='train'
        )
        
        val_dataset = HAM10000Dataset(
            val_df,
            self.config.images_part1,
            self.config.images_part2,
            transform=self.get_val_transforms(),
            mode='val'
        )
        
        # ایجاد dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=min(2, self.config.num_workers),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=min(2, self.config.num_workers),
            pin_memory=True
        )
        
        if test_df is not None:
            test_dataset = HAM10000Dataset(
                test_df,
                self.config.images_part1,
                self.config.images_part2,
                transform=self.get_val_transforms(),
                mode='test'
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=min(2, self.config.num_workers)
            )
            
            return train_loader, val_loader, test_loader
        
        return train_loader, val_loader