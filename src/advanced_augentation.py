import cv2
import numpy as np
import albumentations as A
import os
from tqdm import tqdm

class AdvancedAugmentation:
    """Augmentation دقیقاً مطابق مقاله"""
    
    def __init__(self, target_samples=980):
        self.target_samples = target_samples
        self.transform = self.get_paper_transform()
    
    def get_paper_transform(self):
        """تبدیل‌های دقیقاً مطابق مقاله"""
        return A.Compose([
            # مطابق مقاله: rotation up to 8 degrees
            A.Rotate(limit=8, border_mode=cv2.BORDER_REFLECT, p=0.5),
            
            # مطابق مقاله: horizontal flip
            A.HorizontalFlip(p=0.5),
            
            # مطابق مقاله: تغییرات رنگ (brightness, contrast, saturation)
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # ±10%
                contrast_limit=0.1,    # ±10%
                p=0.3
            ),
            
            # مطابق مقاله: تغییرات Hue
            A.HueSaturationValue(
                hue_shift_limit=10,    # محدود برای تصاویر پوست
                sat_shift_limit=0.1,   # ±10%
                val_shift_limit=0.1,   # ±10%
                p=0.3
            ),
            
            # Normalization برای مدل‌های پیش‌آموزش‌دیده
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            
            # تبدیل به تنسور
            ToTensorV2(),
        ])
    
    def augment_to_target(self, images_paths, class_name, output_dir):
        """Augment کردن تا رسیدن به تعداد هدف (980)"""
        import shutil
        import random
        
        os.makedirs(output_dir, exist_ok=True)
        
        current_count = len(images_paths)
        print(f"Class {class_name}: {current_count} → {self.target_samples}")
        
        # اگر تعداد کافی است، فقط کپی کن
        if current_count >= self.target_samples:
            selected = random.sample(images_paths, self.target_samples)
            augmented_paths = []
            for i, img_path in enumerate(selected):
                new_path = os.path.join(output_dir, f"{class_name}_{i:04d}.jpg")
                shutil.copy2(img_path, new_path)
                augmented_paths.append(new_path)
            return augmented_paths
        
        # اگر نیاز به augmentation داریم
        augmented_paths = []
        
        # مرحله 1: کپی کردن تمام تصاویر اصلی
        for i, img_path in enumerate(images_paths):
            new_path = os.path.join(output_dir, f"{class_name}_orig_{i:04d}.jpg")
            shutil.copy2(img_path, new_path)
            augmented_paths.append(new_path)
        
        # مرحله 2: ایجاد augmented samples تا رسیدن به 980
        needed = self.target_samples - current_count
        aug_per_image = max(1, needed // current_count)
        
        for i in tqdm(range(needed), desc=f"Augmenting {class_name}"):
            # انتخاب تصویر تصادفی
            img_path = random.choice(images_paths)
            
            try:
                # بارگذاری تصویر
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # اعمال augmentation
                augmented = self.transform(image=img)
                aug_img = augmented['image']
                
                # تبدیل به numpy array برای ذخیره
                if isinstance(aug_img, torch.Tensor):
                    aug_img = aug_img.permute(1, 2, 0).numpy()
                
                # تبدیل به uint8
                if aug_img.dtype == np.float32:
                    aug_img = (aug_img * 255).astype(np.uint8)
                elif aug_img.max() <= 1.0:
                    aug_img = (aug_img * 255).astype(np.uint8)
                
                # ذخیره تصویر
                aug_path = os.path.join(output_dir, 
                                       f"{class_name}_aug_{i:04d}.jpg")
                cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                augmented_paths.append(aug_path)
                
            except Exception as e:
                print(f"Error augmenting {img_path}: {e}")
                continue
        
        return augmented_paths[:self.target_samples]
    
# استفاده از کلاس
def run_advanced_augmentation():
    aug = AdvancedAugmentation(target_samples=980)
    
    # اینجا باید دیتاست واقعی را بارگذاری کنید
    # paths_by_class = load_your_dataset()
    
    # برای هر کلاس:
    # augmented_paths = aug.augment_class(paths_by_class[class_name], class_name, './data/augmented')