import os
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Config:
    """پیکربندی کامل پروژه مطابق مقاله"""
    
    # ============ مسیرهای داده ============
    data_dir: str = "data"
    images_part1: str = os.path.join("data", "HAM10000_images_part_1")
    images_part2: str = os.path.join("data", "HAM10000_images_part_2")
    metadata_path: str = os.path.join("data", "HAM10000_metadata.csv")
    
    # ============ پارامترهای تصویر ============
    image_size: tuple = (224, 224)
    batch_size: int = 32
    num_workers: int = 4 if os.cpu_count() > 4 else 2
    
    # ============ کلاس‌ها (7 کلاس مطابق مقاله) ============
    class_mapping: Dict[str, int] = field(default_factory=lambda: {
        'akiec': 0,  # Actinic Keratoses
        'bcc': 1,    # Basal cell carcinoma
        'bkl': 2,    # Benign keratosis-like lesions
        'df': 3,     # Dermatofibroma
        'mel': 4,    # Melanoma
        'nv': 5,     # Melanocytic nevi
        'vasc': 6    # Vascular lesions
    })
    
    reverse_class_mapping: Dict[int, str] = field(default_factory=lambda: {
        0: 'akiec',
        1: 'bcc',
        2: 'bkl',
        3: 'df',
        4: 'mel',
        5: 'nv',
        6: 'vasc'
    })
    
    class_names: Dict[str, str] = field(default_factory=lambda: {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevus',
        'vasc': 'Vascular Lesion'
    })
    
    # ============ پارامترهای آموزش ============
    num_epochs: int = 30
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    dropout_rate: float = 0.3
    
    # ============ پارامترهای ViT ============
    vit_patch_size: int = 16
    vit_dim: int = 768
    vit_depth: int = 12
    vit_heads: int = 12
    vit_mlp_dim: int = 3072
    
    # ============ کنترل آموزش ============
    freeze_backbone: bool = True
    use_pretrained: bool = True
    model_type: str = "resnet50"  # "vit" یا "resnet50" یا "efficientnet"
    
    # ============ Augmentation ============
    rotation_range: int = 8
    brightness_range: tuple = (0.9, 1.1)
    horizontal_flip: bool = True
    
    # ============ تقسیم داده ============
    test_size: float = 0.2  # 20% تست
    val_size: float = 0.1   # 10% اعتبارسنجی
    random_state: int = 42
    
    # ============ مسیرهای ذخیره ============
    model_save_dir: str = "models"
    results_dir: str = "results"
    plots_dir: str = "results/plots"
    logs_dir: str = "results/logs"
    
    def __post_init__(self):
        """ایجاد دایرکتوری‌های لازم"""
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)