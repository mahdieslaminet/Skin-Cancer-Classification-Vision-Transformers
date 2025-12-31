import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

class SkinCancerModel(nn.Module):
    """مدل اصلی برای طبقه‌بندی سرطان پوست"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = len(config.class_mapping)
        
        if config.model_type == "vit":
            self.model = self._create_vit_model()
        elif config.model_type == "resnet50":
            self.model = self._create_resnet50_model()
        elif config.model_type == "efficientnet":
            self.model = self._create_efficientnet_model()
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        print(f"✓ Created {config.model_type} model")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _create_vit_model(self):
        """ایجاد Vision Transformer"""
        try:
            # بارگذاری ViT پیش‌آموزش‌دیده
            vit = timm.create_model(
                'vit_base_patch16_224',
                pretrained=self.config.use_pretrained,
                num_classes=0  # بدون head
            )
            
            # فریز کردن backbone اگر لازم باشد
            if self.config.freeze_backbone:
                for param in vit.parameters():
                    param.requires_grad = False
            
            # اضافه کردن head جدید
            num_features = vit.num_features
            
            head = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)
            )
            
            return nn.Sequential(vit, head)
            
        except Exception as e:
            print(f"Error creating ViT: {e}, using ResNet instead")
            return self._create_resnet50_model()
    
    def _create_resnet50_model(self):
        """ایجاد ResNet50 (پیش‌فرض)"""
        model = models.resnet50(pretrained=self.config.use_pretrained)
        
        # فریز کردن لایه‌ها اگر لازم باشد
        if self.config.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # جایگزینی آخرین لایه
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        return model
    
    def _create_efficientnet_model(self):
        """ایجاد EfficientNet"""
        model = models.efficientnet_b0(pretrained=self.config.use_pretrained)
        
        if self.config.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # جایگزینی آخرین لایه
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        return model
    
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_layers(self):
        """آزاد کردن لایه‌ها برای fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ All layers unfrozen")

class SkinCancerClassifier:
    """کلاس wrapper برای مدل"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # ایجاد مدل
        self.model = SkinCancerModel(config)
        self.model.to(self.device)
        
        # محاسبه وزن‌های کلاس
        self.class_weights = self._calculate_class_weights()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights, device=self.device)
        )
        
        print(f"✓ Model initialized on {self.device}")
    
    def _calculate_class_weights(self):
        """محاسبه وزن‌های کلاس برای مقابله با imbalance"""
        # وزن‌های تقریبی از مقاله
        weights = {
            'akiec': 1.0,
            'bcc': 1.0,
            'bkl': 1.0,
            'df': 1.0,
            'mel': 2.0,  # مهم‌ترین
            'nv': 0.5,   # شایع‌ترین
            'vasc': 1.0
        }
        
        # تبدیل به لیست
        weight_list = []
        for class_name in self.config.class_mapping.keys():
            weight_list.append(weights.get(class_name, 1.0))
        
        return weight_list
    
    def get_optimizer(self, learning_rate=None):
        """ایجاد optimizer"""
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        # فقط پارامترهای trainable
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def get_scheduler(self, optimizer):
        """ایجاد learning rate scheduler"""
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return scheduler
    
    def save_model(self, path):
        """ذخیره مدل"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'class_weights': self.class_weights
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path):
        """بارگذاری مدل"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {path}")
    
    def predict(self, images):
        """پیش‌بینی روی دسته‌ای از تصاویر"""
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
        return predictions, probabilities