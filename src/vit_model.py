import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from src.config import Config
import timm  

class PreNorm(nn.Module):
    """LayerNorm قبل از attention"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """لایه FeedForward"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """Attention چندسر"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """ترانسفورمر کامل"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    """Vision Transformer اصلی"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # پارامترها
        image_size = config.image_size
        patch_size = config.vit_patch_size
        num_classes = len(config.class_mapping)
        dim = config.vit_dim
        depth = config.vit_depth
        heads = config.vit_heads
        mlp_dim = config.vit_mlp_dim
        channels = 3
        
        # محاسبات
        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0, \
            "Image dimensions must be divisible by patch size"
        
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        patch_dim = channels * patch_size * patch_size
        
        # لایه‌های patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Transformer encoder
        self.transformer = Transformer(dim, depth, heads, dim, mlp_dim, config.dropout_rate)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1024),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """مقداردهی اولیه وزن‌ها"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, img):
        # Patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        # افزودن cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # افزودن positional encoding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # گرفتن خروجی cls token
        x = x[:, 0]
        
        # MLP head
        return self.mlp_head(x)
    
    def freeze_backbone(self):
        """فریز کردن لایه‌های پایه"""
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        for param in self.to_patch_embedding.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """آزاد کردن همه لایه‌ها"""
        for param in self.parameters():
            param.requires_grad = True

class SkinCancerClassifier:
    """کلاس جامع برای طبقه‌بندی سرطان پوست"""
    
    def __init__(self, config: Config, use_paper_vit=True):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # انتخاب مدل
        if use_paper_vit:
            self.model = PaperViT(config)
        else:
            self.model = ViT(config)  # مدل قبلی
        
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"✓ Model initialized")
        print(f"  Device: {self.device}")
        print(f"  Model type: {'PaperViT' if use_paper_vit else 'Custom ViT'}")
        
    def _calculate_class_weights(self):
        """محاسبه وزن‌های کلاس"""
        # وزن‌های تقریبی از مقاله
        weights = {
            'akiec': 1.0,
            'bcc': 1.0,
            'bkl': 1.0,
            'df': 1.0,
            'mel': 2.0,  # مهم‌ترین - وزن بیشتر
            'nv': 0.5,   # شایع‌ترین - وزن کمتر
            'vasc': 1.0
        }
        
        # تبدیل به لیست بر اساس ترتیب class_mapping
        weight_list = []
        for class_name, idx in self.config.class_mapping.items():
            weight_list.append(weights.get(class_name, 1.0))
        
        return weight_list
    def save_model(self, path):
        """ذخیره مدل"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_type': 'PaperViT' if hasattr(self.model, 'vit') else 'CustomViT'
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path):
        """بارگذاری مدل"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {path}")  
   
    def get_optimizer(self, learning_rate=None):
        """ایجاد optimizer"""
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def get_scheduler(self, optimizer):
        """ایجاد learning rate scheduler"""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=learning_rate * 0.01
        )
        return scheduler
            
    def unfreeze_model(self):
        """آزاد کردن کل مدل برای fine-tuning"""
        if hasattr(self.model, 'unfreeze_backbone'):
            self.model.unfreeze_backbone()
        
        # همچنین head را هم trainable کنید
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        print("✓ All model layers unfrozen for fine-tuning")

class PaperViT(nn.Module):
    """ViT دقیقاً مطابق مقاله با استفاده از مدل پیش‌آموزش‌دیده"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # بارگذاری ViT پیش‌آموزش‌دیده از timm
        # مقاله از ViT-Base/16 استفاده کرده است
        try:
            self.vit = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=0  # حذف head اصلی
            )
        except Exception as e:
            print(f"Error loading pretrained ViT: {e}")
            print("Creating ViT from scratch...")
            # اگر دانلود نشد، از پایه بساز
            self.vit = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0
            )
        
        # فریز کردن لایه‌های پایه (optional برای fine-tuning)
        if hasattr(config, 'freeze_backbone') and config.freeze_backbone:
            self._freeze_backbone()
        
        # اضافه کردن head جدید مطابق مقاله
        num_features = 768  # برای ViT-Base
        
        # MLP Head مطابق مقاله - ساده‌تر برای جلوگیری از overfitting
        self.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(config.class_mapping))
        )
        
        # Initialize head weights
        self._init_head_weights()
        
        print(f"ViT Model Info:")
        print(f"  Backbone: {'Frozen' if config.freeze_backbone else 'Trainable'}")
        print(f"  Total params: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _freeze_backbone(self):
        """فریز کردن لایه‌های backbone"""
        for name, param in self.vit.named_parameters():
            param.requires_grad = False
        print("✓ ViT backbone frozen")
    
    def unfreeze_backbone(self):
        """آزاد کردن لایه‌های backbone برای fine-tuning"""
        for name, param in self.vit.named_parameters():
            param.requires_grad = True
        print("✓ ViT backbone unfrozen")
    
    def _init_head_weights(self):
        """مقداردهی اولیه وزن‌های head"""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features from ViT
        features = self.vit(x)
        
        # Classification head
        output = self.head(features)
        return output