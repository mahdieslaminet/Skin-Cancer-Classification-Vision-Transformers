import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

print("=" * 60)
print("🧪 SIMPLE MODEL TEST")
print("=" * 60)

# 1. بررسی وجود مدل
model_path = "models/converted_resnet.pth"
if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    
    # جستجو در مسیرهای دیگر
    possible_paths = [
        "../models/fast_model_best.pth",
        "fast_model_best.pth",
        "../models/converted_resnet.pth"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Found model at: {path}")
            break
    
    if not os.path.exists(model_path):
        print("❌ No model found anywhere!")
        exit()

print(f"📂 Model path: {model_path}")

# 2. ایجاد یک تصویر تست
print("\n🖼️ Creating test image...")
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
Image.fromarray(test_image).save("test_image.jpg")
print("✅ Test image created: test_image.jpg")

# 3. ایجاد مدل
print("\n🤖 Creating model...")
try:
    # مدل ResNet18 ساده
    model = models.resnet18(pretrained=False)
    
    # تغییر لایه خروجی برای 7 کلاس
    model.fc = nn.Linear(512, 7)
    
    device = torch.device('cpu')
    model.to(device)
    
    print("✅ Model created successfully")
    
    # 4. بارگذاری وزن‌ها
    print("\n📥 Loading weights...")
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}...")  # فقط 5 key اول
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Using 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Using 'state_dict'")
        else:
            state_dict = checkpoint
            print("Using whole dict as state_dict")
    else:
        state_dict = checkpoint
        print("Checkpoint is directly state_dict")
    
    # بارگذاری وزن‌ها
    try:
        model.load_state_dict(state_dict)
        print("✅ Weights loaded (strict=True)")
    except Exception as e:
        print(f"⚠️ Error with strict=True: {e}")
        print("Trying strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded (strict=False)")
    
    model.eval()
    
    # 5. تست مدل
    print("\n🧪 Testing model...")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # بارگذاری تصویر تست
    image = Image.open("test_image.jpg").convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"Input shape: {image_tensor.shape}")
    print(f"Input range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    # پیش‌بینی
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    print(f"\n📊 Predictions:")
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    for i, prob in enumerate(probabilities):
        print(f"  {class_names[i]}: {prob.item()*100:.2f}%")
    
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item() * 100
    
    print(f"\n🎯 Predicted: {class_names[predicted_class]} ({confidence:.1f}%)")
    
    # 6. بررسی وزن‌ها
    print(f"\n🔍 Checking weights...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # بررسی چند وزن
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv1' in name:
            print(f"First conv layer - Mean: {param.data.mean():.6f}, Std: {param.data.std():.6f}")
            break
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()