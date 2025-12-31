#!/usr/bin/env python3
"""
تحلیل مدل ذخیره شده
"""

import torch
import os

def analyze_saved_model(model_path):
    """تحلیل مدل ذخیره شده"""
    print(f"Analyzing model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        # بارگذاری مدل
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"\n1. Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"2. Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("   Contains 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("   Contains 'state_dict'")
            else:
                state_dict = checkpoint
                print("   Whole dict is state_dict")
        else:
            state_dict = checkpoint
            print("2. Direct state_dict")
        
        print(f"\n3. State_dict keys (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"   {i+1:2}. {key}: {state_dict[key].shape}")
        
        print(f"\n4. Total keys: {len(state_dict)}")
        
        # تحلیل معماری
        print("\n5. Architecture analysis:")
        
        # تشخیص بر اساس keyها
        keys = list(state_dict.keys())
        
        # بررسی ResNet
        resnet_keys = [k for k in keys if 'backbone' in k or 'layer' in k]
        if resnet_keys:
            print("   - Likely a ResNet-based model")
            print(f"   - Found {len(resnet_keys)} ResNet-related keys")
        
        # بررسی SimpleCNN
        simple_keys = [k for k in keys if 'features' in k or 'classifier' in k]
        if simple_keys:
            print("   - Likely a SimpleCNN model")
            print(f"   - Found {len(simple_keys)} SimpleCNN-related keys")
        
        # بررسی ViT
        vit_keys = [k for k in keys if 'transformer' in k or 'attention' in k or 'patch' in k]
        if vit_keys:
            print("   - Likely a Vision Transformer model")
            print(f"   - Found {len(vit_keys)} ViT-related keys")
        
        print("\n6. Sample weights:")
        for key in list(state_dict.keys())[:5]:
            tensor = state_dict[key]
            if tensor.ndim > 0:
                print(f"   {key}: shape={tensor.shape}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")
        
        return state_dict
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        import traceback
        traceback.print_exc()

def convert_model_for_web(model_path, output_path):
    """تبدیل مدل به فرمت مناسب برای وب"""
    print(f"\nConverting model for web...")
    
    state_dict = analyze_saved_model(model_path)
    if not state_dict:
        return
    
    # اگر مدل ResNet است
    keys = list(state_dict.keys())
    if any('backbone' in k for k in keys):
        print("\nDetected ResNet model. Converting...")
        
        # تغییر نام keys برای تطبیق با LightweightResNet
        new_state_dict = {}
        for key, value in state_dict.items():
            # حذف 'backbone.' از ابتدای key
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '', 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # ذخیره
        torch.save(new_state_dict, output_path)
        print(f"✅ Converted model saved to: {output_path}")
        
        # بررسی مجدد
        analyze_saved_model(output_path)
    
    elif any('features' in k for k in keys):
        print("\nModel is already SimpleCNN format. No conversion needed.")
        # کپی مستقیم
        torch.save(state_dict, output_path)
        print(f"✅ Model copied to: {output_path}")
    
    else:
        print("\n⚠️ Unknown model format. Saving as is...")
        torch.save(state_dict, output_path)
        print(f"✅ Model saved to: {output_path}")

if __name__ == "__main__":
    # تحلیل مدل‌های موجود
    models_to_analyze = [
        '../models/fast_model_best.pth',
        '../models/simple_cnn_best.pth',
        'models/fast_model_best.pth',
        'models/simple_cnn_best.pth'
    ]
    
    for model_path in models_to_analyze:
        if os.path.exists(model_path):
            print("\n" + "="*60)
            analyze_saved_model(model_path)
            
            # تبدیل به فرمت مناسب
            if 'fast_model_best' in model_path:
                output_path = 'models/converted_resnet.pth'
                convert_model_for_web(model_path, output_path)