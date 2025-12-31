import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

def predict_image(model, image_path):
    """انجام پیش‌بینی روی یک تصویر"""
    
    # پیش‌پردازش تصویر
    image = Image.open(image_path).convert('RGB')
    image_tensor = model.transform(image).unsqueeze(0).to(model.device)
    
    # پیش‌بینی
    with torch.no_grad():
        outputs = model.model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predictions = probabilities.cpu().numpy()
    
    # گرفتن 3 پیش‌بینی برتر
    top_indices = np.argsort(predictions)[-3:][::-1]
    
    # ایجاد لیست نتایج
    all_predictions = []
    for idx in top_indices:
        class_name = model.class_names[idx]
        confidence = float(predictions[idx] * 100)
        
        all_predictions.append({
            'class': class_name,
            'description': model.class_descriptions[class_name],
            'confidence': confidence,
            'risk_level': model.get_risk_level(class_name)
        })
    
    # تشخیص اصلی
    main_idx = np.argmax(predictions)
    main_class = model.class_names[main_idx]
    main_confidence = float(predictions[main_idx] * 100)
    
    # ایجاد نتیجه نهایی
    result = {
        'is_cancer': model.is_cancer_class(main_class),
        'main_class': main_class,
        'main_description': model.class_descriptions[main_class],
        'main_confidence': main_confidence,
        'risk_level': model.get_risk_level(main_class),
        'recommendation': model.get_recommendation(main_class, main_confidence),
        'all_predictions': all_predictions,
        'top_3_classes': [model.class_names[i] for i in top_indices],
        'top_3_confidences': [float(predictions[i] * 100) for i in top_indices]
    }
    
    # اطلاعات دیباگ (در حالت production می‌توانید حذف کنید)
    result['debug'] = {
        'model_type': model.model_type,
        'predictions_raw': predictions.tolist(),
        'image_size': model.image_size
    }
    
    return result

def validate_prediction(result):
    """اعتبارسنجی نتایج پیش‌بینی"""
    # بررسی اعتبار پیش‌بینی اصلی
    if result['main_confidence'] < 10:
        result['warning'] = 'Low confidence prediction'
    
    # بررسی توزیع احتمالات
    if result['debug']['predictions_raw'][result['main_class']] < 0.5:
        result['warning'] = 'Uncertain prediction'
    
    return result