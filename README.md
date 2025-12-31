# 🏥 Skin Cancer Classification using Vision Transformers

## 📋 Project Demo
http://skin-cancer-classifier.ir/

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture](#️-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web Application](#-web-application)
- [Results](#-results)
- [Technical Details](#-technical-details)

## 🎯 Overview

A state-of-the-art deep learning system for multi-class skin cancer classification using **Vision Transformers (ViT)** and **CNN-based models**. This project implements cutting-edge computer vision techniques to accurately classify **7 different types of skin lesions** from dermatoscopic images with **92.14% accuracy**.

**Key Features:**
- 🏥 **Medical-grade accuracy** for skin cancer detection
- 🤖 **Multiple model architectures** (ViT, ResNet, DenseNet, VGG)
- 🌐 **Web application** for easy testing and deployment
- 📊 **Comprehensive evaluation** with detailed metrics
- ⚡ **GPU/CPU support** with optimized training

## 📊 Dataset

**HAM10000 Dataset** - 10,015 dermatoscopic images across 7 classes:

| Class | Full Name | Samples | Percentage | Severity |
|-------|-----------|---------|------------|----------|
| **nv** | Melanocytic Nevi | 6,705 | 66.95% | Benign |
| **mel** | Melanoma | 1,113 | 11.11% | Malignant |
| **bkl** | Benign Keratosis | 1,099 | 10.97% | Benign |
| **bcc** | Basal Cell Carcinoma | 514 | 5.13% | Malignant |
| **akiec** | Actinic Keratoses | 327 | 3.27% | Pre-cancerous |
| **vasc** | Vascular Lesions | 142 | 1.42% | Benign |
| **df** | Dermatofibroma | 115 | 1.15% | Benign |

*Dataset Source: [Kaggle - HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)*

## 🏗️ Architecture

### 🧠 Vision Transformer (ViT) - Primary Model
```
Input (224×224×3) → Patch Embedding → Transformer Encoder → MLP Head → Output (7 classes)
```
- **Patch Size**: 16×16 pixels
- **Transformer Layers**: 12
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Trainable Parameters**: 85.8M
- **Pre-trained**: ImageNet-21k

### 🔄 Alternative Models
- **ResNet50/101/152** (CNN baseline)
- **DenseNet121/169/201**
- **VGG16/19**
- **Custom CNN architectures**

### ⚙️ Training Pipeline
```
Data Loading → Augmentation → Model Training → Evaluation → Deployment
```

## 🚀 Features

### 🔬 **Medical AI Capabilities**
- ✅ **7-class classification** of skin lesions
- ✅ **High accuracy** (92.14% with ViT)
- ✅ **Class imbalance handling** for medical data
- ✅ **Robust data augmentation** techniques
- ✅ **Comprehensive medical evaluation metrics**

### 💻 **Technical Features**
- ✅ **Multi-framework support** (PyTorch & TensorFlow)
- ✅ **Two-phase training** with fine-tuning
- ✅ **Advanced augmentation** with Albumentations
- ✅ **Real-time web interface** with Flask
- ✅ **Model interpretability** tools
- ✅ **Cross-platform compatibility**

### 📈 **Performance Highlights**
| Metric | Vision Transformer | ResNet50 | Improvement |
|--------|-------------------|----------|-------------|
| **Accuracy** | **92.14%** | 82.00% | +10.14% |
| **Precision** | **92.61%** | 81.50% | +11.11% |
| **Recall** | **92.14%** | 82.00% | +10.14% |
| **F1-Score** | **92.17%** | 81.75% | +10.42% |

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- GPU with CUDA support (optional but recommended)

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/skin-cancer-classification.git
cd skin-cancer-classification
```

2. **Create virtual environment (recommended)**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
```bash
# Download from Kaggle (requires Kaggle API)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/
```

5. **Verify installation**
```bash
python -c "import torch; import tensorflow as tf; print('Installation successful!')"
```

### Quick Installation (Minimal)
```bash
# Minimal dependencies for quick testing
pip install torch torchvision albumentations pillow numpy pandas
```

## 🏃 Usage

### Training the Model

**Option 1: Quick Training (CPU-friendly)**
```bash
python scripts/train_incremental.py
```
*Trains a lightweight model on a subset of data*

**Option 2: Full Training with ViT**
```bash
python scripts/train_model.py
```
*Trains the complete Vision Transformer model (requires GPU)*

**Option 3: Google Colab (Recommended for GPU)**
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `notebooks/colab_training.ipynb`
3. Follow the instructions in the notebook

### Evaluating the Model
```bash
# Evaluate on test set
python scripts/evaluate_model.py

# Generate detailed reports
python scripts/generate_report.py
```

### Making Predictions
```python
from src.predictor import SkinCancerPredictor

# Load model
predictor = SkinCancerPredictor("models/best_model.pth")

# Predict single image
result = predictor.predict("path/to/image.jpg")
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🌐 Web Application

### 🚀 Live Demo
Access the web application at: [Your Deployment URL]

### 🏗️ Local Deployment

1. **Navigate to web app directory**
```bash
cd web_app
```

2. **Install web dependencies**
```bash
pip install flask flask-cors pillow
```

3. **Run the Flask server**
```bash
python app.py
```

4. **Open your browser**
```
http://localhost:5000
```

### 🖥️ Web App Features
- **Image Upload**: Drag & drop or file selector
- **Real-time Prediction**: Instant classification results
- **Confidence Scores**: Percentage for each class
- **History**: Save and view previous predictions
- **Mobile Responsive**: Works on all devices

### 📱 API Usage
```bash
# Upload image via API
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict

# Response format
{
    "success": true,
    "prediction": "melanoma",
    "confidence": 0.9214,
    "all_predictions": {
        "melanoma": 0.9214,
        "nevus": 0.0452,
        "bcc": 0.0211,
        ...
    }
}
```

## 📊 Results

### 📈 Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Vision Transformer** | **92.14%** | **92.61%** | **92.14%** | **92.17%** | 4.5 hours |
| ResNet152 | 88.21% | 88.50% | 88.21% | 88.35% | 3.2 hours |
| DenseNet201 | 89.43% | 89.70% | 89.43% | 89.56% | 3.8 hours |
| VGG19 | 85.12% | 85.40% | 85.12% | 85.26% | 2.9 hours |

### 📊 Confusion Matrix (ViT)
![Confusion Matrix](results/plots/confusion_matrix.png)

### 📉 Training Curves
![Training History](results/plots/training_history.png)

### 🎯 ROC Curves
![ROC Curves](results/plots/roc_curves.png)

## 🔧 Technical Details

### Data Preprocessing Pipeline
```python
1. Resize → 224×224 pixels
2. Normalize → ImageNet statistics
3. Augmentation → Rotation, flipping, color adjustments
4. Batching → 32 samples per batch
```

### Data Augmentation Strategies
- **Geometric**: Rotation (±15°), Horizontal/Vertical Flip, Zoom (80-120%)
- **Color**: Brightness (±20%), Contrast (±20%), Hue (±0.1)
- **Advanced**: CLAHE, Gaussian Noise, Motion Blur
- **Medical-specific**: Elastic deformations, grid distortions

### Class Imbalance Solutions
1. **Weighted Loss Function**
   ```python
   weights = [2.0, 1.5, 1.5, 3.0, 2.5, 0.3, 2.0]  # Based on class frequency
   criterion = nn.CrossEntropyLoss(weight=weights)
   ```

2. **Oversampling Minority Classes**
3. **Data Augmentation for Rare Classes**
4. **Focal Loss for Hard Examples**

### Training Configuration
```yaml
# Hyperparameters
learning_rate: 0.001
batch_size: 32
epochs: 50
optimizer: AdamW
scheduler: CosineAnnealingWarmRestarts
dropout: 0.3
weight_decay: 0.0001
```

## 🎮 Quick Start Examples

### Example 1: Quick Test
```python
# Quick test with sample image
python -c "
from src.utils import load_sample_image, predict
image = load_sample_image()
result = predict(image)
print(f'Prediction: {result}')
"
```

### Example 2: Batch Prediction
```bash
# Predict all images in a folder
python scripts/batch_predict.py --input_dir test_images/ --output_dir predictions/
```

### Example 3: Train Custom Model
```bash
# Train with custom parameters
python scripts/train_model.py \
  --model vit \
  --epochs 30 \
  --batch_size 16 \
  --learning_rate 0.0005
```

## 🏆 Achievements

- ✅ **92.14% Accuracy** on HAM10000 dataset
- ✅ **Real-time web application** for easy access
- ✅ **Comprehensive documentation** for researchers
- ✅ **Production-ready code** with best practices
- ✅ **Active maintenance** and updates

---
