from setuptools import setup, find_packages

setup(
    name="skin_cancer_classification",
    version="1.0.0",
    author="Your Name",
    description="Skin cancer classification using Vision Transformer",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.4.0",
        "tqdm>=4.65.0",
        "albumentations>=1.3.0",
        "einops>=0.6.1",
    ],
    python_requires=">=3.8",
)