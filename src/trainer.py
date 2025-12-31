import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class Trainer:
    """کلاس آموزش مدل"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ایجاد classifier
        from src.model import SkinCancerClassifier
        self.classifier = SkinCancerClassifier(config)
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """آموزش یک epoch"""
        self.classifier.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.classifier.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.model.parameters(), 1.0)
            optimizer.step()
            
            # آمارها
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        # محاسبه متریک‌ها
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate(self, val_loader, criterion):
        """اعتبارسنجی"""
        self.classifier.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.classifier.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                current_loss = total_loss / len(all_preds) * val_loader.batch_size
                current_acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.4f}'
                })
        
        # محاسبه متریک‌ها
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """آموزش کامل مدل"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        # آماده‌سازی
        optimizer = self.classifier.get_optimizer()
        scheduler = self.classifier.get_scheduler(optimizer)
        criterion = self.classifier.criterion
        
        best_val_acc = 0
        patience_counter = 0
        patience = 7
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Model type: {self.config.model_type}")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            # آموزش
            train_loss, train_acc, train_f1 = self.train_epoch(
                train_loader, optimizer, criterion, epoch
            )
            
            # اعتبارسنجی
            val_loss, val_acc, val_f1, val_preds, val_labels = self.validate(
                val_loader, criterion
            )
            
            # به‌روزرسانی scheduler
            scheduler.step(val_loss)
            
            # ذخیره تاریخچه
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # نمایش نتایج
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # ذخیره بهترین مدل
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # ذخیره مدل
                model_path = Path(self.config.model_save_dir) / f"best_model.pth"
                self.classifier.save_model(str(model_path))
                print(f"✓ New best model saved! (Accuracy: {val_acc:.4f})")
            else:
                patience_counter += 1
                print(f"✗ No improvement ({patience_counter}/{patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # رسم نمودارها هر 5 epoch
            if (epoch + 1) % 5 == 0:
                self.plot_training_history(epoch + 1)
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"{'='*50}")
        
        # رسم نمودارهای نهایی
        self.plot_training_history(num_epochs)
        
        return self.history
    
    def plot_training_history(self, epoch):
        """رسم نمودارهای آموزش"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs_range, self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs_range, self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs_range, self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(epochs_range, self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[0, 2].plot(epochs_range, self.history['train_f1'], label='Train F1')
        axes[0, 2].plot(epochs_range, self.history['val_f1'], label='Val F1')
        axes[0, 2].set_title('Training and Validation F1 Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs_range, self.history['learning_rates'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss Comparison
        axes[1, 1].bar(['Train', 'Validation'], 
                      [self.history['train_loss'][-1], self.history['val_loss'][-1]])
        axes[1, 1].set_title('Final Loss Comparison')
        axes[1, 1].set_ylabel('Loss')
        
        # Accuracy Comparison
        axes[1, 2].bar(['Train', 'Validation'], 
                      [self.history['train_acc'][-1], self.history['val_acc'][-1]])
        axes[1, 2].set_title('Final Accuracy Comparison')
        axes[1, 2].set_ylabel('Accuracy')
        
        plt.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        
        # ذخیره تصویر
        plot_path = Path(self.config.plots_dir) / f'training_history_epoch_{epoch}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training plots saved to {plot_path}")
    
    def evaluate(self, test_loader):
        """ارزیابی روی تست ست"""
        self.classifier.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.classifier.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # محاسبه متریک‌ها
        from sklearn.metrics import classification_report, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, 
                                      target_names=list(self.config.class_mapping.keys()))
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        return results