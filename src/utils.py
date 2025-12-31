import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import tensorflow as tf
import yaml
import json
import os

class ModelEvaluator:
    """کلاس برای ارزیابی مدل‌های مختلف"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = list(self.config['data']['classes'].keys())
        self.num_classes = len(self.class_names)
        
        # ایجاد دایرکتوری results
        os.makedirs('./results', exist_ok=True)
    
    def evaluate_model(self, model, test_generator, model_name='Model'):
        """ارزیابی یک مدل خاص"""
        print(f"\nEvaluating {model_name}...")
        
        # پیش‌بینی
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for i in range(len(test_generator)):
            X_batch, y_batch = test_generator[i]
            
            # پیش‌بینی
            predictions = model.predict(X_batch, verbose=0)
            
            # تبدیل one-hot به label
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
            y_pred_proba.extend(predictions)
        
        # محاسبه metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # محاسبه AUC (برای multi-class)
        y_true_onehot = tf.keras.utils.to_categorical(y_true, self.num_classes)
        try:
            auc = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovo')
        except:
            auc = 0.0
        
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # نمایش results
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, cm, model_name, save_path=None):
        """رسم confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history, model_name, save_path=None):
        """رسم نمودارهای آموزش"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # نمودار accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # نمودار loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, results_dict, save_path='./results/model_comparison.csv'):
        """مقایسه عملکرد مدل‌های مختلف"""
        comparison_data = []
        
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC': results['auc']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Accuracy', ascending=False)
        
        # ذخیره results
        df_comparison.to_csv(save_path, index=False)
        
        # نمایش جدول مقایسه
        print("\nModel Comparison:")
        print(df_comparison.to_string())
        
        # رسم نمودار مقایسه
        self.plot_model_comparison(df_comparison)
        
        return df_comparison
    
    def plot_model_comparison(self, df_comparison):
        """رسم نمودار مقایسه مدل‌ها"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            sorted_df = df_comparison.sort_values(metric, ascending=True)
            
            colors = ['green' if x == sorted_df[metric].iloc[-1] else 'blue' 
                     for x in sorted_df[metric]]
            
            bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=colors)
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Comparison')
            
            # اضافه کردن مقادیر روی bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center')
        
        # حذف axes اضافی
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('./results/model_comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_all_results(self, results_dict, history_dict=None):
        """ذخیره همه results"""
        # ذخیره results عددی
        with open('./results/all_results.json', 'w') as f:
            # تبدیل numpy arrays به لیست
            json_serializable = {}
            for model_name, results in results_dict.items():
                json_serializable[model_name] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in results.items()
                }
            json.dump(json_serializable, f, indent=4)
        
        # ذخیره histories
        if history_dict:
            for model_name, history in history_dict.items():
                hist_df = pd.DataFrame(history.history)
                hist_df.to_csv(f'./results/{model_name}_history.csv', index=False)
        
        print("All results saved to ./results/")