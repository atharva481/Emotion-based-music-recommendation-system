"""
Model Evaluation Script for Facial Emotion Recognition
Calculates: Confusion Matrix, Accuracy, Precision, Recall, F1 Score, ROC-AUC
"""

import numpy as np
import cv2
from keras.models import load_model
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, 
                             classification_report, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_PATH = "facialemotionmodel.h5"
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_EMOTIONS = len(EMOTIONS)

def load_fer2013_sample_data(num_samples=100):
    """
    Load a sample of FER2013 dataset
    This creates synthetic data for demonstration. For real evaluation, 
    download FER2013 from: https://www.kaggle.com/datasets/msambare/fer2013
    """
    print("Loading sample emotion dataset...")
    
    # Create synthetic test data (in production, use real FER2013 data)
    np.random.seed(42)
    
    X_test = np.random.randint(0, 256, (num_samples, 48, 48, 1), dtype=np.uint8)
    y_test = np.random.randint(0, NUM_EMOTIONS, num_samples)
    
    # Normalize
    X_test = X_test.astype('float32') / 255.0
    
    print(f"Loaded {num_samples} test samples")
    print(f"Test data shape: {X_test.shape}")
    print(f"Labels: {np.unique(y_test)}")
    
    return X_test, y_test

def load_model_checkpoint(model_path):
    """Load the pre-trained emotion detection model"""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    return model

def predict_emotions(model, X_test):
    """Make predictions on test data"""
    print("Making predictions...")
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_pred_proba = predictions
    print(f"Predictions shape: {y_pred.shape}")
    return y_pred, y_pred_proba

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all performance metrics"""
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Precision, Recall, F1 (weighted for multi-class)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nWeighted Average:")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall: {recall_weighted:.4f}")
    print(f"  F1 Score: {f1_weighted:.4f}")
    
    print(f"\nMacro Average:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1 Score: {f1_macro:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print("-" * 60)
    for i, emotion in enumerate(EMOTIONS):
        class_precision = precision_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
        class_recall = recall_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
        class_f1 = f1_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
        print(f"{emotion:12} - Precision: {class_precision:.4f}, Recall: {class_recall:.4f}, F1: {class_f1:.4f}")
    
    # ROC-AUC (for multi-class)
    try:
        y_true_bin = label_binarize(y_true, classes=range(NUM_EMOTIONS))
        roc_auc_weighted = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
        print(f"\nROC-AUC Score (weighted): {roc_auc_weighted:.4f}")
    except Exception as e:
        print(f"\nROC-AUC calculation skipped: {str(e)}")
        roc_auc_weighted = None
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix shape: {cm.shape}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc_weighted,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(cm, figsize=(10, 8)):
    """Plot confusion matrix"""
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, cbar=True)
    plt.title('Confusion Matrix - Emotion Detection Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    plt.show()

def plot_roc_curves(y_true, y_pred_proba, figsize=(12, 8)):
    """Plot ROC curves for each emotion class"""
    y_true_bin = label_binarize(y_true, classes=range(NUM_EMOTIONS))
    
    plt.figure(figsize=figsize)
    colors = plt.cm.Set3(np.linspace(0, 1, NUM_EMOTIONS))
    
    for i, emotion in enumerate(EMOTIONS):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{emotion} (AUC = {roc_auc:.2f})', 
                color=colors[i], linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Emotion Detection Model', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("ROC curves saved as 'roc_curves.png'")
    plt.show()

def plot_metric_comparison(metrics, figsize=(12, 6)):
    """Plot comparison of key metrics"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy, Precision, Recall, F1
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    weighted_values = [
        metrics['accuracy'],
        metrics['precision_weighted'],
        metrics['recall_weighted'],
        metrics['f1_weighted']
    ]
    macro_values = [
        metrics['accuracy'],
        metrics['precision_macro'],
        metrics['recall_macro'],
        metrics['f1_macro']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0].bar(x - width/2, weighted_values, width, label='Weighted', color='steelblue')
    axes[0].bar(x + width/2, macro_values, width, label='Macro', color='coral')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Weighted vs Macro Averages', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names, rotation=45)
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Per-class F1 scores
    per_class_f1 = []
    for i in range(NUM_EMOTIONS):
        class_f1 = f1_score(metrics.get('y_true', []), 
                           metrics.get('y_pred', []), 
                           labels=[i], average='micro', zero_division=0)
        per_class_f1.append(class_f1)
    
    axes[1].barh(EMOTIONS, per_class_f1, color='mediumseagreen')
    axes[1].set_xlabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score by Emotion Class', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("Metrics comparison saved as 'metrics_comparison.png'")
    plt.show()

def main():
    """Main evaluation pipeline"""
    print("\n" + "="*60)
    print("EMOTION DETECTION MODEL EVALUATION")
    print("="*60 + "\n")
    
    try:
        # Load model
        model = load_model_checkpoint(MODEL_PATH)
        
        # Load test data
        X_test, y_test = load_fer2013_sample_data(num_samples=500)
        
        # Make predictions
        y_pred, y_pred_proba = predict_emotions(model, X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store predictions for plotting
        metrics['y_true'] = y_test
        metrics['y_pred'] = y_pred
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_confusion_matrix(metrics['confusion_matrix'])
        plot_roc_curves(y_test, y_pred_proba)
        plot_metric_comparison(metrics)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print("\nGenerated files:")
        print("  - confusion_matrix.png")
        print("  - roc_curves.png")
        print("  - metrics_comparison.png")
        
    except FileNotFoundError:
        print(f"\nError: Model file '{MODEL_PATH}' not found!")
        print("Please ensure the model is in the same directory as this script.")
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
