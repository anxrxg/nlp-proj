#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model evaluation module for sentiment classification.

This script evaluates the performance of a trained sentiment classification model.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Import local modules
from train import main as train_model
from feature_extract import extract_features

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')

def load_model(model_type='logistic_regression'):
    """
    Load a trained model from disk.
    
    Args:
        model_type: Type of model to load ('logistic_regression', 'naive_bayes', 'svm', or 'random_forest')
        
    Returns:
        Loaded model
    """
    # Create the model filename
    model_filename = f"{model_type}_model.joblib"
    model_path = os.path.join(models_dir, model_filename)
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating model on test data...")
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Print evaluation metrics
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Return evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }
    
    return metrics

def plot_confusion_matrix(cm, save_path=None):
    """
    Plot the confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot. If None, the plot is displayed but not saved.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def plot_roc_curve(model, X_test, y_test, save_path=None):
    """
    Plot the ROC curve for binary classification.
    
    Args:
        model: Trained model with predict_proba method
        X_test: Test features
        y_test: Test labels
        save_path: Path to save the plot. If None, the plot is displayed but not saved.
    """
    # Check if the model has predict_proba method
    if not hasattr(model, 'predict_proba'):
        print("Model does not support probability predictions. Skipping ROC curve.")
        return
    
    # Get probability predictions
    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        print("Error getting probability predictions. Skipping ROC curve.")
        return
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path)
        print(f"ROC curve plot saved to {save_path}")
    
    plt.show()

def main(model_type='logistic_regression', train_new_model=False):
    """
    Main function to execute the model evaluation pipeline.
    
    Args:
        model_type: Type of model to evaluate ('logistic_regression', 'naive_bayes', 'svm', or 'random_forest')
        train_new_model: Whether to train a new model or load an existing one
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Starting model evaluation...")
    
    # Train a new model or load an existing one
    if train_new_model:
        print("Training new model...")
        model, X_train, y_train, X_test, y_test = train_model(model_type=model_type)
    else:
        # Load the model
        model = load_model(model_type=model_type)
        
        # Extract features
        _, _, X_test, y_test, _ = extract_features()
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Plot ROC curve if the model supports probability predictions
    try:
        plot_roc_curve(model, X_test, y_test)
    except:
        print("Skipping ROC curve plot due to error.")
    
    print("Model evaluation completed successfully!")
    
    return metrics

if __name__ == "__main__":
    # Evaluate a logistic regression model by default
    main(model_type='logistic_regression')