#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script for sentiment classification.

This script demonstrates the entire sentiment classification pipeline from data preparation to prediction.
"""

import os
import time
import pandas as pd

# Import local modules
from data_prep import main as prepare_data
from feature_extract import extract_features
from train import train_model, evaluate_on_training, save_model
from evaluate import evaluate_model, plot_confusion_matrix
from predict import predict_sentiment

def run_demo():
    """
    Run a demonstration of the sentiment classification pipeline.
    """
    print("\n" + "=" * 80)
    print("SENTIMENT CLASSIFICATION DEMO")
    print("=" * 80)
    
    # Step 1: Data Preparation
    print("\n[Step 1] Preparing data...")
    train_df, test_df = prepare_data()
    
    # Step 2: Feature Extraction
    print("\n[Step 2] Extracting features...")
    X_train, y_train, X_test, y_test, extractor = extract_features()
    
    # Step 3: Model Training
    print("\n[Step 3] Training models...")
    models = {}
    model_types = ['logistic_regression', 'naive_bayes', 'svm']
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        model = train_model(X_train, y_train, model_type=model_type)
        train_accuracy = evaluate_on_training(model, X_train, y_train)
        save_model(model, model_type=model_type)
        models[model_type] = model
    
    # Step 4: Model Evaluation
    print("\n[Step 4] Evaluating models...")
    best_model_type = None
    best_accuracy = 0
    
    for model_type, model in models.items():
        print(f"\nEvaluating {model_type} model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Track the best model
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model_type = model_type
    
    print(f"\nBest model: {best_model_type} with accuracy {best_accuracy:.4f}")
    
    # Step 5: Prediction Demo
    print("\n[Step 5] Prediction demo...")
    
    # Sample texts for prediction
    sample_texts = [
        "I really enjoyed this product. It works great and the customer service was excellent!",
        "This is the worst experience I've ever had. The product broke after one use.",
        "The movie was okay, not great but not terrible either.",
        "I'm very satisfied with my purchase and would recommend it to others."
    ]
    
    # Make predictions using the best model
    print(f"\nMaking predictions using the {best_model_type} model:")
    for text in sample_texts:
        result = predict_sentiment(text, model_type=best_model_type)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()