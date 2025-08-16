#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentiment Analysis Pipeline Runner

This script runs the complete sentiment analysis pipeline:
1. Data preparation
2. Feature extraction
3. Model training
4. Model evaluation
5. Example predictions
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.data_prep import load_imdb_dataset, preprocess_data, split_and_save_data
from src.feature_extract import TfidfFeatureExtractor
from src.model import SentimentClassifier, train_and_evaluate
from src.predict_sentiment import predict_sentiment

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

def run_pipeline(dataset_choice='auto', max_samples=None, test_size=0.2, 
                model_type='logistic_regression', evaluate_only=False,
                predict_samples=True):
    """
    Run the complete sentiment analysis pipeline.
    
    Args:
        dataset_choice: Which dataset to use ('1', '2', '3', or 'auto')
        max_samples: Maximum number of samples to use (for faster testing)
        test_size: Proportion of data to use for testing
        model_type: Type of model to train ('logistic_regression', 'naive_bayes', 'svm', 'random_forest')
        evaluate_only: If True, skip training and only evaluate existing model
        predict_samples: If True, run predictions on sample texts
    """
    print("\n" + "=" * 80)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Step 1: Data Preparation
    if not evaluate_only:
        print("\n[1/5] Data Preparation")
        print("-" * 80)
        
        # Load and preprocess data
        print(f"Loading dataset (choice: {dataset_choice})...")
        data = load_imdb_dataset(dataset_choice=dataset_choice)
        
        # Limit samples if specified
        if max_samples and max_samples < len(data):
            print(f"Limiting to {max_samples} samples for faster processing")
            data = data.sample(max_samples, random_state=42).reset_index(drop=True)
        
        print(f"Dataset size: {len(data)} samples")
        print("Class distribution:")
        print(data['sentiment'].value_counts())
        
        # Preprocess data
        print("\nPreprocessing data...")
        data = preprocess_data(data)
        
        # Split data
        print("\nSplitting data into train/test sets...")
        train_data, test_data = split_and_save_data(data, test_size=test_size)
        print(f"Train set: {len(train_data)} samples")
        print(f"Test set: {len(test_data)} samples")
    else:
        print("\n[1/5] Data Preparation - SKIPPED (Evaluation only mode)")
        print("-" * 80)
        
        # Load preprocessed data
        train_data = pd.read_csv(os.path.join(data_dir, 'processed', 'train.csv'))
        test_data = pd.read_csv(os.path.join(data_dir, 'processed', 'test.csv'))
        print(f"Loaded preprocessed data - Train: {len(train_data)}, Test: {len(test_data)} samples")
    
    # Step 2: Feature Extraction
    if not evaluate_only:
        print("\n[2/5] Feature Extraction")
        print("-" * 80)
        
        # Create and fit TF-IDF vectorizer
        print("Extracting TF-IDF features...")
        feature_extractor = TfidfFeatureExtractor(max_features=10000, min_df=5, max_df=0.8, ngram_range=(1, 2))
        X_train = feature_extractor.fit_transform(train_data['preprocessed_text'])
        X_test = feature_extractor.transform(test_data['preprocessed_text'])
        y_train = train_data['sentiment']
        y_test = test_data['sentiment']
        
        print(f"Feature matrix shape - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Save vectorizer
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        feature_extractor.save(vectorizer_path)
        print(f"Saved TF-IDF vectorizer to {vectorizer_path}")
    else:
        print("\n[2/5] Feature Extraction - SKIPPED (Evaluation only mode)")
        print("-" * 80)
        
        # Load vectorizer
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        feature_extractor = TfidfFeatureExtractor.load(vectorizer_path)
        print(f"Loaded TF-IDF vectorizer from {vectorizer_path}")
        
        # Transform data
        X_train = feature_extractor.transform(train_data['preprocessed_text'])
        X_test = feature_extractor.transform(test_data['preprocessed_text'])
        y_train = train_data['sentiment']
        y_test = test_data['sentiment']
    
    # Step 3: Model Training
    if not evaluate_only:
        print("\n[3/5] Model Training")
        print("-" * 80)
        
        # Create and train model
        print(f"Training {model_type} model...")
        start_time = time.time()
        
        model = SentimentClassifier(classifier_type=model_type)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save model
        model_path = os.path.join(models_dir, f'sentiment_classifier_{model_type}.joblib')
        model.save(model_path)
        print(f"Saved model to {model_path}")
    else:
        print("\n[3/5] Model Training - SKIPPED (Evaluation only mode)")
        print("-" * 80)
        
        # Load model
        model_path = os.path.join(models_dir, f'sentiment_classifier_{model_type}.joblib')
        model = SentimentClassifier.load(model_path)
        print(f"Loaded model from {model_path}")
    
    # Step 4: Model Evaluation
    print("\n[4/5] Model Evaluation")
    print("-" * 80)
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Step 5: Example Predictions
    if predict_samples:
        print("\n[5/5] Example Predictions")
        print("-" * 80)
        
        # Sample texts for prediction
        sample_texts = [
            "This movie was amazing! I loved every minute of it. Best film I've seen all year.",
            "The acting was terrible and the plot made no sense. Worst movie ever.",
            "It was okay, not great but not terrible either. Just average.",
            "I've never been so bored in my life. Complete waste of time and money.",
            "The cinematography was beautiful and the performances were outstanding. Highly recommend."
        ]
        
        print("Predicting sentiment for sample texts:")
        for text in sample_texts:
            predict_sentiment(text, model, feature_extractor.vectorizer)
            print("-" * 40)
    else:
        print("\n[5/5] Example Predictions - SKIPPED")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

def main():
    """
    Parse command line arguments and run the pipeline.
    """
    parser = argparse.ArgumentParser(description='Run the sentiment analysis pipeline.')
    parser.add_argument('--dataset', type=str, choices=['1', '2', '3', 'auto'], default='auto',
                        help='Which dataset to use (1, 2, 3, or auto)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use (for faster testing)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--model', type=str, 
                        choices=['logistic_regression', 'naive_bayes', 'svm', 'random_forest'],
                        default='logistic_regression',
                        help='Type of model to train')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Skip training and only evaluate existing model')
    parser.add_argument('--no-predictions', action='store_true',
                        help='Skip example predictions')
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            dataset_choice=args.dataset,
            max_samples=args.max_samples,
            test_size=args.test_size,
            model_type=args.model,
            evaluate_only=args.evaluate_only,
            predict_samples=not args.no_predictions
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())