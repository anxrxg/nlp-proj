#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model training module for sentiment classification.

This script trains a machine learning classifier on TF-IDF features for sentiment prediction.
It provides a command-line interface to select the dataset, model type, and other parameters.
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import local modules
from data_prep import load_imdb_dataset, preprocess_data, create_demo_dataset, preprocess_text
from feature_extract import TfidfFeatureExtractor

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')
data_dir = os.path.join(base_dir, 'data')

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

def train_model(X_train, y_train, model_type='logistic_regression', **kwargs):
    """
    Train a machine learning model for sentiment classification.
    
    Args:
         X_train: Training features
         y_train: Training labels
         model_type: Type of model to train ('logistic_regression', 'naive_bayes', 'svm', or 'random_forest')
         **kwargs: Additional arguments to pass to the model constructor
         
     Returns:
         Trained model
     """
    print(f"Training {model_type} model...")
    start_time = time.time()
    
    # Initialize the model based on the specified type
    if model_type == 'logistic_regression':
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            **kwargs
        )
    elif model_type == 'naive_bayes':
        model = MultinomialNB(alpha=0.1, **kwargs)
    elif model_type == 'svm':
        model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            **kwargs
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            class_weight='balanced',
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Model trained in {training_time:.2f} seconds")
    
    return model

def evaluate_on_training(model, X_train, y_train):
    """
    Evaluate the model on the training data.
    
    
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Training accuracy
    """
    # Make predictions on the training data
    y_train_pred = model.predict(X_train)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {accuracy:.4f}")
    
    return accuracy


def train_model_from_dataset(dataset_choice='auto', max_samples=None, test_size=0.2, model_type='logistic_regression'):
    """
    Train a sentiment classification model using the specified dataset and model type.
    
    Args:
        dataset_choice: Which dataset to use ('1', '2', '3', 'auto', or 'demo')
        max_samples: Maximum number of samples to use (for faster testing)
        test_size: Proportion of data to use for testing
        model_type: Type of model to train ('logistic_regression', 'naive_bayes', 'svm', 'random_forest')
        
    Returns:
        Trained model, vectorizer, and evaluation metrics
    """
    print(f"\n{'='*80}\nTraining sentiment classification model\n{'='*80}")
    print(f"Dataset: {dataset_choice}")
    print(f"Model type: {model_type}")
    print(f"Max samples: {max_samples}")
    print(f"Test size: {test_size}\n")
    
    start_time = time.time()
    
    # Load and preprocess the dataset
    if dataset_choice == 'auto':
        # Try to load datasets in order of preference
        for choice in [3, 2, 1, 'demo']:
            if choice == 'demo':
                print("No datasets found, creating demo dataset...")
                df = create_demo_dataset()
                break
            else:
                df = load_imdb_dataset(choice)
                if df is not None and len(df) > 0:
                    print(f"Using dataset {choice}")
                    break
    elif dataset_choice == 'demo':
        df = create_demo_dataset()
    else:
        try:
            choice = int(dataset_choice)
            df = load_imdb_dataset(choice)
        except ValueError:
            print(f"Invalid dataset choice: {dataset_choice}. Using demo dataset.")
            df = create_demo_dataset()
    
    if df is None or len(df) == 0:
        print("Failed to load dataset. Creating demo dataset...")
        df = create_demo_dataset()
    
    # Limit the number of samples if specified
    if max_samples is not None and max_samples < len(df):
        df = df.sample(max_samples, random_state=42)
    
    # Preprocess the data
    df = df.copy()
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x) if hasattr(preprocess_data, '__call__') else x)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['sentiment'], test_size=test_size, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Extract features using TF-IDF
    print("\nExtracting features...")
    feature_extractor = TfidfFeatureExtractor(max_features=5000)
    X_train_features = feature_extractor.fit_transform(X_train)
    X_test_features = feature_extractor.transform(X_test)
    
    # Train the model
    print("\nTraining model...")
    model = train_model(X_train_features, y_train, model_type=model_type)
    
    # Evaluate the model on training data
    train_accuracy = evaluate_on_training(model, X_train_features, y_train)
    
    # Evaluate the model on test data
    print("\nEvaluating model on test data...")
    y_pred = model.predict(X_test_features)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save the model and vectorizer
    model_path = os.path.join(models_dir, f'sentiment_classifier_{model_type}.joblib')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(feature_extractor.vectorizer, vectorizer_path)
    
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal training time: {elapsed_time:.2f} seconds")
    
    return model, feature_extractor, test_accuracy


def main():
    """
    Main function to parse command-line arguments and train the model.
    """
    parser = argparse.ArgumentParser(description='Train a sentiment classification model')
    parser.add_argument('--dataset', type=str, default='auto',
                        help='Dataset to use (1, 2, 3, auto, or demo)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--model', type=str, default='logistic_regression',
                        choices=['logistic_regression', 'naive_bayes', 'svm', 'random_forest'],
                        help='Type of model to train')
    
    args = parser.parse_args()
    
    train_model_from_dataset(
        dataset_choice=args.dataset,
        max_samples=args.max_samples,
        test_size=args.test_size,
        model_type=args.model
    )


if __name__ == '__main__':
    main()