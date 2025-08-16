#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model training script for sentiment classification.

This script trains a sentiment classification model using the IMDB dataset.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import local modules
from data_prep import main as prepare_data
from feature_extract import TfidfFeatureExtractor, extract_features
from model import SentimentClassifier

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')
data_dir = os.path.join(base_dir, 'data')

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)

def train_and_evaluate_model(dataset_choice=2, classifier_type='logistic_regression', max_features=5000):
    """
    Train and evaluate a sentiment classification model.
    
    Args:
        dataset_choice: Which dataset to use (1, 2, or 3)
        classifier_type: Type of classifier to use ('logistic_regression', 'naive_bayes', 'svm', or 'random_forest')
        max_features: Maximum number of features to extract
        
    Returns:
        Trained classifier and evaluation metrics
    """
    print(f"\n{'='*80}\nTraining sentiment classification model\n{'='*80}")
    print(f"Dataset: {dataset_choice}")
    print(f"Classifier: {classifier_type}")
    print(f"Max features: {max_features}\n")
    
    # Prepare the data
    train_df, test_df = prepare_data(dataset_choice)
    
    # Extract features
    print("\nExtracting TF-IDF features...")
    extractor = TfidfFeatureExtractor(max_features=max_features, min_df=5, max_df=0.7, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = extract_features(train_df, test_df, extractor)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    
    # Train the model
    print(f"\nTraining {classifier_type} model...")
    classifier = SentimentClassifier(classifier_type=classifier_type)
    classifier.fit(X_train, y_train)
    
    # Evaluate on training data
    train_metrics = classifier.evaluate(X_train, y_train)
    print(f"\nTraining metrics: {train_metrics}")
    
    # Evaluate on test data
    test_metrics = classifier.evaluate(X_test, y_test)
    print(f"\nTest metrics: {test_metrics}")
    
    # Generate classification report
    y_pred = classifier.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {classifier_type}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(base_dir, 'results', f'confusion_matrix_{classifier_type}.png')
    plt.savefig(plot_path)
    print(f"\nConfusion matrix saved to {plot_path}")
    
    # Save the model
    model_path = classifier.save()
    print(f"Model saved to {model_path}")
    
    # Save the feature extractor
    extractor_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    extractor.save(extractor_path)
    print(f"Feature extractor saved to {extractor_path}")
    
    return classifier, test_metrics

def compare_models(dataset_choice=2, max_features=5000):
    """
    Train and compare different classifier models.
    
    Args:
        dataset_choice: Which dataset to use (1, 2, or 3)
        max_features: Maximum number of features to extract
        
    Returns:
        Dictionary of trained classifiers and their metrics
    """
    print(f"\n{'='*80}\nComparing different classifier models\n{'='*80}")
    
    # Prepare the data
    train_df, test_df = prepare_data(dataset_choice)
    
    # Extract features
    print("\nExtracting TF-IDF features...")
    extractor = TfidfFeatureExtractor(max_features=max_features, min_df=5, max_df=0.7, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = extract_features(train_df, test_df, extractor)
    
    # Define the models to train
    model_types = ['logistic_regression', 'naive_bayes', 'svm', 'random_forest']
    
    # Train and evaluate each model
    models = {}
    metrics = {}
    
    for model_type in model_types:
        print(f"\n{'-'*50}\nTraining {model_type} model...")
        
        # Train the model
        classifier = SentimentClassifier(classifier_type=model_type)
        classifier.fit(X_train, y_train)
        models[model_type] = classifier
        
        # Evaluate on test data
        test_metrics = classifier.evaluate(X_test, y_test)
        metrics[model_type] = test_metrics
        
        print(f"Test metrics: {test_metrics}")
    
    # Create a DataFrame for comparison
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'Model'
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    metrics_df_melted = pd.melt(metrics_df.reset_index(), id_vars=['Model'], var_name='Metric', value_name='Score')
    
    sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df_melted, palette='viridis')
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(base_dir, 'results', 'model_comparison.png')
    plt.savefig(plot_path)
    print(f"\nModel comparison plot saved to {plot_path}")
    
    # Save the best model
    best_model_type = metrics_df['accuracy'].idxmax()
    best_model = models[best_model_type]
    model_path = best_model.save(os.path.join(models_dir, f"best_model_{best_model_type}.joblib"))
    
    print(f"\nBest model: {best_model_type} with accuracy {metrics_df.loc[best_model_type, 'accuracy']:.4f}")
    print(f"Best model saved to {model_path}")
    
    # Save the feature extractor
    extractor_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    extractor.save(extractor_path)
    print(f"Feature extractor saved to {extractor_path}")
    
    return models, metrics

def main():
    """
    Main function to parse arguments and train the model.
    """
    parser = argparse.ArgumentParser(description='Train a sentiment classification model.')
    parser.add_argument('--dataset', type=int, default=2, choices=[1, 2, 3],
                        help='Which dataset to use (1, 2, or 3)')
    parser.add_argument('--classifier', type=str, default='logistic_regression',
                        choices=['logistic_regression', 'naive_bayes', 'svm', 'random_forest'],
                        help='Type of classifier to use')
    parser.add_argument('--max-features', type=int, default=5000,
                        help='Maximum number of features to extract')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different classifier models')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.dataset, args.max_features)
    else:
        train_and_evaluate_model(args.dataset, args.classifier, args.max_features)

if __name__ == "__main__":
    main()