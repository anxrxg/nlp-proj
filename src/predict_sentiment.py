#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentiment prediction script for text classification.

This script provides a command-line interface for predicting sentiment of text inputs.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Import local modules
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep import preprocess_text
from src.feature_extract import TfidfFeatureExtractor
from src.model import SentimentClassifier

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')

def load_model(model_path=None):
    """
    Load a trained sentiment classification model.
    
    Args:
        model_path: Path to the model file. If None, use the default path.
        
    Returns:
        Loaded model
    """
    if model_path is None:
        # Try to find the best model first
        best_model_path = os.path.join(models_dir, 'best_model_logistic_regression.joblib')
        if os.path.exists(best_model_path):
            model_path = best_model_path
        else:
            # Fall back to any available model
            model_files = [f for f in os.listdir(models_dir) if f.startswith('sentiment_classifier_') and f.endswith('.joblib')]
            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
            else:
                raise FileNotFoundError("No trained model found. Please train a model first.")
    
    print(f"Loading model from {model_path}")
    model = SentimentClassifier.load(model_path)
    return model

def load_vectorizer(vectorizer_path=None):
    """
    Load a trained TF-IDF vectorizer.
    
    Args:
        vectorizer_path: Path to the vectorizer file. If None, use the default path.
        
    Returns:
        Loaded vectorizer
    """
    if vectorizer_path is None:
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
    
    print(f"Loading vectorizer from {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer

def predict_sentiment(text, model=None, vectorizer=None, verbose=True):
    """
    Predict the sentiment of a text input.
    
    Args:
        text: Input text string
        model: Trained sentiment classification model. If None, load the default model.
        vectorizer: Trained TF-IDF vectorizer. If None, load the default vectorizer.
        verbose: Whether to print prediction details
        
    Returns:
        Dictionary with prediction results
    """
    # Load model and vectorizer if not provided
    if model is None:
        model = load_model()
    
    if vectorizer is None:
        vectorizer = load_vectorizer()
    
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Transform the text to TF-IDF features
    X = vectorizer.transform([preprocessed_text])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get probability if available
    probability = None
    try:
        probability = model.predict_proba(X)[0][prediction]
    except:
        pass
    
    # Create result dictionary
    result = {
        'text': text,
        'preprocessed_text': preprocessed_text,
        'sentiment': 'Positive' if prediction == 1 else 'Negative',
        'confidence': probability
    }
    
    # Print prediction details if verbose
    if verbose:
        print(f"\nText: {text}")
        print(f"Preprocessed text: {preprocessed_text}")
        print(f"Predicted sentiment: {result['sentiment']}")
        if result['confidence'] is not None:
            print(f"Confidence: {result['confidence']:.4f}")
    
    return result

def predict_from_file(file_path, model=None, vectorizer=None):
    """
    Predict sentiment for texts in a file.
    
    Args:
        file_path: Path to the file containing texts (one per line)
        model: Trained sentiment classification model. If None, load the default model.
        vectorizer: Trained TF-IDF vectorizer. If None, load the default vectorizer.
        
    Returns:
        List of prediction results
    """
    # Load model and vectorizer if not provided
    if model is None:
        model = load_model()
    
    if vectorizer is None:
        vectorizer = load_vectorizer()
    
    # Read texts from file
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # Make predictions
    results = []
    for text in texts:
        result = predict_sentiment(text, model, vectorizer, verbose=True)
        results.append(result)
        print('-' * 80)
    
    return results

def interactive_mode(model=None, vectorizer=None):
    """
    Run the sentiment prediction in interactive mode.
    
    Args:
        model: Trained sentiment classification model. If None, load the default model.
        vectorizer: Trained TF-IDF vectorizer. If None, load the default vectorizer.
    """
    # Load model and vectorizer if not provided
    if model is None:
        model = load_model()
    
    if vectorizer is None:
        vectorizer = load_vectorizer()
    
    print("\nSentiment Prediction Interactive Mode")
    print("Enter text to predict sentiment. Type 'exit' to quit.")
    
    while True:
        print("\n" + "-" * 80)
        text = input("Enter text: ")
        
        if text.lower() in ['exit', 'quit', 'q']:
            break
        
        if not text.strip():
            continue
        
        predict_sentiment(text, model, vectorizer, verbose=True)

def main():
    """
    Main function to parse arguments and run the prediction.
    """
    parser = argparse.ArgumentParser(description='Predict sentiment of text inputs.')
    parser.add_argument('--text', type=str, help='Text to predict sentiment for')
    parser.add_argument('--file', type=str, help='Path to file containing texts (one per line)')
    parser.add_argument('--model', type=str, help='Path to trained model file')
    parser.add_argument('--vectorizer', type=str, help='Path to trained vectorizer file')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Load model and vectorizer
        model = load_model(args.model)
        vectorizer = load_vectorizer(args.vectorizer)
        
        if args.text:
            # Predict sentiment for a single text
            predict_sentiment(args.text, model, vectorizer)
        elif args.file:
            # Predict sentiment for texts in a file
            predict_from_file(args.file, model, vectorizer)
        elif args.interactive:
            # Run in interactive mode
            interactive_mode(model, vectorizer)
        else:
            # Default to interactive mode
            interactive_mode(model, vectorizer)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())