#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction interface for sentiment classification.

This script provides an interface for making sentiment predictions on new text.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Import local modules
from feature_extract import TfidfFeatureExtractor

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

def preprocess_text(text):
    """
    Preprocess a single text input for prediction.
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (keeping letters, numbers, and spaces)
    text = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in text])
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def predict_sentiment(text, model_type='logistic_regression', verbose=True):
    """
    Predict the sentiment of a text input.
    
    Args:
        text: Input text string
        model_type: Type of model to use ('logistic_regression', 'naive_bayes', 'svm', or 'random_forest')
        verbose: Whether to print prediction details
        
    Returns:
        Dictionary containing prediction results
    """
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Load the model
    model = load_model(model_type=model_type)
    
    # Load the vectorizer
    vectorizer = TfidfFeatureExtractor.load()
    
    # Transform the text to TF-IDF features
    X = vectorizer.transform([preprocessed_text])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        try:
            probability = model.predict_proba(X)[0][prediction]
        except:
            pass
    
    # Create result dictionary
    result = {
        'text': text,
        'preprocessed_text': preprocessed_text,
        'sentiment': 'Positive' if prediction == 1 else 'Negative',
        'sentiment_score': float(prediction),
        'confidence': float(probability) if probability is not None else None
    }
    
    # Print prediction details if verbose
    if verbose:
        print(f"\nInput text: {text}")
        print(f"Preprocessed text: {preprocessed_text}")
        print(f"Predicted sentiment: {result['sentiment']}")
        if result['confidence'] is not None:
            print(f"Confidence: {result['confidence']:.4f}")
    
    return result

def main():
    """
    Main function to execute the prediction interface.
    """
    # Check if a text argument is provided
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your text here\"")
        print("\nInteractive mode:")
        
        # Enter interactive mode
        while True:
            # Get user input
            text = input("\nEnter text to analyze (or 'q' to quit): ")
            
            # Check if the user wants to quit
            if text.lower() == 'q':
                break
            
            # Skip empty input
            if not text.strip():
                continue
            
            # Predict sentiment
            predict_sentiment(text)
    else:
        # Use the provided text argument
        text = ' '.join(sys.argv[1:])
        predict_sentiment(text)

if __name__ == "__main__":
    main()