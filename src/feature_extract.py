#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature extraction module for sentiment classification.

This script implements TF-IDF feature extraction for text data.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')
data_dir = os.path.join(base_dir, 'data')

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

class TfidfFeatureExtractor:
    """
    TF-IDF feature extractor for text data.
    """
    
    def __init__(self, max_features=5000, min_df=5, max_df=0.7, ngram_range=(1, 2)):
        """
        Initialize the TF-IDF feature extractor.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for a term to be included
            max_df: Maximum document frequency for a term to be included
            ngram_range: Range of n-grams to extract (e.g., (1, 2) for unigrams and bigrams)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
        )
        self.is_fitted = False
        
    def fit(self, texts):
        """
        Fit the vectorizer on the training texts.
        
        Args:
            texts: List or Series of text documents
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """
        Transform texts to TF-IDF feature matrix.
        
        Args:
            texts: List or Series of text documents
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer is not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform texts in one step.
        
        Args:
            texts: List or Series of text documents
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """
        Get the feature names (terms) used by the vectorizer.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer is not fitted. Call fit() first.")
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath=None):
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath: Path to save the vectorizer. If None, use default path.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer is not fitted. Call fit() first.")
        
        if filepath is None:
            filepath = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath=None):
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath: Path to the saved vectorizer. If None, use default path.
            
        Returns:
            TfidfFeatureExtractor instance with loaded vectorizer
        """
        if filepath is None:
            filepath = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        
        instance = cls()
        instance.vectorizer = joblib.load(filepath)
        instance.is_fitted = True
        print(f"Vectorizer loaded from {filepath}")
        return instance

def extract_features(train_path=None, test_path=None, save_vectorizer=True):
    """
    Extract TF-IDF features from training and testing data.
    
    Args:
        train_path: Path to the training data CSV. If None, use default path.
        test_path: Path to the testing data CSV. If None, use default path.
        save_vectorizer: Whether to save the fitted vectorizer
        
    Returns:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        extractor: Fitted TfidfFeatureExtractor instance
    """
    # Set default paths if not provided
    if train_path is None:
        train_path = os.path.join(data_dir, 'train.csv')
    if test_path is None:
        test_path = os.path.join(data_dir, 'test.csv')
    
    # Load the data
    print(f"Loading data from {train_path} and {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Initialize and fit the feature extractor
    print("Extracting TF-IDF features...")
    extractor = TfidfFeatureExtractor()
    X_train = extractor.fit_transform(train_df['text'])
    X_test = extractor.transform(test_df['text'])
    
    # Get the labels
    y_train = train_df['sentiment']
    y_test = test_df['sentiment']
    
    # Save the vectorizer if requested
    if save_vectorizer:
        extractor.save()
    
    # Print feature information
    print(f"Feature extraction complete:")
    print(f"  - Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  - Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, y_train, X_test, y_test, extractor

def main():
    """
    Main function to execute the feature extraction pipeline.
    """
    print("Starting feature extraction...")
    
    # Extract features
    X_train, y_train, X_test, y_test, extractor = extract_features()
    
    print("Feature extraction completed successfully!")
    
    # Return the extracted features and labels for potential further use
    return X_train, y_train, X_test, y_test, extractor

if __name__ == "__main__":
    main()