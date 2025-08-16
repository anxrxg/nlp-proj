#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model implementation for sentiment classification.

This script implements a sentiment classification model using TF-IDF features
and various machine learning classifiers.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)


class SentimentClassifier(BaseEstimator, ClassifierMixin):
    """
    Sentiment classification model using TF-IDF features.
    
    This class wraps various scikit-learn classifiers and provides a consistent
    interface for training, prediction, and evaluation.
    """
    
    def __init__(self, classifier_type='logistic_regression', **kwargs):
        """
        Initialize the sentiment classifier.
        
        Args:
            classifier_type: Type of classifier to use ('logistic_regression', 'naive_bayes', 'svm', or 'random_forest')
            **kwargs: Additional arguments to pass to the classifier constructor
        """
        self.classifier_type = classifier_type
        self.kwargs = kwargs
        self.classifier = self._create_classifier()
        self.is_fitted = False
        
    def _create_classifier(self):
        """
        Create a classifier based on the specified type.
        
        Returns:
            A scikit-learn classifier instance
        """
        if self.classifier_type == 'logistic_regression':
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                **self.kwargs
            )
        elif self.classifier_type == 'naive_bayes':
            return MultinomialNB(
                alpha=0.1,
                **self.kwargs
            )
        elif self.classifier_type == 'svm':
            return LinearSVC(
                C=1.0,
                class_weight='balanced',
                random_state=42,
                max_iter=10000,
                **self.kwargs
            )
        elif self.classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                class_weight='balanced',
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
    
    def fit(self, X, y):
        """
        Fit the classifier on the training data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self: The fitted classifier
        """
        self.classifier.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict sentiment labels for the input features.
        
        Args:
            X: Input features
            
        Returns:
            Predicted sentiment labels (0 for negative, 1 for positive)
        """
        if not self.is_fitted:
            raise ValueError("Classifier is not fitted yet. Call 'fit' first.")
        return self.classifier.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input features.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier is not fitted yet. Call 'fit' first.")
        
        # Not all classifiers support predict_proba
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X)
        elif hasattr(self.classifier, 'decision_function'):
            # For SVM, convert decision function to probabilities
            df = self.classifier.decision_function(X)
            if df.ndim == 1:
                df = df.reshape(-1, 1)
                df = np.hstack([(-df), df])
            probs = np.exp(df) / np.sum(np.exp(df), axis=1, keepdims=True)
            return probs
        else:
            raise NotImplementedError(f"Classifier {self.classifier_type} does not support probability prediction")
    
    def evaluate(self, X, y):
        """
        Evaluate the classifier on the test data.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Classifier is not fitted yet. Call 'fit' first.")
        
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        return metrics
    
    def save(self, filepath=None):
        """
        Save the trained classifier to disk.
        
        Args:
            filepath: Path to save the model to. If None, a default path is used.
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            raise ValueError("Classifier is not fitted yet. Call 'fit' first.")
        
        if filepath is None:
            filepath = os.path.join(models_dir, f"sentiment_classifier_{self.classifier_type}.joblib")
        
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained classifier from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded classifier
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


def train_and_evaluate(X_train, y_train, X_test, y_test, classifier_type='logistic_regression', **kwargs):
    """
    Train and evaluate a sentiment classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        classifier_type: Type of classifier to use
        **kwargs: Additional arguments to pass to the classifier constructor
        
    Returns:
        Trained classifier and evaluation metrics
    """
    # Create and train the classifier
    classifier = SentimentClassifier(classifier_type=classifier_type, **kwargs)
    classifier.fit(X_train, y_train)
    
    # Evaluate on training data
    train_metrics = classifier.evaluate(X_train, y_train)
    print(f"Training metrics: {train_metrics}")
    
    # Evaluate on test data
    test_metrics = classifier.evaluate(X_test, y_test)
    print(f"Test metrics: {test_metrics}")
    
    return classifier, test_metrics


if __name__ == "__main__":
    # This code runs when the script is executed directly
    print("This module provides the SentimentClassifier class for sentiment classification.")
    print("Import and use it in your own scripts.")