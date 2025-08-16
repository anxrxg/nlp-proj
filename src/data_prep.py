#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preparation module for sentiment classification.

This script downloads and preprocesses a sentiment dataset for training and evaluation.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create data directory if it doesn't exist
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

def load_imdb_dataset(dataset_choice=1):
    """
    Load IMDB movie review dataset based on the specified choice.
    
    Args:
        dataset_choice: Which dataset to load (1, 2, or 3)
        
    Returns:
        DataFrame with 'text' and 'sentiment' columns
    """
    if dataset_choice == 1:
        # Load from imdb-dataset-1
        train_path = os.path.join(data_dir, 'imdb-dataset-1', 'Train.csv')
        test_path = os.path.join(data_dir, 'imdb-dataset-1', 'Test.csv')
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Rename columns to match our expected format
            train_df = train_df.rename(columns={'text': 'text', 'label': 'sentiment'})
            test_df = test_df.rename(columns={'text': 'text', 'label': 'sentiment'})
            
            # Combine train and test for preprocessing
            df = pd.concat([train_df, test_df], ignore_index=True)
            print(f"Dataset 1 loaded: {len(df)} samples")
            return df
        else:
            print(f"Dataset 1 files not found at {train_path} and {test_path}")
    
    elif dataset_choice == 2:
        # Load from IMDB Dataset-2.csv
        dataset_path = os.path.join(data_dir, 'IMDB Dataset-2.csv')
        
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            
            # Convert sentiment labels to binary (0 for negative, 1 for positive)
            df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
            
            print(f"Dataset 2 loaded: {len(df)} samples")
            return df
        else:
            print(f"Dataset 2 file not found at {dataset_path}")
    
    elif dataset_choice == 3:
        # Load from imdb-dataset-3 (text files in pos/neg directories)
        pos_dir = os.path.join(data_dir, 'imdb-dataset-3', 'train', 'pos')
        neg_dir = os.path.join(data_dir, 'imdb-dataset-3', 'train', 'neg')
        
        if os.path.exists(pos_dir) and os.path.exists(neg_dir):
            # Load positive reviews
            pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.endswith('.txt')]
            pos_reviews = []
            for file_path in pos_files[:12500]:  # Limit to 12500 reviews
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        pos_reviews.append(file.read())
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            
            # Load negative reviews
            neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith('.txt')]
            neg_reviews = []
            for file_path in neg_files[:12500]:  # Limit to 12500 reviews
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        neg_reviews.append(file.read())
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            
            # Create DataFrame
            pos_df = pd.DataFrame({'text': pos_reviews, 'sentiment': 1})
            neg_df = pd.DataFrame({'text': neg_reviews, 'sentiment': 0})
            df = pd.concat([pos_df, neg_df], ignore_index=True)
            
            print(f"Dataset 3 loaded: {len(df)} samples")
            return df
        else:
            print(f"Dataset 3 directories not found at {pos_dir} and {neg_dir}")
    
    # Fallback to the original demo dataset
    print("No valid dataset found. Using demo dataset...")
    return download_movie_reviews()

def create_demo_dataset():
    """
    Create a simple demo dataset with clear positive and negative examples.
    """
    return download_movie_reviews()

def download_movie_reviews():
    """
    Create a simple demo dataset with clear positive and negative examples.
    """
    print("Creating demo dataset...")
    
    # Create a simple dataset with clear positive and negative examples
    positive_texts = [
        "This movie was amazing! I loved every minute of it.",
        "The best film I've seen all year. Absolutely fantastic.",
        "Brilliant performances by all the actors. A must-watch.",
        "I thoroughly enjoyed this film from start to finish.",
        "An outstanding achievement in filmmaking. Loved it!",
        "This is my favorite movie of all time. Perfect in every way.",
        "The director did an incredible job. I was captivated throughout.",
        "A masterpiece of modern cinema. Highly recommended.",
        "I can't stop thinking about how good this movie was.",
        "Five stars! This movie exceeded all my expectations.",
        "The cinematography was beautiful and the performances were outstanding. Highly recommend.",
        "Beautiful cinematography and outstanding performances make this a must-see film.",
        "The visuals were stunning and the acting was superb in this excellent movie.",
        "Great performances by the entire cast. I was impressed by their talent.",
        "The cinematography in this film was breathtaking. A visual masterpiece.",
        "I absolutely loved this movie. The story was engaging and the characters were well-developed.",
        "The story was engaging and the characters were well-developed. I loved it.",
        "Well-developed characters and an engaging storyline made this film a joy to watch.",
        "I didn't expect to like it, but I was pleasantly surprised by how good it was.",
        "The movie had a slow start but ended up being really enjoyable.",
        "Despite some flaws, I really enjoyed this film overall.",
        "The acting was superb and the dialogue was witty and engaging.",
        "A heartwarming story with characters you can't help but root for.",
        "The plot twists kept me on the edge of my seat. Great movie!",
        "I loved how the film balanced humor and drama so effectively."
    ] * 30  # Repeat to get more samples
    
    negative_texts = [
        "This was the worst movie I've ever seen. Terrible acting.",
        "I hated every minute of this film. Complete waste of time.",
        "The plot made no sense and the dialogue was awful.",
        "I walked out of the theater. It was that bad.",
        "Don't waste your money on this garbage movie.",
        "Poorly directed with terrible special effects. Avoid at all costs.",
        "I've never been so bored watching a film. Absolutely dreadful.",
        "The characters were unlikable and the story was predictable.",
        "A complete disaster from start to finish. Zero stars.",
        "I want my money back. This movie was a huge disappointment."
    ] * 50  # Repeat to get more samples
    
    # Create DataFrame
    positive_df = pd.DataFrame({'text': positive_texts, 'sentiment': 1})
    negative_df = pd.DataFrame({'text': negative_texts, 'sentiment': 0})
    df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Demo dataset downloaded: {len(df)} samples")
    return df

def preprocess_data(df):
    """
    Preprocess the text data for sentiment analysis.
    
    Args:
        df: DataFrame with 'text' and 'sentiment' columns
        
    Returns:
        DataFrame with preprocessed text
    """
    print("Preprocessing data...")
    
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Convert to string (in case there are any non-string values)
    df['text'] = df['text'].astype(str)
    
    # Apply preprocessing to each text and store in a new column
    df['preprocessed_text'] = df['text'].apply(preprocess_text)
    
    print("Data preprocessing completed.")
    return df

def preprocess_text(text):
    """
    Preprocess a single text for sentiment analysis.
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def split_and_save_data(df, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets and save them to CSV files.
    
    Args:
        df: DataFrame with 'text' and 'sentiment' columns
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, test_df: Training and testing DataFrames
    """
    print("Splitting data into training and testing sets...")
    
    # Split the data
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['sentiment']
    )
    
    # Save to CSV files
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training data saved to {train_path}")
    print(f"Testing data saved to {test_path}")
    
    return train_df, test_df

def main(dataset_choice=1):
    """
    Main function to prepare the data for sentiment analysis.
    
    Args:
        dataset_choice: Which dataset to use (1, 2, or 3)
        
    Returns:
        train_df, test_df: Training and testing DataFrames
    """
    # Load the dataset
    df = load_imdb_dataset(dataset_choice)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Split and save the data
    train_df, test_df = split_and_save_data(df)
    
    return train_df, test_df

if __name__ == "__main__":
    # This code runs when the script is executed directly
    main()