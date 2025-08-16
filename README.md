# IMDB Sentiment Classification

This project implements a text sentiment classification model that predicts whether a given text expresses a positive or negative sentiment. It uses TF-IDF features with various machine learning classifiers to analyze IMDB movie reviews for sentiment analysis.

## Project Structure

```
├── data/                      # Directory for datasets
│   ├── IMDB Dataset-2.csv     # Dataset 2 (CSV format)
│   ├── imdb-dataset-1/        # Dataset 1 (Train/Test/Valid CSV files)
│   └── imdb-dataset-3/        # Dataset 3 (Text files in pos/neg folders)
├── models/                    # Directory for saved models
├── src/                       # Source code
│   ├── data_prep.py           # Data preparation utilities
│   ├── feature_extract.py     # TF-IDF feature extraction
│   ├── model.py               # Model implementation
│   ├── train.py               # Original model training script
│   ├── train_model.py         # Enhanced model training script
│   ├── evaluate.py            # Model evaluation script
│   ├── predict.py             # Original prediction interface
│   └── predict_sentiment.py   # Enhanced prediction interface
├── notebooks/                 # Jupyter notebooks for exploration
├── run_pipeline.py            # End-to-end pipeline runner
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Features

- Support for multiple IMDB dataset formats
- Text preprocessing with stopword removal and lemmatization
- TF-IDF feature extraction with n-gram support
- Multiple classification models:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- Model evaluation with accuracy, precision, recall, F1-score
- Command-line interface for predictions
- Interactive prediction mode

## Usage

### Running the Complete Pipeline

To run the complete pipeline (data preparation, feature extraction, model training, evaluation, and example predictions):

```bash
python run_pipeline.py
```

Options:
- `--dataset {1,2,3,auto}`: Choose which dataset to use (default: auto)
- `--max-samples N`: Limit to N samples for faster processing
- `--test-size 0.X`: Set proportion for test split (default: 0.2)
- `--model {logistic_regression,naive_bayes,svm,random_forest}`: Choose model type
- `--evaluate-only`: Skip training and only evaluate existing model
- `--no-predictions`: Skip example predictions

Example:

```bash
python run_pipeline.py --dataset 2 --model svm --max-samples 10000
```

### Making Predictions

To predict sentiment for new text inputs:

```bash
python src/predict_sentiment.py --text "This movie was amazing!"
```

For interactive mode:

```bash
python src/predict_sentiment.py --interactive
```

Options:
- `--text "Your text here"`: Predict sentiment for a single text
- `--file path/to/file.txt`: Predict for texts in a file (one per line)
- `--model path/to/model.joblib`: Use a specific model file
- `--vectorizer path/to/vectorizer.joblib`: Use a specific vectorizer file
- `--interactive`: Run in interactive mode

### Original Scripts

The original scripts are still available:

```bash
python src/data_prep.py   # Data preparation
python src/train.py       # Model training
python src/evaluate.py    # Model evaluation
python src/predict.py     # Prediction interface
```

## Model Details

This project uses a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert text data into numerical features. These features are then used to train a machine learning classifier (e.g., Logistic Regression, SVM, or Naive Bayes) to predict sentiment.

The model is designed to be simple, fast to train, and easy to deploy, making it suitable for various applications requiring sentiment analysis.