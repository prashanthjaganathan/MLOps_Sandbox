import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def train_model(data_path='data/IMDB Dataset.csv', model_path='models/sentiment_model.pkl'):
    """Train sentiment analysis model"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Preprocess
    print("Preprocessing text...")
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    # Convert sentiment to binary (positive=1, negative=0)
    df['sentiment_binary'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], 
        df['sentiment_binary'], 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create pipeline
    print("Training model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Save model
    print(f"Saving model to {model_path}")
    joblib.dump(pipeline, model_path)
    
    # Save test data for evaluation
    test_data = pd.DataFrame({
        'review': X_test,
        'sentiment': y_test
    })
    test_data.to_csv('data/test_data.csv', index=False)
    
    print("Training complete!")
    return pipeline

if __name__ == "__main__":
    train_model()