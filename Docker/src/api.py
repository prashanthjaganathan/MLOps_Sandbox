from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
from typing import Dict

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Load model at startup
try:
    model = joblib.load('models/sentiment_model.pkl')
except FileNotFoundError:
    model = None
    print("Warning: Model not found. Please train the model first.")

def preprocess_text(text: str) -> str:
    """Preprocess text for prediction"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

class ReviewRequest(BaseModel):
    review: str

class PredictionResponse(BaseModel):
    review: str
    sentiment: str
    confidence: float

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "Sentiment Analysis API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest) -> Dict:
    """Predict sentiment of a movie review"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # Preprocess
    cleaned_review = preprocess_text(request.review)
    
    # Predict
    prediction = model.predict([cleaned_review])[0]
    probability = model.predict_proba([cleaned_review])[0]
    
    sentiment = "positive" if prediction == 1 else "negative"
    confidence = float(probability[prediction])
    
    return {
        "review": request.review,
        "sentiment": sentiment,
        "confidence": confidence
    }

@app.post("/batch-predict")
def batch_predict(reviews: list[ReviewRequest]):
    """Predict sentiment for multiple reviews"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    results = []
    for review_req in reviews:
        cleaned_review = preprocess_text(review_req.review)
        prediction = model.predict([cleaned_review])[0]
        probability = model.predict_proba([cleaned_review])[0]
        
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(probability[prediction])
        
        results.append({
            "review": review_req.review,
            "sentiment": sentiment,
            "confidence": confidence
        })
    
    return {"predictions": results}