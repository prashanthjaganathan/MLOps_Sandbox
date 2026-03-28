"""FastAPI application for Iris SVM training and prediction."""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from .pipeline import MODEL_PATH, TARGET_NAMES, load_pipeline, predict, train_pipeline

app = FastAPI(title="Iris SVM API", version="1.0.0")

_model: Optional[Pipeline] = None


class TrainRequest(BaseModel):
    test_size: float = Field(0.2, gt=0, lt=1)
    random_state: int = 42
    C: float = Field(1.0, gt=0)
    kernel: str = "rbf"


class TrainResponse(BaseModel):
    message: str
    accuracy: float
    train_size: int
    test_size: int


class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[sepal_length, sepal_width, petal_length, petal_width]",
    )


class PredictResponse(BaseModel):
    species_id: int
    species: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest = TrainRequest()):
    global _model
    pipe, result = train_pipeline(
        test_size=req.test_size,
        random_state=req.random_state,
        C=req.C,
        kernel=req.kernel,
    )
    _model = pipe
    return TrainResponse(
        message="Model trained and persisted",
        accuracy=result.accuracy,
        train_size=result.train_size,
        test_size=result.test_size,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = load_pipeline()
        else:
            raise HTTPException(status_code=503, detail="Model not trained yet. Call /train first.")
    species_id, species = predict(_model, req.features)
    return PredictResponse(species_id=species_id, species=species)


@app.get("/species")
def list_species():
    return {"species": TARGET_NAMES}
