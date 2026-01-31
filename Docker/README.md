# Lab 1 (Docker): IMDB Sentiment Analysis ML Pipeline

A simple machine learning pipeline that trains a sentiment analysis model on the IMDB movie reviews dataset and serves predictions via FastAPI.

## What This App Does

1. **Trains** a sentiment classifier on 50k IMDB movie reviews
2. **Evaluates** the model performance (accuracy, precision, recall, F1-score)
3. **Serves** predictions through a REST API endpoint

## Folder Structure

```
Docker/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── data/                   # Dataset folder
│   └── IMDB Dataset.csv
├── models/                 # Trained models saved here
├── src/                    # Source code
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── api.py             # FastAPI application
└── README.md              # This file
```

## Prerequisites

- Docker installed on your system
- Docker Compose (usually comes with Docker Desktop)

## Setup & Run

### Option 1: Using docker-compose (Recommended)

```bash
# Navigate to Docker folder
cd Docker

# Build and run the complete pipeline
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Option 2: Using Dockerfile

```bash
# Navigate to Docker folder
cd Docker

# Build the image
docker build -t sentiment-app .

# Run the container
docker run -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  sentiment-app
```

## Access the API

Once running, access the API at:
- API Base: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/

## Test Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely fantastic!"}'
```

## What Happens When You Run

1. Trains sentiment model on IMDB dataset
2. Evaluates model and prints metrics
3. Starts FastAPI server on port 8000
4. Ready to accept prediction requests
