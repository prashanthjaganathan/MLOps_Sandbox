FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create models directory
RUN mkdir -p models

EXPOSE 8000

# Default command runs full pipeline
CMD ["sh", "-c", "python src/train.py && python src/evaluate.py && uvicorn src.api:app --host 0.0.0.0 --port 8000"]