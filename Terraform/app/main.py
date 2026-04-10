"""Minimal FastAPI app for Cloud Run (listens on PORT, default 8080)."""

import os

from fastapi import FastAPI

app = FastAPI(title="Terraform lab API")


@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI on Cloud Run"}


@app.get("/health")
def health():
    return {"healthy": True}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
