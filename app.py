import os
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
PIPELINE_FILE = os.path.join(BASE_DIR, "pipeline.pkl")

app = FastAPI(title="House Price Prediction API")


def load_artifacts():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
        raise FileNotFoundError("model.pkl or pipeline.pkl not found. Train the model first.")
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(records: List[Dict[str, Any]]):
    try:
        model, pipeline = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not records:
        raise HTTPException(status_code=400, detail="Input list is empty.")

    df = pd.DataFrame(records)
    try:
        transformed = pipeline.transform(df)
        preds = model.predict(transformed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return {"predictions": preds.tolist()}
