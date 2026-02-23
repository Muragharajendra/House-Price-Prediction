import os
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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


@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>House Price Prediction</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 32px; max-width: 820px; }
      h1 { margin-bottom: 8px; }
      .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
      label { display: block; font-size: 14px; margin-bottom: 4px; }
      input, select { width: 100%; padding: 8px; }
      button { margin-top: 16px; padding: 10px 16px; }
      pre { background: #f6f8fa; padding: 12px; }
    </style>
  </head>
  <body>
    <h1>House Price Prediction</h1>
    <p>Fill the form and click Predict to see the output.</p>
    <div class="grid">
      <div><label>Longitude</label><input id="longitude" value="-122.23" /></div>
      <div><label>Latitude</label><input id="latitude" value="37.88" /></div>
      <div><label>Housing Median Age</label><input id="housing_median_age" value="41.0" /></div>
      <div><label>Total Rooms</label><input id="total_rooms" value="880.0" /></div>
      <div><label>Total Bedrooms</label><input id="total_bedrooms" value="129.0" /></div>
      <div><label>Population</label><input id="population" value="322.0" /></div>
      <div><label>Households</label><input id="households" value="126.0" /></div>
      <div><label>Median Income</label><input id="median_income" value="8.3252" /></div>
      <div>
        <label>Ocean Proximity</label>
        <select id="ocean_proximity">
          <option>NEAR BAY</option>
          <option>INLAND</option>
          <option>NEAR OCEAN</option>
          <option>ISLAND</option>
          <option>&lt;1H OCEAN</option>
        </select>
      </div>
    </div>
    <button onclick="predict()">Predict</button>
    <h3>Result</h3>
    <pre id="result">No prediction yet.</pre>
    <script>
      async function predict() {
        const payload = [{
          longitude: parseFloat(document.getElementById('longitude').value),
          latitude: parseFloat(document.getElementById('latitude').value),
          housing_median_age: parseFloat(document.getElementById('housing_median_age').value),
          total_rooms: parseFloat(document.getElementById('total_rooms').value),
          total_bedrooms: parseFloat(document.getElementById('total_bedrooms').value),
          population: parseFloat(document.getElementById('population').value),
          households: parseFloat(document.getElementById('households').value),
          median_income: parseFloat(document.getElementById('median_income').value),
          ocean_proximity: document.getElementById('ocean_proximity').value
        }];
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        document.getElementById('result').textContent = JSON.stringify(data, null, 2);
      }
    </script>
  </body>
</html>
"""


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
