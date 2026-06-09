# Project Notes - House Price Prediction

## Summary
This project builds a housing price prediction system using Scikit-learn and exposes a FastAPI REST API for real-time predictions. The model and preprocessing pipeline are serialized with `joblib`. Deployment is configured for Render.

## What Was Implemented
- Data preprocessing pipeline with `ColumnTransformer` and `Pipeline`.
- Stratified train/test split on income categories.
- Model selection via RMSE cross-validation across:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Best-performing model is trained on full training data and saved.
- Model + preprocessing pipeline serialized to:
  - `model.pkl`
  - `pipeline.pkl`
- Batch inference from `input.csv` to `output.csv`.
- FastAPI service exposing:
  - `GET /health`
  - `POST /predict`
- Render deployment files:
  - `render.yaml`
  - `Procfile`
  - `requirements.txt`

## Files Added / Updated
- `Main.py`
  - Relative paths for data/model files.
  - Cross-validation model selection.
  - Training + inference logic.
- `app.py`
  - FastAPI app loads `model.pkl` and `pipeline.pkl`.
  - `/predict` endpoint accepts a list of records and returns predictions.
- `requirements.txt`
  - API + ML dependencies.
- `render.yaml`
  - Build: `pip install -r requirements.txt && python Main.py`
  - Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- `Procfile`
  - For Render/Heroku process definition.
- `README.md`
  - Updated with API usage and Render deployment notes.

## Local Run Instructions

### 1. Create and activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train + generate model artifacts
```bash
python Main.py
```
Outputs:
- `model.pkl`
- `pipeline.pkl`
- `output.csv`

### 4. Run FastAPI
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Test prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"longitude":-122.23,"latitude":37.88,"housing_median_age":41.0,"total_rooms":880.0,"total_bedrooms":129.0,"population":322.0,"households":126.0,"median_income":8.3252,"ocean_proximity":"NEAR BAY"}]'
```

## Render Deployment Notes

### Problem encountered
Initial Render deployment tried to run:
```
Running 'gunicorn House-Price-Prediction.wsgi'
```
This indicated Render did not use `render.yaml` (defaulted to Django).

### Fix
Use Blueprint or correct service settings.

**Option A (Recommended): Blueprint deployment**
1. Render → New → Blueprint
2. Select repo `Muragharajendra/House-Price-Prediction`
3. Render reads `render.yaml` automatically

**Option B: Manual service settings**
- Build Command:
```
pip install -r requirements.txt && python Main.py
```
- Start Command:
```
uvicorn app:app --host 0.0.0.0 --port $PORT
```
Then redeploy (Clear cache & deploy).

### Verify live API
Once deployed, use the Render URL:
```bash
curl -X POST https://YOUR_RENDER_URL/predict \
  -H "Content-Type: application/json" \
  -d '[{"longitude":-122.23,"latitude":37.88,"housing_median_age":41.0,"total_rooms":880.0,"total_bedrooms":129.0,"population":322.0,"households":126.0,"median_income":8.3252,"ocean_proximity":"NEAR BAY"}]'
```

## Current Status
- Code and configs complete.
- Local training and API work when deps are installed.
- Render deployment needs to be completed/verified using the correct commands above.

