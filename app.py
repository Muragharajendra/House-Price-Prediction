from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Literal
import joblib
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.pkl")
pipeline = joblib.load(BASE_DIR / "pipeline.pkl")


class House_data(BaseModel):
     longitude: float=Field(..., description="Enter longitude")
     latitude:float=Field(..., description="Enter latitude")
     housing_median_age: float=Field(..., ge=0, description="Enter housing median age")
     total_rooms: float=Field(..., ge=0, description="Enter total rooms")
     total_bedrooms: float=Field(..., ge=0, description="Enter total bedrooms")
     population: float=Field(..., ge=0, description="Enter total population")
     households: float=Field(..., ge=0, description="Enter total households")
     median_income: float=Field(..., ge=0, description="Enter median income")
    #  median_house_value: float=Field(..., ge=0, description="Enter median house value")
     ocean_proximity: Literal['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
        
app=FastAPI()
@app.get("/")
def Home_page():
    return {"msg":" Welcome to House price prediction system"}

@app.post("/predict")
def predict(data: House_data):
    pydantic_data=data.model_dump()
    prediction_data=pd.DataFrame([pydantic_data])
    try:
        transformed_data=pipeline.transform(prediction_data)
        predicted_data=model.predict(transformed_data)
        prediction=float(predicted_data[0])       # predicted_data is usually a NumPy array.
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cant process model or pipeline: {e}")
    return JSONResponse(status_code=200, content={"prediction": prediction})
