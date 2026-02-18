from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Random Forest Predictive Maintenance API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = "random_forest_model/rf_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

class PredictionInput(BaseModel):
    Type: str  # L, M, H
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float

@app.get("/")
def read_root():
    return {"message": "Random Forest Predictive Maintenance API is running"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        data = {
            'Type': [input_data.Type],
            'Air_temperature_K': [input_data.Air_temperature_K],
            'Process_temperature_K': [input_data.Process_temperature_K],
            'Rotational_speed_rpm': [input_data.Rotational_speed_rpm],
            'Torque_Nm': [input_data.Torque_Nm],
            'Tool_wear_min': [input_data.Tool_wear_min]
        }
        df = pd.DataFrame(data)

        # Feature Engineering (matching test_models.ipynb)
        df['Temperature_Difference'] = df['Process_temperature_K'] - df['Air_temperature_K']
        df['Power'] = df['Rotational_speed_rpm'] * df['Torque_Nm']

        # Encode Type
        type_map = {'L': 0, 'M': 1, 'H': 2}
        if input_data.Type not in type_map:
            raise HTTPException(status_code=400, detail="Invalid Type. Must be L, M, or H.")
        
        df['Type'] = df['Type'].map(type_map)

        # Ensure column order matches training
        # Features used in training (based on test_models.ipynb drop logic):
        # Type, Air_temperature_K, Process_temperature_K, Rotational_speed_rpm, Torque_Nm, Tool_wear_min, Temperature_Difference, Power
        cols = ['Type', 'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', 'Temperature_Difference', 'Power']
        df = df[cols]

        # Predict
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        return {
            "prediction": int(prediction[0]),
            "failure_probability": float(prediction_proba[0][1]),
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
