from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import os

# Load model bundle on startup
MODEL_PATH = os.path.join("saved_models", "final_model_bundle.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
        model = bundle["model"]
        threshold = bundle["threshold"]
        features = bundle["features"]
except Exception as e:
    raise RuntimeError(f"Failed to load model bundle: {e}")

# Define the expected input schema
class PassengerData(BaseModel):
    Pclass: int
    Sex: int  # 0 for male, 1 for female
    Age: float
    Fare: float

app = FastAPI(title="Titanic Survival Predictor")

@app.get("/")
def read_root():
    return {"message": "Titanic Survival Prediction API is running."}

@app.post("/predict")
def predict(data: PassengerData):
    try:
        input_df = pd.DataFrame([data.dict()], columns=features)
        prob = model.predict_proba(input_df)[:, 1][0]
        prediction = int(prob >= threshold)

        return {
            "prediction": prediction,
            "probability": round(prob, 3),
            "threshold_used": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict-batch")
def predict_batch(data: List[PassengerData]):
    try:
        input_df = pd.DataFrame([d.dict() for d in data], columns=features)
        probs = model.predict_proba(input_df)[:, 1]
        predictions = (probs >= threshold).astype(int)

        results = []
        for i in range(len(data)):
            results.append({
                "input": data[i].dict(),
                "prediction": int(predictions[i]),
                "probability": round(probs[i], 3)
            })

        return {
            "results": results,
            "threshold_used": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")
