from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# Define the expected input schema using Pydantic
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

        # Predict the probability and apply the threshold
        prob = model.predict_proba(input_df)[:, 1][0]
        prediction = int(prob >= threshold)

        return {
            "prediction": prediction,
            "probability": round(prob, 3),
            "threshold_used": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
