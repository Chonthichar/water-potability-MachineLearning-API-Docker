from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, json
import pandas as pd
from pathlib import Path
from typing import Optional

app = FastAPI(title="Water Potability API", version="1.0.0")

MODEL_PATH = Path("best_model/model.joblib")
META_PATH = Path("best_model/model_meta.json")

model = joblib.load(MODEL_PATH)

default_features = [
    "ph","Hardness","Solids","Chloramines","Sulfate",
    "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"
]

if META_PATH.exists():
    try:
        feature_names = json.loads(META_PATH.read_text(encoding="utf-8")).get("feature_names", default_features)
    except Exception:
        feature_names = default_features
else:
    feature_names = default_features

class WaterInput(BaseModel):
    ph: float = Field(...)
    Hardness: float = Field(...)
    Solids: float = Field(...)
    Chloramines: float = Field(...)
    Sulfate: float = Field(...)
    Conductivity: float = Field(...)
    Organic_carbon: float = Field(...)
    Trihalomethanes: float = Field(...)
    Turbidity: float = Field(...)

@app.get("/")
def root():
    return {"message": "Water Potability API is running. See /docs."}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "n_features_expected": len(feature_names), "features": feature_names}

@app.post("/predict")
def predict(x: WaterInput):
    try:
        row = x.model_dump()
        X = pd.DataFrame([row], columns=feature_names)
        y = int(model.predict(X)[0])
        p: Optional[float] = None
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(X)[0][1])
        return {
            "class": y,
            "label": {0: "Unsafe (not potable)", 1: "Safe (potable)"}[y],
            "probability": round(p, 4) if p is not None else None,
            "features_used_in_order": feature_names,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")