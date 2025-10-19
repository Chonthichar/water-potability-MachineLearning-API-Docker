# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
from pathlib import Path

app = FastAPI()

# --- Load model and (optionally) the feature list you saved during training ---
MODEL_PATH = Path("model.joblib")
META_PATH = Path("model_meta.json")

model = joblib.load(MODEL_PATH)

# If you stored feature names in model_meta.json, use them; otherwise fall back to the common 9 features
default_features = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

if META_PATH.exists():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta.get("feature_names", default_features)
else:
    feature_names = default_features

# --- Pydantic schema (use lowercase aliases but map to the model’s feature names) ---
class WaterInput(BaseModel):
    ph: float = Field(..., description="Water pH")
    hardness: float = Field(..., alias="Hardness")
    solids: float = Field(..., alias="Solids")
    chloramines: float = Field(..., alias="Chloramines")
    sulfate: float = Field(..., alias="Sulfate")
    conductivity: float = Field(..., alias="Conductivity")
    organic_carbon: float = Field(..., alias="Organic_carbon")
    trihalomethanes: float = Field(..., alias="Trihalomethanes")
    turbidity: float = Field(..., alias="Turbidity")

    class Config:
        populate_by_name = True  # allow either alias or field name in JSON


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(payload: WaterInput):
    try:
        row = {
            "ph": payload.ph,
            "Hardness": payload.hardness,
            "Solids": payload.solids,
            "Chloramines": payload.chloramines,
            "Sulfate": payload.sulfate,
            "Conductivity": payload.conductivity,
            "Organic_carbon": payload.organic_carbon,
            "Trihalomethanes": payload.trihalomethanes,
            "Turbidity": payload.turbidity,
        }

        X = pd.DataFrame([row], columns=feature_names)

        # predicted class (0/1)
        y = int(model.predict(X)[0])

        # probability for class 1 (potable), if available
        p = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

        label_map = {0: "Unsafe (not potable)", 1: "Safe (potable)"}
        return {
            "class": y,
            "label": label_map[y],
            "probability": round(p, 4) if p is not None else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
