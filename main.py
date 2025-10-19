from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
import joblib
import json
import pandas as pd
from pathlib import Path
from typing import Optional

app = FastAPI(title="Water Potability Classifier", version="1.0.0")

# --- Paths ---
MODEL_PATH = Path("model.joblib")
META_PATH = Path("model_meta.json")

# --- Load model ---
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH.resolve()}")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# --- Feature names (from training meta) ---
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
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names", default_features)
    except Exception:
        feature_names = default_features
else:
    feature_names = default_features

# --- Pydantic schema ---
# Accept both lower-case and training-time column names
class WaterInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ph: float = Field(..., validation_alias=AliasChoices("ph", "Ph", "PH"))
    hardness: float = Field(..., alias="Hardness",
                            validation_alias=AliasChoices("hardness", "Hardness"))
    solids: float = Field(..., alias="Solids",
                          validation_alias=AliasChoices("solids", "Solids"))
    chloramines: float = Field(..., alias="Chloramines",
                               validation_alias=AliasChoices("chloramines", "Chloramines"))
    sulfate: float = Field(..., alias="Sulfate",
                           validation_alias=AliasChoices("sulfate", "Sulfate"))
    conductivity: float = Field(..., alias="Conductivity",
                                validation_alias=AliasChoices("conductivity", "Conductivity"))
    organic_carbon: float = Field(..., alias="Organic_carbon",
                                  validation_alias=AliasChoices("organic_carbon", "Organic_carbon"))
    trihalomethanes: float = Field(..., alias="Trihalomethanes",
                                   validation_alias=AliasChoices("trihalomethanes", "Trihalomethanes"))
    turbidity: float = Field(..., alias="Turbidity",
                             validation_alias=AliasChoices("turbidity", "Turbidity"))

@app.get("/")
async def root():
    return {"message": "Water Potability API is running. See /docs for Swagger UI."}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features_expected": len(feature_names),
        "features": feature_names,
    }

@app.post("/predict")
def predict(payload: WaterInput):
    try:
        # Build a row with the exact training feature names (order matters)
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

        # Ensure column order matches what the model expects
        X = pd.DataFrame([row], columns=feature_names)

        # Predicted class
        y = int(model.predict(X)[0])

        # Probability for class 1 (if available)
        p: Optional[float] = None
        if hasattr(model, "predict_proba"):
            try:
                p = float(model.predict_proba(X)[0][1])
            except Exception:
                p = None

        label_map = {0: "Unsafe (not potable)", 1: "Safe (potable)"}
        return {
            "class": y,
            "label": label_map.get(y, str(y)),
            "probability": round(p, 4) if p is not None else None,
            "features_used_in_order": feature_names,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}