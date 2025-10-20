# main.py (root)
# ===========================================
# What this one do?
# We expose a REST API with FastAPI that loads our trained artifact
# (ideally the *full pipeline* we saved in train.py), receives a single JSON
# sample, and returns a water potability prediction (+ optional probability).

# Why a *pipeline*? Because it already contains preprocessing, SMOTE and model,
# so we can safely accept missing values (the imputers handle them) and keep
# the feature order consistent. If we only had a raw model, we'd need to do
# all preprocessing by hand before calling .predict().
# ===========================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
import os, json, joblib
import pandas as pd

app = FastAPI(title="Water Potability API", version="2.3.0")

# -------------------------------------------------
# 1) Paths: we resolve everything relative to this file
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BM_DIR = PROJECT_ROOT / "best_model"

PIPELINE_PATH = BM_DIR / "pipeline.joblib"
MODEL_PATH    = BM_DIR / "pipeline.joblib"
META_PATH     = BM_DIR / "model_meta.json"

# allow override via env
env_path = os.getenv("MODEL_PATH")
artifact_path = Path(env_path) if env_path else (PIPELINE_PATH if PIPELINE_PATH.exists() else MODEL_PATH)

if not artifact_path.exists():
    raise RuntimeError(
        f"No model found. Tried:\n"
        f" - {PIPELINE_PATH}\n"
        f" - {MODEL_PATH}\n"
        f"Or set MODEL_PATH env var to a valid file."
    )

# -------------------------------------------------
# 2) Keep feature order stable
# -------------------------------------------------
# Why does order matter? Pandas DataFrames index columns by name, but models
# are trained in a fixed order. We load from meta if available to avoid
# accidental column swaps breaking predictions.
default_features = [
    "ph","Hardness","Solids","Chloramines","Sulfate",
    "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"
]

if META_PATH.exists():
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        feature_names = meta.get("feature_names_in_order") or meta.get("feature_names") or default_features
    except Exception:
        feature_names = default_features
else:
    feature_names = default_features

# -------------------------------------------------
# 3) Load artifact and detect if it's a *pipeline*
# -------------------------------------------------
# If it's a pipeline, we can accept Nones (imputers handle them).
# If it's *not* a pipeline (raw model), all fields must be present.
model = joblib.load(artifact_path)

# A simple heuristic: imblearn/sklearn pipelines expose .named_steps (a dict)
IS_PIPELINE = hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict)


# -------------------------------------------------
# 4) Request schema
# -------------------------------------------------
# We make every field Optional so the same schema works for both cases:
# - PIPELINE: Optional is OK (imputation inside)
# - RAW MODEL: We'll actively check and reject missing fields
class WaterInput(BaseModel):
    ph: Optional[float] = Field(None)
    Hardness: Optional[float] = Field(None)
    Solids: Optional[float] = Field(None)
    Chloramines: Optional[float] = Field(None)
    Sulfate: Optional[float] = Field(None)
    Conductivity: Optional[float] = Field(None)
    Organic_carbon: Optional[float] = Field(None)
    Trihalomethanes: Optional[float] = Field(None)
    Turbidity: Optional[float] = Field(None)

# -------------------------------------------------
# 5) Endpoints for quick checks
# -------------------------------------------------
@app.get("/")
def root():
    # FastAPI auto-generates docs at /docs (Swagger UI) and /redoc.
    return {"message": "Water Potability API is running. See /docs."}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "artifact_path": str(artifact_path),
        "is_pipeline": IS_PIPELINE,
        "n_features_expected": len(feature_names),
        "features": feature_names,
    }


# -------------------------------------------------
# 6) Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(payload: WaterInput):
    """
    How we predict (single row):
      1) We collect the JSON into a dict (preserve keys).
      2) If we DON'T have a pipeline, we enforce all fields exist (no None),
         because there's no imputer to fill gaps.
      3) We build a one-row DataFrame in the exact training order.
      4) We call model.predict (and model.predict_proba if available) and return JSON.
    """
    try:
        data: Dict[str, Any] = payload.model_dump()
        if not IS_PIPELINE:
            missing = [f for f in feature_names if data.get(f) is None]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing features {missing}. "
                           f"This artifact has no preprocessing; provide all fields."
                )

        # Build a one-row frame **in the trained order**
        row = {f: data.get(f, None) for f in feature_names}
        X = pd.DataFrame([row], columns=feature_names)

        # Predict class (0=unsafe, 1=safe)
        y_hat = int(model.predict(X)[0])

        # Try to compute probability of class 1 when available
        # proba = None
        if hasattr(model, "predict_proba"):
            # e.g., raw classifiers (LogReg/RF/XGB)
            proba = float(model.predict_proba(X)[0][1])
        else:
            # e.g., pipeline: check final step "clf"
            try:
                proba = float(model.named_steps["clf"].predict_proba(X)[0][1])  # pipeline final step
            except Exception:
                proba = None
        # We return both the numeric class and label
        resp = {
            "class": y_hat,
            "label": "Safe (potable)" if y_hat == 1 else "Unsafe (not potable)",
            "features_used_in_order": feature_names,
        }
        if proba is not None:
            resp["probability"] = round(proba, 4)
        return resp
    except HTTPException:
        raise
    except Exception as e:
        # Any unexpected error shows up here with a 400 and a helpful message.
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")