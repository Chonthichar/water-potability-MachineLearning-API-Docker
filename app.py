# app.py
import json
import joblib
import numpy as np
import gradio as gr
from pathlib import Path
import pandas as pd

PIPE_PATH = Path("best_model/pipeline.joblib")
META_PATH = Path("best_model/model_meta.json")

# ---- load once ----
pipeline = joblib.load(PIPE_PATH)

# Try to respect feature order saved in metadata; fallback to a known order
default_order = [
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
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    feature_order = meta.get("feature_order") or meta.get("features_used_in_order") or default_order
else:
    feature_order = default_order

label_map = {0: "Unsafe (not potable)", 1: "Safe (potable)"}

def predict_one(
        ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
):
    values = {
        "ph": ph,
        "Hardness": Hardness,
        "Solids": Solids,
        "Chloramines": Chloramines,
        "Sulfate": Sulfate,
        "Conductivity": Conductivity,
        "Organic_carbon": Organic_carbon,
        "Trihalomethanes": Trihalomethanes,
        "Turbidity": Turbidity,
    }
    X = np.array([[values[k] for k in feature_order]])
    proba = getattr(pipeline, "predict_proba", None)
    if proba:
        p = float(proba(X)[0, 1])
    else:
        # if model has no predict_proba, estimate from decision_function or 0/1
        y_hat = int(pipeline.predict(X)[0])
        p = 0.5 if y_hat == 1 else 0.0
    y = int(pipeline.predict(X)[0])
    return label_map[y], round(p, 4), feature_order

# --- A simple "chat" wrapper that lets users type numbers freely ---
import re
def chat_predict(message, history):
    # Accept formats like: "ph=7.2 Hardness: 160 ...", or a JSON blob.
    try:
        if message.strip().startswith("{"):
            values = json.loads(message)
        else:
            # extract key=value pairs crudely
            pairs = dict(re.findall(r'(\b[a-zA-Z_]+)\s*[:=]\s*([+-]?\d+(\.\d+)?)', message))
            values = {k: float(v) for k, v in pairs.items()}
        # Fill missing with NaN -> pipeline imputer will handle
        x = []
        for k in feature_order:
            x.append(float(values.get(k, np.nan)))
        X = np.array([x])
        proba = getattr(pipeline, "predict_proba", None)
        if proba:
            p = float(proba(X)[0, 1])
        else:
            y_hat = int(pipeline.predict(X)[0])
            p = 0.5 if y_hat == 1 else 0.0
        y = int(pipeline.predict(X)[0])
        return f"{label_map[y]} (p={p:.4f})"
    except Exception as e:
        return ("Please provide inputs as JSON or 'key=value' pairs for: "
                f"{', '.join(feature_order)}. Example:\n"
                "ph=7.2 Hardness=160 Solids=1800 Chloramines=6.5 Sulfate=220 "
                "Conductivity=350 Organic_carbon=6 Trihalomethanes=25 Turbidity=2.5")

# --------- Gradio UI ----------
with gr.Blocks(title="Water Potability Predictor") as demo:
    gr.Markdown("# 💧 Water Potability Predictor\nEnter water chemistry and get a quick potability prediction.")
    with gr.Tabs():
        with gr.Tab("Form"):
            with gr.Row():
                ph = gr.Number(value=7.0, label="ph")
                Hardness = gr.Number(value=150.0, label="Hardness (mg/L)")
                Solids = gr.Number(value=1000.0, label="Solids (ppm)")
                Chloramines = gr.Number(value=6.0, label="Chloramines (ppm)")
                Sulfate = gr.Number(value=200.0, label="Sulfate (mg/L)")
                Conductivity = gr.Number(value=300.0, label="Conductivity (μS/cm)")
                Organic_carbon = gr.Number(value=6.0, label="Organic carbon (ppm)")
                Trihalomethanes = gr.Number(value=40.0, label="Trihalomethanes (μg/L)")
                Turbidity = gr.Number(value=2.0, label="Turbidity (NTU)")
            btn = gr.Button("Predict")
            out_label = gr.Textbox(label="Label")
            out_prob = gr.Number(label="Probability (class=Safe)")
            out_order = gr.JSON(label="Feature order used")
            btn.click(
                predict_one,
                inputs=[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity],
                outputs=[out_label, out_prob, out_order],
            )
        with gr.Tab("Chat"):
            gr.Markdown(
                "Type values as JSON or key=value pairs.\n\n"
                "**Example:** `ph=7.2 Hardness=160 Solids=1800 Chloramines=6.5 "
                "Sulfate=220 Conductivity=350 Organic_carbon=6 Trihalomethanes=25 Turbidity=2.5`"
            )
            chat = gr.ChatInterface(fn=chat_predict, type="messages")

# Spaces will look for this
if __name__ == "__main__":
    demo.launch()
