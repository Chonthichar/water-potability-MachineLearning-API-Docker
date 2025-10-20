# Week 43 Lab - Water Potability Prediction with FastAPI
# ======================================================
# This script trains and saves a full ML pipeline:
#    1. Preprocess missing values
#    2. Balance classes with SMOTE
#    3. Train and select the best model via GridSearchCV
#    4. Save the final pipeline and metadata

# Dataset:
# Water Potability dataset (Kaggle) with 3276 samples,
# 9 input features, and 1 target variable "Potability" (0 = unsafe, 1 = safe)

# -----------------------
# #0: Setup
# -----------------------

from pathlib import Path
import json, time
import pandas as pd

# sklearn and imblearn utilities
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline  # to include SMOTE
from imblearn.over_sampling import SMOTE

# Classifiers we want to test
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import joblib

# -----------------------
# Paths and data loading
# -----------------------
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_CSV  = REPO_ROOT / "data" / "water_potability.csv"
OUT_DIR    = REPO_ROOT / "best_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Let’s write a helper function so we can load data easily
def load_df():
    # First we try to load the local dataset
    if LOCAL_CSV.exists():
        print(f"Loading dataset from: {LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV)
    # If the file is missing, we fall back to KaggleHub
    try:
        import kagglehub
        kaggle_path = kagglehub.dataset_download("adityakadiwal/water-potability")
        csv_path = Path(kaggle_path) / "water_potability.csv"
        print(f"Loading dataset from KaggleHub: {csv_path}")
        return pd.read_csv(csv_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find {LOCAL_CSV} and KaggleHub download failed. Error: {e}"
        )
# Load the data
df = load_df()

# -----------------------
# 1) Features / target
# -----------------------
# what we want to predict
TARGET = "Potability"
FEATURES = [
    "ph","Hardness","Solids","Chloramines","Sulfate",
    "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"
]
# We separate input features (X) and target (y)
X = df[FEATURES].copy()
# Target variable (0 or 1)
y = df[TARGET].astype(int)


# -----------------------
# 2) Train / test split
# -----------------------
# We always keep aside a test set (20%) to measure final performance.
# We use stratify=y so that safe/unsafe water proportions stay balanced in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# -----------------------
# 3: Preprocessing (handle missing values)
# -----------------------
# Notice that some columns have missing values.
# Instead of dropping rows, we fill them
# - KNNImputer: for ph, Sulfate, Trihalomethanes (uses neighbors to guess values)
# - Median imputer: for the rest (for robust against outliers)


knn_cols = ["ph", "Sulfate", "Trihalomethanes"]
other_cols = [c for c in FEATURES if c not in knn_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("knn_imp", KNNImputer(n_neighbors=5), knn_cols),
        ("med_imp", SimpleImputer(strategy="median"), other_cols),
    ],
    # drop unused columns
    remainder="drop",
    verbose_feature_names_out=False,
)

# -----------------------
# 4) Build a FULL training pipeline:
# -----------------------
# Let’s put everything together in a pipeline:
#   - Step 1: Preprocess missing values
#   - Step 2: Balance classes with SMOTE
#   - Step 3: Train classifier
# Using a pipeline means we don’t have to repeat preprocessing
# every time we test a new model. Much cleaner!

base_pipe = ImbPipeline(steps=[
    ("preprocess", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression())
])

# -----------------------
# 5) Define models and hyperparameters to test
# -----------------------
# Here we let GridSearchCV choose the best model for us.
# We will compare four algorithms:
#   1. Logistic Regression (simple linear baseline)
#   2. Decision Tree (non-linear baseline)
#   3. Random Forest (ensemble of trees)
#   4. XGBoost (boosted trees, often very strong)

param_grids = [
    {
        "clf": [LogisticRegression(max_iter=1000)],
        "clf__C": [0.5, 1.0, 2.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
    {
        "clf": [DecisionTreeClassifier(random_state=42)],
        "clf__max_depth": [None, 8, 12, 20],
        "clf__min_samples_split": [2, 5, 10],
    },
    {
        "clf": [RandomForestClassifier(random_state=42, n_jobs=-1)],
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    },
    {
        "clf": [XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)],
        "clf__n_estimators": [300],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.1, 0.05],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    },
]

# Cross-validation: we use StratifiedKFold so every fold
# has a balanced ratio of safe/unsafe water.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch = try all combinations of models + parameters,
# pick the best one based on accuracy.
gs = GridSearchCV(
    base_pipe,
    param_grid=param_grids,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

# Train (this may take a few minutes)
gs.fit(X_train, y_train)

# -------------------
# 6) Evaluate best model on test set
# -------------------
y_pred = gs.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nBest model:", type(gs.best_estimator_.named_steps["clf"]).__name__)
print("Best params:", gs.best_params_)
print("Test Accuracy:", round(acc, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# -----------------------
# 7: Save pipeline and metadata
# -----------------------
# We don’t just save the model – we save the *whole pipeline*
# so we can reuse it later inside FastAPI without worrying about preprocessing.
pipeline_path = OUT_DIR / "pipeline.joblib"
joblib.dump(gs.best_estimator_, pipeline_path)

# Save metadata (extra info about the training)
raw_params = gs.best_params_.copy()
if "clf" in raw_params:
    raw_params["clf"] = type(raw_params["clf"]).__name__

# Convert to JSON format
safe_params = {}
for k, v in raw_params.items():
    try:
        v = v.item()
    except Exception:
        pass
    safe_params[k] = v


meta = {
    "artifact": "pipeline.joblib",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "feature_names_in_order": FEATURES,
    "target": TARGET,
    "winner": type(gs.best_estimator_.named_steps["clf"]).__name__,
    "cv_best_params": safe_params,            # <--- use sanitized version
    "cv_best_score": float(gs.best_score_),   # optional but useful
    "test_accuracy": float(acc),
    "preprocessing": {
        "KNNImputer_on": knn_cols,
        "MedianImputer_on": other_cols,
        "SMOTE": True
    },
    "scoring": "accuracy"
}
with open(OUT_DIR / "model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved pipeline to: {pipeline_path}")
print(f"Saved metadata to: {OUT_DIR / 'model_meta.json'}")