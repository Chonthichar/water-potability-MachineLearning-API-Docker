# Water Potability Prediction Train ML, API and Docker

We build and ship a full workflow to predict whether water is safe (potable) or unsafe from its chemical/physical properties.
###  why this project is matter

Clean water is essential, but testing can often be slow and costly. With machine learning, we can predict water safety directly from chemical properties such as pH, hardness, and turbidity.

###  Real-world impact
- Faster decisions: instead of waiting for full lab tests, authorities can quickly screen water.
- Cost-efficient: automated predictions can reduce testing costs.
- Scalable: the same trained model can be used across many regions.


### What we’ll learn and do from the lab

By running this repo, we’ll see the full ML : API : Docker workflow:
1) Model training : we compare ML models, handle class imbalance, and pick the best one (Random Forest).
2) Save: persist a full pipeline (preprocess and model) with metadata.
3) Deployment : we wrap the trained model in a FastAPI service so any app can use it.
4) Ship: containerize with Docker and call the API from any client.

In short: we don’t just predict water safety, we follow the same pipeline used in real-world AI projects.

---

### #0. Clone the repository
```bash
git clone https://github.com/Chonthichar/water-potability-MachineLearning-API-Docker.git
```
```bash
cd water-potability-MachineLearning-API-Docker
```

Requirements: Python 3.10+ recommended. See requirements.txt.

## Workflow

### 1. Train the model (and save the pipeline)


We train multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost) using cross-validation. We handle missing values (KNN and median imputers) and balance classes with SMOTE. Then we save the entire pipeline so serving is easy.
```bash
python train_pipeline.py
```
Artifacts (created under best_model/):

- `pipeline.joblib` – the full pipeline (preprocess and model)

- `model_meta.json` – metadata (feature order, best params, scores)

Why a pipeline? We keep preprocessing and the model together, so the API can accept inputs (even with some nulls) and predict without re-implementing preprocessing.

### 2. Run the FastAPI service (locally)
```bash
uvicorn main:app --reload --port 8000
```

Open interactive docs: http://127.0.0.1:8000/docs

The model is wrapped in a REST API (`main.py`) with two endpoints:
- `GET /health`: service health check
- `POST /predict`: make water safety predictions


---

### 3. Docker: build and run
We ship a minimal image that includes the API and artifact.

A `Dockerfile` and `requirements.txt` are provided.

The API will now be available at:

```bash
docker build -t water-api .
docker run -d -p 8000:8000 water-api
```
---

Open http://127.0.0.1:8000/docs
to test.
### 4. Call the API
Using Swagger UI

1. Open http://127.0.0.1:8000/docs
2. Expand the /predict endpoint 
3. Click "Try it out" and enter sample JSON input 
4. Execute to see the response

Using Postman

1. Create a new POST request 
2. URL: http://127.0.0.1:8000/predict
3. Body : raw : JSON, e.g:

```bash
{
    "ph": 9.4,
    "Hardness": 90.0,
    "Solids": 350.0,
    "Chloramines": 3.0,
    "Sulfate": 200.0,
    "Conductivity": 280.0,
    "Organic_carbon": 3.5,
    "Trihalomethanes": 40.0,
    "Turbidity": 0.9
}

```
Example curl

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"ph":7.2,"Hardness":160,"Solids":1800,"Chloramines":6.5,
      "Sulfate":220,"Conductivity":350,"Organic_carbon":6.0,
      "Trihalomethanes":25,"Turbidity":2.5}'
```
Example Response

```bash
{
    "class": 1,
    "label": "Safe (potable)",
    "probability": 0.53,
    "features_used_in_order": [
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity"
    ]
}
```

With this setup, we can demonstrate how data science models are trained, deployed, and served through APIs in production-like environments.
