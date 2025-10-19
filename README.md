# Water Potability Prediction API

This project predicts whether water is **safe (potable)** or **unsafe** based on its chemical and physical properties.

---

### #0. Clone the repository
```bash
git clone https://github.com/<your-username>/water-potability-ML-API.git
cd water-potability-ML-API
```


## Workflow

### 1. Train Machine Laerning Model

- Train multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Select best model (**Random Forest**)
- Save artifacts in `best_model/`:
    - `best_model/model.joblib`: trained model
    - `best_model/model_meta.json`: feature names and metadata

---

### 2. FastAPI Service
The model is wrapped in a REST API (`main.py`) with two endpoints:
- `GET /health`: service health check
- `POST /predict`: make water safety predictions

---

### 3. Dockerize
Make sure you have Docker installed.

A `Dockerfile` and `requirements.txt` are provided.

The API will now be available at:
```bash
docker build -t water-api .
docker run -d -p 8001:8000 water-api
```
---


### 4. Call the API
Using Swagger UI

1. Open http://127.0.0.1:8001/docs
2. Expand the /predict endpoint 
3. Click "Try it out" and enter sample JSON input 
4. Execute to see the response

Using Postman

1. Create a new POST request 
2. URL: http://127.0.0.1:8001/predict
3. Body : raw : JSON, e.g:

```bash
{
  "ph": 7.2,
  "Hardness": 160,
  "Solids": 1800,
  "Chloramines": 6.5,
  "Sulfate": 220,
  "Conductivity": 350,
  "Organic_carbon": 6.0,
  "Trihalomethanes": 25,
  "Turbidity": 2.5
}

```
Example curl

```bash
curl -s -X POST http://127.0.0.1:8001/predict \
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
  "probability": 0.72
}
```