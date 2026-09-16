"""
FastAPI — Ride Demand Prediction API
=====================================
Regression  : /predict        → predicted demand count
Classification: /classify     → demand category + probabilities
Batch       : /predict/batch  → multiple predictions at once
Info        : /stats, /model-info, /peak-hours
"""

import os, json
import numpy as np
import joblib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

def _load(name):
    p = os.path.join(MODEL_DIR, name)
    if not os.path.exists(p):
        raise RuntimeError(f"{name} not found. Run python model/train.py first.")
    return joblib.load(p)

model      = _load("model.pkl")
classifier = _load("classifier.pkl")

metrics_path = os.path.join(MODEL_DIR, "metrics.json")
metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}

stats_path = os.path.join(MODEL_DIR, "stats.json")
stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

CLASS_LABELS = ["Low", "Medium", "High", "Very High"]
CLASS_ACTIONS = {
    "Low":       "Normal operations — no surge needed",
    "Medium":    "Monitor closely — consider pre-positioning drivers",
    "High":      "Increase driver availability in this zone",
    "Very High": "Activate surge pricing — critical demand zone",
}

app = FastAPI(
    title="Ride Demand Prediction API",
    description="Regression + Classification endpoints for NYC ride demand forecasting.",
    version="2.0.0",
)


class PredictRequest(BaseModel):
    location_code: int = Field(..., ge=0,  description="Encoded location zone ID")
    hour:          int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week:   int = Field(0,  ge=0, le=6,  description="0=Monday … 6=Sunday")
    month:         int = Field(6,  ge=1, le=12, description="Month (1-12)")
    is_weekend:    int = Field(0,  ge=0, le=1,  description="1=weekend, 0=weekday")
    hour_bin:      int = Field(2,  ge=0, le=7,  description="Hour bucket (hour//3)")
    cluster:       int = Field(0,  ge=0, le=3,  description="KMeans demand cluster (0-3)")


class BatchRequest(BaseModel):
    inputs: List[PredictRequest]


def _features(req: PredictRequest):
    return np.array([[
        req.location_code, req.hour, req.day_of_week,
        req.month, req.is_weekend, req.hour_bin, req.cluster
    ]])


def _interpret_demand(demand: float) -> str:
    if demand < 5:   return "Very low demand"
    elif demand < 10: return "Low demand"
    elif demand < 20: return "Moderate demand"
    elif demand < 50: return "High demand — increase driver availability"
    else:             return "Very high demand — surge pricing recommended"


@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "2.0.0",
        "endpoints": {
            "regression":     "GET /predict",
            "classification": "GET /classify",
            "batch":          "POST /predict/batch",
            "stats":          "GET /stats",
            "model_info":     "GET /model-info",
            "docs":           "GET /docs",
        }
    }


@app.get("/predict")
def predict(location_code: int, hour: int, day_of_week: int = 0,
            month: int = 6, is_weekend: int = 0,
            hour_bin: int = 2, cluster: int = 0):
    """REGRESSION — predict exact ride demand count."""
    req = PredictRequest(
        location_code=location_code, hour=hour,
        day_of_week=day_of_week, month=month,
        is_weekend=is_weekend, hour_bin=hour_bin, cluster=cluster
    )
    demand = round(float(model.predict(_features(req))[0]), 1)
    return {
        "type":               "regression",
        "location_code":      location_code,
        "hour":               hour,
        "predicted_demand":   demand,
        "interpretation":     _interpret_demand(demand),
    }


@app.get("/classify")
def classify(location_code: int, hour: int, day_of_week: int = 0,
             month: int = 6, is_weekend: int = 0,
             hour_bin: int = 2, cluster: int = 0):
    """CLASSIFICATION — predict demand category with probabilities."""
    req = PredictRequest(
        location_code=location_code, hour=hour,
        day_of_week=day_of_week, month=month,
        is_weekend=is_weekend, hour_bin=hour_bin, cluster=cluster
    )
    X = _features(req)
    pred_class = int(classifier.predict(X)[0])
    label = CLASS_LABELS[pred_class]

    proba = {}
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X)[0]
        proba = {CLASS_LABELS[i]: round(float(probs[i]) * 100, 1) for i in range(len(CLASS_LABELS))}

    return {
        "type":             "classification",
        "location_code":    location_code,
        "hour":             hour,
        "predicted_class":  pred_class,
        "demand_category":  label,
        "probabilities_%":  proba,
        "recommended_action": CLASS_ACTIONS[label],
    }


@app.post("/predict/batch")
def predict_batch(body: BatchRequest):
    """Batch regression + classification for multiple inputs."""
    if len(body.inputs) > 1000:
        raise HTTPException(400, "Max 1000 inputs per batch")
    results = []
    for req in body.inputs:
        X = _features(req)
        demand     = round(float(model.predict(X)[0]), 1)
        pred_class = int(classifier.predict(X)[0])
        results.append({
            "location_code":    req.location_code,
            "hour":             req.hour,
            "predicted_demand": demand,
            "demand_category":  CLASS_LABELS[pred_class],
        })
    return {"count": len(results), "predictions": results}


@app.get("/stats")
def get_stats():
    return stats if stats else {"error": "Run model/train.py first"}


@app.get("/model-info")
def model_info():
    pca_model = joblib.load(os.path.join(MODEL_DIR, "pca.pkl")) \
        if os.path.exists(os.path.join(MODEL_DIR, "pca.pkl")) else None
    pca_variance = {}
    if pca_model:
        ev = pca_model.explained_variance_ratio_
        pca_variance = {"pc1": round(ev[0],4), "pc2": round(ev[1],4),
                        "total": round(sum(ev),4)}
    return {
        "regression": {
            "best_model":        metrics.get("best_model"),
            "results":           metrics.get("results"),
            "feature_importance":metrics.get("feature_importance"),
        },
        "classification": {
            "best_model":        metrics.get("best_classifier"),
            "class_labels":      CLASS_LABELS,
            "demand_thresholds": metrics.get("demand_thresholds"),
            "results":           metrics.get("classification_results"),
            "feature_importance":metrics.get("cls_feature_importance"),
        },
        "pca_variance": pca_variance,
    }


@app.get("/peak-hours")
def peak_hours():
    return {
        "top_demand_hours": [17, 18, 19, 20, 21],
        "low_demand_hours": [3, 4, 5],
        "peak_days":        ["Friday", "Saturday"],
        "insight": "Evening hours 17-21 show highest ride demand. Early morning 3-5 AM is lowest.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)