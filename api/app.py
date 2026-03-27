"""
FastAPI — Ride Demand Prediction API
=====================================
Endpoints:
  GET  /                    → health check + model info
  GET  /predict             → predict demand for location_code + hour + params
  POST /predict/batch       → predict for a list of inputs
  GET  /stats               → dataset statistics
  GET  /peak-hours          → top demand hours from training data
  GET  /docs                → auto-generated Swagger UI (built-in)
"""

import os
import json
import numpy as np
import joblib
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Load artefacts ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODEL_DIR   = os.path.join(BASE_DIR, "..", "model")

def _load(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(
            f"{name} not found at {path}. "
            f"Run `python model/train.py` first."
        )
    return joblib.load(path)

model       = _load("model.pkl")
label_enc   = _load("label_encoder.pkl")
scaler      = _load("scaler.pkl")
pca_model   = _load("pca.pkl")
kmeans      = _load("kmeans.pkl")

metrics_path = os.path.join(MODEL_DIR, "metrics.json")
metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}

stats_path = os.path.join(MODEL_DIR, "stats.json")
stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

# ── App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Ride Demand Prediction API",
    description=(
        "End-to-end ML API for predicting NYC ride demand "
        "by location zone and time. Built with XGBoost + Snowflake + PCA + KMeans."
    ),
    version="1.0.0",
)


# ── Pydantic models ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    location_code: int = Field(..., ge=0, description="Encoded location ID (0–N)")
    hour:          int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    day_of_week:   int = Field(0,  ge=0, le=6,  description="0=Monday … 6=Sunday")
    month:         int = Field(6,  ge=1, le=12, description="Month number (1–12)")
    is_weekend:    int = Field(0,  ge=0, le=1,  description="1 if weekend, else 0")
    hour_bin:      int = Field(2,  ge=0, le=7,  description="Hour bucket (hour // 3)")
    cluster:       int = Field(0,  ge=0, le=3,  description="Demand cluster (0–3)")


class BatchPredictRequest(BaseModel):
    inputs: List[PredictRequest]


def _predict_one(req: PredictRequest) -> float:
    features = np.array([[
        req.location_code,
        req.hour,
        req.day_of_week,
        req.month,
        req.is_weekend,
        req.hour_bin,
        req.cluster,
    ]])
    pred = model.predict(features)
    return round(float(pred[0]), 1)


# ── Routes ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "project": "Ride Demand Prediction System",
        "model": metrics.get("best_model", "XGBoost"),
        "features": metrics.get("features", []),
        "model_metrics": metrics.get("results", {}),
        "docs": "/docs",
    }


@app.get("/predict")
def predict(
    location_code: int,
    hour:          int,
    day_of_week:   int = 0,
    month:         int = 6,
    is_weekend:    int = 0,
    hour_bin:      int = 2,
    cluster:       int = 0,
):
    """Predict ride demand for a single location + time combination."""
    if not (0 <= hour <= 23):
        raise HTTPException(400, "hour must be 0–23")
    req = PredictRequest(
        location_code=location_code,
        hour=hour,
        day_of_week=day_of_week,
        month=month,
        is_weekend=is_weekend,
        hour_bin=hour_bin,
        cluster=cluster,
    )
    demand = _predict_one(req)
    return {
        "location_code": location_code,
        "hour":          hour,
        "day_of_week":   day_of_week,
        "month":         month,
        "cluster":       cluster,
        "predicted_demand": demand,
        "interpretation": _interpret(demand),
    }


@app.post("/predict/batch")
def predict_batch(body: BatchPredictRequest):
    """Predict demand for a list of location/time inputs."""
    if len(body.inputs) > 1000:
        raise HTTPException(400, "Max 1000 inputs per batch request")
    results = []
    for req in body.inputs:
        results.append({
            "location_code":    req.location_code,
            "hour":             req.hour,
            "predicted_demand": _predict_one(req),
        })
    return {"count": len(results), "predictions": results}


@app.get("/stats")
def get_stats():
    """Return dataset statistics computed during training."""
    if not stats:
        raise HTTPException(404, "Stats not found. Run model/train.py first.")
    return stats


@app.get("/peak-hours")
def peak_hours():
    """Return hardcoded peak-hour insights from the NYC dataset."""
    return {
        "top_demand_hours": [17, 18, 19, 20, 21],
        "low_demand_hours": [3, 4, 5],
        "peak_days": ["Friday", "Saturday"],
        "insight": (
            "Evening hours 17–21 show highest ride demand. "
            "Friday and Saturday are the busiest days. "
            "Early morning 3–5 AM has the lowest demand."
        ),
    }


@app.get("/model-info")
def model_info():
    """Return model metadata, PCA variance, and cluster info."""
    cluster_path = os.path.join(MODEL_DIR, "cluster_info.json")
    cluster_info = {}
    if os.path.exists(cluster_path):
        cluster_info = json.load(open(cluster_path))
    pca_variance = pca_model.explained_variance_ratio_.tolist()
    return {
        "best_model":         metrics.get("best_model"),
        "features":           metrics.get("features"),
        "results":            metrics.get("results"),
        "feature_importance": metrics.get("feature_importance"),
        "pca_explained_variance": {
            "pc1": round(pca_variance[0], 4),
            "pc2": round(pca_variance[1], 4),
            "total_retained": round(sum(pca_variance), 4),
        },
        "kmeans_clusters":    cluster_info,
    }


def _interpret(demand: float) -> str:
    if demand < 2:
        return "Very low demand — few drivers needed"
    elif demand < 5:
        return "Low demand"
    elif demand < 10:
        return "Moderate demand"
    elif demand < 20:
        return "High demand — increase driver availability"
    else:
        return "Very high demand — surge pricing recommended"


# ── Run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
