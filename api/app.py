"""
FastAPI — ZonePulse Ride Demand Prediction & Real-Time Ride Booking API
=======================================================================
Ride Booking   : /ride/quote, /ride/book, /ride/history, /landmarks
Auth           : /auth/signup, /auth/login, /auth/me
Regression     : /predict        → predicted demand count
Classification : /classify       → demand category + probabilities
Batch          : /predict/batch  → multiple predictions at once
Info           : /stats, /model-info, /peak-hours
Docs           : /docs           → auto-generated Swagger UI
"""

import os, json
import numpy as np
import joblib
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

try:
    from api.auth import (
        init_db, register_user, authenticate_user,
        SignUpRequest, LoginRequest, AuthResponse, decode_token
    )
    from api.ride_service import (
        compute_ride_quote, save_booking, get_bookings,
        NYC_LANDMARKS, init_booking_db
    )
except ImportError:
    from auth import (
        init_db, register_user, authenticate_user,
        SignUpRequest, LoginRequest, AuthResponse, decode_token
    )
    from ride_service import (
        compute_ride_quote, save_booking, get_bookings,
        NYC_LANDMARKS, init_booking_db
    )

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

def _load(name):
    p = os.path.join(MODEL_DIR, name)
    if not os.path.exists(p):
        raise RuntimeError(f"{name} not found at {p}. Run python model/train.py first.")
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

SECURITY = HTTPBearer(auto_error=False)

app = FastAPI(
    title="ZonePulse Ride Demand & Booking API",
    description="Enterprise REST service for ride availability calculation, dynamic surge pricing, ML demand forecasting, and real-time ride booking.",
    version="2.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_init():
    init_db()
    init_booking_db()


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(SECURITY)) -> Optional[dict]:
    if credentials and credentials.scheme.lower() == "bearer":
        payload = decode_token(credentials.credentials)
        if payload:
            return payload
    return None


# ── Consumer Ride Booking Schemas ─────────────────────────────────────────────
class RideQuoteRequest(BaseModel):
    passenger_name: str = Field(default="Valued Passenger")
    pickup_address: str = Field(default="Times Square, Manhattan")
    pickup_lat: float = Field(ge=40.4, le=41.2)
    pickup_lon: float = Field(ge=-74.4, le=-73.6)
    dropoff_address: str = Field(default="JFK International Airport (Terminal 4)")
    dropoff_lat: float = Field(ge=40.4, le=41.2)
    dropoff_lon: float = Field(ge=-74.4, le=-73.6)
    hour: int = Field(default=18, ge=0, le=23)
    day_of_week: int = Field(default=4, ge=0, le=6)


class RideBookingRequest(BaseModel):
    passenger_name: str
    phone: Optional[str] = "+1 (212) 555-0199"
    pickup_address: str
    pickup_lat: float
    pickup_lon: float
    dropoff_address: str
    dropoff_lat: float
    dropoff_lon: float
    distance_km: float
    duration_mins: int
    ride_tier: str = "Standard (UberX/Mini)"
    fare: float
    driver_name: Optional[str] = "Michael A."
    driver_rating: Optional[float] = 4.95
    car_model: Optional[str] = "Toyota Camry Hybrid"
    car_plate: Optional[str] = "TLC #7K82"
    eta_mins: Optional[int] = 3


# ── Operational ML Schemas ────────────────────────────────────────────────────
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
    if demand < 5:    return "Very low demand"
    elif demand < 10: return "Low demand"
    elif demand < 20: return "Moderate demand"
    elif demand < 50: return "High demand — increase driver availability"
    else:             return "Very high demand — surge pricing recommended"


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "ZonePulse Ride Demand & Booking API",
        "version": "2.5.0",
        "endpoints": {
            "health":         "GET /",
            "landmarks":      "GET /landmarks",
            "ride_quote":     "POST /ride/quote",
            "ride_book":      "POST /ride/book",
            "ride_history":   "GET /ride/history",
            "auth_signup":    "POST /auth/signup",
            "auth_login":     "POST /auth/login",
            "auth_me":        "GET /auth/me",
            "regression":     "GET /predict",
            "classification": "GET /classify",
            "batch":          "POST /predict/batch",
            "stats":          "GET /stats",
            "model_info":     "GET /model-info",
            "docs":           "GET /docs",
        }
    }


# ── Consumer Ride Booking Endpoints ───────────────────────────────────────────

@app.get("/landmarks")
def list_landmarks():
    """Return pre-mapped NYC landmarks with exact latitude and longitude."""
    return [
        {"name": name, "lat": coords[0], "lon": coords[1]}
        for name, coords in NYC_LANDMARKS.items()
    ]


@app.post("/ride/quote")
def quote_ride(req: RideQuoteRequest):
    """
    Calculates probability of finding a car in the requested area and distance,
    estimated wait time, ML predicted demand, surge tier, fare options, and nearby drivers.
    """
    quote = compute_ride_quote(
        pickup_lat=req.pickup_lat,
        pickup_lon=req.pickup_lon,
        dropoff_lat=req.dropoff_lat,
        dropoff_lon=req.dropoff_lon,
        hour=req.hour,
        day_of_week=req.day_of_week,
    )
    quote["passenger_name"] = req.passenger_name
    quote["pickup_address"] = req.pickup_address
    quote["dropoff_address"] = req.dropoff_address
    return quote


@app.post("/ride/book")
def book_ride(req: RideBookingRequest):
    """Confirm and persist a ride booking."""
    booking_record = save_booking(req.model_dump())
    return {
        "status": "Booking Confirmed",
        "message": f"Your ride has been confirmed! Driver {req.driver_name} is arriving in {req.eta_mins} minutes.",
        "booking": booking_record,
    }


@app.get("/ride/history")
def ride_history(passenger_name: Optional[str] = None):
    """Retrieve historical booked trips."""
    return get_bookings(passenger_name)


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/signup", response_model=AuthResponse)
def signup(req: SignUpRequest):
    try:
        return register_user(req)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    try:
        return authenticate_user(req)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(err))


@app.get("/auth/me")
def me(user: Optional[dict] = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Valid token required")
    return {"username": user.get("sub"), "role": user.get("role")}


# ── Prediction Endpoints ──────────────────────────────────────────────────────

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
