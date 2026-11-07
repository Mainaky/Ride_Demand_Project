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
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from functools import lru_cache
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

try:
    from api.auth import (
        init_db, register_user, authenticate_user,
        SignUpRequest, LoginRequest, AuthResponse, decode_token
    )
except ImportError:
    from auth import (
        init_db, register_user, authenticate_user,
        SignUpRequest, LoginRequest, AuthResponse, decode_token
    )

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

def _load(name):
    import joblib

    p = os.path.join(MODEL_DIR, name)
    if not os.path.exists(p):
        raise RuntimeError(f"{name} not found at {p}. Run python model/train.py first.")
    return joblib.load(p)


@lru_cache(maxsize=1)
def _get_models():
    """Load expensive estimators only when a forecast is requested."""
    return _load("model.pkl"), _load("classifier.pkl")


def _get_ride_service():
    """Import pandas and NumPy dependent ride helpers only when needed."""
    try:
        from api import ride_service
    except ImportError:
        import ride_service
    return ride_service

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
    description="Portfolio API for global ride planning estimates, NYC-trained demand forecasting, and passenger trip records.",
    version="2.7.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

@app.on_event("startup")
def startup_init():
    # Auth is the only database needed before the first request.
    init_db()


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(SECURITY)) -> Optional[dict]:
    if credentials and credentials.scheme.lower() == "bearer":
        payload = decode_token(credentials.credentials)
        if payload:
            return payload
    return None

def require_current_user(user: Optional[dict] = Depends(get_current_user)) -> dict:
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in to continue")
    return user


# ── Consumer Ride Booking Schemas ─────────────────────────────────────────────
class RideQuoteRequest(BaseModel):
    passenger_name: str = Field(default="Valued Passenger")
    pickup_address: str = Field(default="Times Square, Manhattan")
    pickup_lat: float = Field(ge=-90, le=90)
    pickup_lon: float = Field(ge=-180, le=180)
    dropoff_address: str = Field(default="JFK International Airport (Terminal 4)")
    dropoff_lat: float = Field(ge=-90, le=90)
    dropoff_lon: float = Field(ge=-180, le=180)
    hour: int = Field(default=18, ge=0, le=23)
    day_of_week: int = Field(default=4, ge=0, le=6)


class RideBookingRequest(BaseModel):
    passenger_name: str
    phone: Optional[str] = None
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
    driver_id: Optional[str] = None
    driver_name: Optional[str] = "Michael A."
    driver_rating: Optional[float] = 4.95
    car_model: Optional[str] = "Toyota Camry Hybrid"
    car_color: Optional[str] = None
    car_plate: Optional[str] = "DEMO-7K82"
    driver_lat: Optional[float] = None
    driver_lon: Optional[float] = None
    driver_distance_km: Optional[float] = None
    eta_mins: Optional[int] = 3
    predicted_hourly_demand: Optional[float] = None
    demand_tier: Optional[str] = None
    surge_multiplier: Optional[float] = None
    probability_finding_car_pct: Optional[float] = None


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
    import numpy as np

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
        "version": "2.7.0",
        "endpoints": {
            "health":         "GET /",
            "landmarks":      "GET /landmarks",
            "geocode":        "GET /geocode?q=place",
            "ride_quote":     "POST /ride/quote",
            "ride_book":      "POST /ride/book",
            "ride_history":   "GET /ride/history",
            "drivers":        "GET /drivers",
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
    landmarks = _get_ride_service().NYC_LANDMARKS
    return [
        {"name": name, "lat": coords[0], "lon": coords[1]}
        for name, coords in landmarks.items()
    ]


@app.get("/geocode")
def geocode_location(q: str = Query(..., min_length=3, max_length=160),
                     user: dict = Depends(require_current_user)):
    """Resolve a worldwide place or address to selectable map coordinates."""
    params = urlencode({"q": q.strip(), "limit": 5, "lang": "en"})
    request = Request(
        f"https://photon.komoot.io/api/?{params}",
        headers={"User-Agent": "ZonePulse/1.0 (mobility planning demo)"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Location search is temporarily unavailable. Please try again.") from exc

    results = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        coordinates = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coordinates) < 2:
            continue
        parts = [properties.get("name"), properties.get("street"),
                 properties.get("city") or properties.get("county"),
                 properties.get("state"), properties.get("country")]
        label = ", ".join(dict.fromkeys(str(part).strip() for part in parts if part))
        results.append({"label": label or q.strip(), "lat": float(coordinates[1]),
                        "lon": float(coordinates[0]), "country_code": properties.get("countrycode")})
    return results


@app.get("/weather")
def current_weather(lat: float = Query(..., ge=-90, le=90),
                    lon: float = Query(..., ge=-180, le=180),
                    user: dict = Depends(require_current_user)):
    """Return current conditions and local clock time for any global coordinate."""
    params = urlencode({
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
    })
    request = Request(
        f"https://api.open-meteo.com/v1/forecast?{params}",
        headers={"User-Agent": "ZonePulse/1.0 (mobility planning demo)"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        current = payload["current"]
        units = payload.get("current_units", {})
        return {
            "time": current["time"],
            "timezone": payload.get("timezone", "Local time"),
            "temperature": current["temperature_2m"],
            "feels_like": current["apparent_temperature"],
            "humidity": current["relative_humidity_2m"],
            "precipitation": current["precipitation"],
            "wind_speed": current["wind_speed_10m"],
            "is_day": bool(current["is_day"]),
            "weather_code": current["weather_code"],
            "units": units,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Current weather is temporarily unavailable. Please try again.") from exc


@app.post("/ride/quote")
def quote_ride(req: RideQuoteRequest, user: dict = Depends(require_current_user)):
    """
    Calculates probability of finding a car in the requested area and distance,
    estimated wait time, ML predicted demand, surge tier, fare options, and nearby drivers.
    """
    quote = _get_ride_service().compute_ride_quote(
        pickup_lat=req.pickup_lat,
        pickup_lon=req.pickup_lon,
        dropoff_lat=req.dropoff_lat,
        dropoff_lon=req.dropoff_lon,
        hour=req.hour,
        day_of_week=req.day_of_week,
    )
    _get_ride_service().save_quote_drivers(quote.get("nearby_drivers", []))
    quote["passenger_name"] = req.passenger_name
    quote["pickup_address"] = req.pickup_address
    quote["dropoff_address"] = req.dropoff_address
    return quote


@app.post("/ride/book")
def book_ride(req: RideBookingRequest, user: dict = Depends(require_current_user)):
    """Confirm and persist a ride booking for the signed-in account."""
    booking_data = req.model_dump()
    booking_data["passenger_name"] = user["sub"]
    booking_record = _get_ride_service().save_booking(booking_data, user["sub"])
    return {
        "status": "Booking Confirmed",
        "message": f"Your ride has been confirmed! Driver {req.driver_name} is arriving in {req.eta_mins} minutes.",
        "booking": booking_record,
    }


@app.get("/ride/history")
def ride_history(user: dict = Depends(require_current_user)):
    """Retrieve only the signed-in user's historical bookings."""
    return _get_ride_service().get_bookings(user["sub"])


@app.get("/drivers")
def driver_fleet(user: dict = Depends(require_current_user)):
    """Return persisted fleet profiles observed by the signed-in rider."""
    return _get_ride_service().get_driver_fleet()


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
    try:
        from api.storage import get_passenger
    except ImportError:
        from storage import get_passenger
    passenger = get_passenger(user.get("sub", ""))
    if not passenger:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Passenger account not found")
    return {
        "username": passenger["username"],
        "full_name": passenger["full_name"],
        "email": passenger["email"],
        "phone": passenger.get("phone"),
        "role": passenger["role"],
        "created_at": passenger["created_at"].isoformat() if passenger.get("created_at") else None,
    }


# ── Prediction Endpoints ──────────────────────────────────────────────────────

@app.get("/predict")
def predict(location_code: int, hour: int, day_of_week: int = 0,
            month: int = 6, is_weekend: int = 0,
            hour_bin: int = 2, cluster: int = 0,
            user: dict = Depends(require_current_user)):
    """REGRESSION — predict exact ride demand count."""
    req = PredictRequest(
        location_code=location_code, hour=hour,
        day_of_week=day_of_week, month=month,
        is_weekend=is_weekend, hour_bin=hour_bin, cluster=cluster
    )
    model, _ = _get_models()
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
             hour_bin: int = 2, cluster: int = 0,
             user: dict = Depends(require_current_user)):
    """CLASSIFICATION — predict demand category with probabilities."""
    req = PredictRequest(
        location_code=location_code, hour=hour,
        day_of_week=day_of_week, month=month,
        is_weekend=is_weekend, hour_bin=hour_bin, cluster=cluster
    )
    X = _features(req)
    _, classifier = _get_models()
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
def predict_batch(body: BatchRequest, user: dict = Depends(require_current_user)):
    """Batch regression + classification for multiple inputs."""
    if len(body.inputs) > 1000:
        raise HTTPException(400, "Max 1000 inputs per batch")
    model, classifier = _get_models()
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
def get_stats(user: dict = Depends(require_current_user)):
    return stats if stats else {"error": "Run model/train.py first"}


@app.get("/model-info")
def model_info(user: dict = Depends(require_current_user)):
    pca_model = _load("pca.pkl") if os.path.exists(os.path.join(MODEL_DIR, "pca.pkl")) else None
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
def peak_hours(user: dict = Depends(require_current_user)):
    return {
        "top_demand_hours": [17, 18, 19, 20, 21],
        "low_demand_hours": [3, 4, 5],
        "peak_days":        ["Friday", "Saturday"],
        "insight": "Evening hours 17-21 show highest ride demand. Early morning 3-5 AM is lowest.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
