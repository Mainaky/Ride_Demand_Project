"""
Global Ride Planning & Booking Service — ZonePulse
================================================
Handles:
  1. Worldwide geocoding plus optional NYC landmarks
  2. Haversine Distance & Duration calculation
  3. Spatial Grid & ML Model Inference (XGBoost Demand & Surge Tier)
  4. Probability of Finding a Car in this area and distance
  5. Dynamic Fare Estimation across vehicle tiers
  6. Nearby Driver Simulation around the pickup zone
  7. Persistent Ride Booking & Order History
"""

import math
import os
import random
import time
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

# ── 1. Comprehensive NYC Landmarks Database ────────────────────────────────────
NYC_LANDMARKS = {
    "Times Square, Manhattan": (40.7580, -73.9855),
    "Empire State Building, Manhattan": (40.7484, -73.9857),
    "Grand Central Terminal, Manhattan": (40.7527, -73.9772),
    "Penn Station / Madison Square Garden": (40.7505, -73.9934),
    "Central Park South / Columbus Circle": (40.7681, -73.9819),
    "Central Park (The Great Lawn)": (40.7812, -73.9665),
    "Wall Street / Financial District": (40.7074, -74.0113),
    "World Trade Center / One WTC": (40.7127, -74.0134),
    "Rockefeller Center, Manhattan": (40.7587, -73.9787),
    "Bryant Park / NY Public Library": (40.7536, -73.9832),
    "SoHo Shopping District, Manhattan": (40.7233, -74.0030),
    "Chinatown / Little Italy": (40.7158, -73.9970),
    "Greenwich Village / Washington Square": (40.7308, -73.9973),
    "Chelsea Market / High Line": (40.7420, -74.0048),
    "Hudson Yards / The Vessel": (40.7538, -74.0022),
    "Metropolitan Museum of Art (The Met)": (40.7794, -73.9632),
    "Upper East Side (86th St & Lex)": (40.7795, -73.9555),
    "Upper West Side (Broadway & 72nd)": (40.7785, -73.9820),
    "Columbia University, Morningside": (40.8075, -73.9626),
    "Harlem (125th St & Malcolm X Blvd)": (40.8078, -73.9454),
    "JFK International Airport (Terminal 4)": (40.6413, -73.7781),
    "LaGuardia Airport (LGA Terminal B)": (40.7769, -73.8740),
    "Brooklyn Bridge (Manhattan Entrance)": (40.7127, -74.0059),
    "DUMBO Waterfront / Brooklyn Bridge Park": (40.7033, -73.9896),
    "Williamsburg (Bedford Ave & N 7th)": (40.7183, -73.9567),
    "Barclays Center, Downtown Brooklyn": (40.6826, -73.9754),
    "Brooklyn Heights Promenade": (40.6974, -73.9976),
    "Prospect Park / Brooklyn Museum": (40.6712, -73.9638),
    "Coney Island Boardwalk": (40.5755, -73.9707),
    "Long Island City Waterfront (Gantry Plaza)": (40.7454, -73.9575),
    "Astoria (Ditmars Blvd), Queens": (40.7750, -73.9073),
    "Flushing Chinatown / Main St, Queens": (40.7595, -73.8300),
    "Yankee Stadium, The Bronx": (40.8296, -73.9262),
    "Staten Island Ferry Terminal": (40.6437, -74.0736),
}


# ── 2. Lazy Model Artifact Loader ─────────────────────────────────────────────
class ModelEngine:
    _instance = None

    def __init__(self):
        self.le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        self.model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        self.classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
        self.kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))

        # Pre-parse class coordinates for lightning-fast spatial search (<0.1ms)
        self.class_lats = np.array([float(c.split("_")[0]) for c in self.le.classes_])
        self.class_lons = np.array([float(c.split("_")[1]) for c in self.le.classes_])

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = ModelEngine()
        return cls._instance

    def find_nearest_location_code(self, lat: float, lon: float) -> int:
        loc_key = f"{round(lat, 2)}_{round(lon, 2)}"
        if loc_key in self.le.classes_:
            return int(self.le.transform([loc_key])[0])
        # Find nearest spatial cell by Euclidean distance
        dists = (self.class_lats - lat) ** 2 + (self.class_lons - lon) ** 2
        return int(np.argmin(dists))


# ── 3. Distance & Geometry Calculations ───────────────────────────────────────
def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns distance between two coordinates in kilometers."""
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


def estimate_travel_time_minutes(distance_km: float, hour: int) -> int:
    """Rough, city-agnostic travel-time estimate; not live traffic routing."""
    # Generic urban peak periods used only for a rough duration estimate
    if (8 <= hour <= 10) or (17 <= hour <= 20):
        speed_kmh = 20.0  # Heavy traffic
    elif 23 <= hour or hour <= 5:
        speed_kmh = 35.0  # Late night open traffic
    else:
        speed_kmh = 26.0  # Normal traffic

    minutes = int((distance_km / speed_kmh) * 60) + 4  # +4 mins for turns/traffic lights
    return max(6, minutes)


# ── 4. Core Prediction & Availability Probability Engine ─────────────────────
def compute_ride_quote(
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    hour: int = 18,
    day_of_week: int = 4,
    month: int = 6,
) -> dict:
    """
    Computes real-time ride quote:
      - Distance & Duration
      - ML Demand prediction & Surge Tier
      - Probability of Finding a Car in this particular area and distance
      - Dynamic Fares for 3 vehicle categories
      - Nearby Driver Simulations
    """
    engine = ModelEngine.get()

    distance_km = calculate_haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    distance_miles = round(distance_km * 0.621371, 2)
    duration_mins = estimate_travel_time_minutes(distance_km, hour)

    is_weekend = 1 if day_of_week in [5, 6] else 0
    hour_bin = hour // 3
    # The stored model was trained on NYC zones only; do not present NYC
    # nearest-zone extrapolation as a locally trained forecast elsewhere.
    in_nyc_training_area = 40.4 <= pickup_lat <= 41.2 and -74.4 <= pickup_lon <= -73.6
    class_labels = ["Low", "Medium", "High", "Very High"]
    if in_nyc_training_area:
        location_code = engine.find_nearest_location_code(pickup_lat, pickup_lon)
        cluster = int(engine.kmeans.predict([[location_code, hour]])[0])
        feature_vector = np.array([[location_code, hour, day_of_week, month, is_weekend, hour_bin, cluster]])
        raw_demand = float(engine.model.predict(feature_vector)[0])
        predicted_demand = round(max(1.0, raw_demand), 1)
        pred_class = int(engine.classifier.predict(feature_vector)[0])
        probs = engine.classifier.predict_proba(feature_vector)[0]
        prediction_source = "NYC-trained ML model"
        prediction_note = "This route is within the model's New York training coverage."
    else:
        # Transparent global planning prior: time-of-week framework, not ML.
        location_code = None
        cluster = 0
        rush_hour_factor = 1.45 if hour in (7, 8, 9, 16, 17, 18, 19) else 1.0
        weekend_factor = 1.15 if is_weekend else 1.0
        predicted_demand = round(8.0 * rush_hour_factor * weekend_factor, 1)
        pred_class = 0 if predicted_demand < 5 else 1 if predicted_demand < 10 else 2 if predicted_demand < 20 else 3
        probs = [0.7 if idx == pred_class else 0.1 for idx in range(4)]
        prediction_source = "Global planning baseline"
        prediction_note = "Outside New York training coverage. This directional baseline is not a locally trained or live-demand forecast."
    tier_label = class_labels[pred_class]
    tier_probabilities = {class_labels[i]: round(float(probs[i]) * 100, 1) for i in range(len(class_labels))}

    # 3. Probability of Finding a Car in that Area & Distance:
    # Illustrative availability heuristic for this simulated demo fleet:
    if pred_class == 0:  # Low demand zone
        base_prob = 0.76  # Fewer drivers cruising residential/remote areas
        wait_time_min, wait_time_max = 6, 11
        surge_mult = 1.0
        status_text = "Moderate estimated availability — simulated"
        status_color = "amber"
    elif pred_class == 1:  # Medium demand
        base_prob = 0.94  # Optimal driver supply matching demand
        wait_time_min, wait_time_max = 2, 4
        surge_mult = 1.1
        status_text = "High estimated availability — simulated"
        status_color = "green"
    elif pred_class == 2:  # High demand
        base_prob = 0.89  # Active zone, drivers present but high pickup requests
        wait_time_min, wait_time_max = 3, 6
        surge_mult = 1.35
        status_text = "Elevated demand — simulated availability"
        status_color = "blue"
    else:  # Very High demand (Surge Zone)
        base_prob = 0.68  # Temporary shortage of idle drivers due to sudden surge
        wait_time_min, wait_time_max = 7, 12
        surge_mult = 1.75
        status_text = "High demand — simulated availability and wait"
        status_color = "red"

    # Trip distance factor adjustment (drivers accept typical 2-15km trips with highest priority)
    if distance_km < 1.0 or distance_km > 35.0:
        base_prob *= 0.93  # slight reduction for extreme distances
    car_find_probability = round(min(0.99, max(0.45, base_prob)) * 100, 1)

    # 4. Dynamic Pricing Calculation for 3 Tiers
    # Base = ($3.50 base + $1.85/km + $0.35/min) * surge
    base_calc = (3.50 + (distance_km * 1.85) + (duration_mins * 0.35)) * surge_mult
    fare_uberx = round(max(8.0, base_calc), 2)
    fare_comfort = round(fare_uberx * 1.28, 2)
    fare_uberxl = round(fare_uberx * 1.65, 2)

    # 5. Nearby Driver Simulation around the pickup point
    random.seed(int(pickup_lat * 1000 + pickup_lon * 1000 + hour))
    driver_first_names = ["Michael", "David", "Ahmed", "Carlos", "Alex", "James", "Elena", "Marcus", "Dmitri", "Sarah"]
    driver_cars = [
        ("Toyota Camry Hybrid", "Black", "DEMO-7K82"),
        ("Honda Accord", "Silver", "DEMO-4M19"),
        ("Hyundai Sonata", "White", "DEMO-9B44"),
        ("Toyota RAV4", "Gray", "DEMO-3X71"),
        ("Tesla Model Y", "Blue", "DEMO-8Z50"),
    ]

    nearby_drivers = []
    num_drivers = 5 if pred_class in [1, 2] else (3 if pred_class == 0 else 2)
    for i in range(num_drivers):
        d_lat = pickup_lat + random.uniform(-0.0055, 0.0055)
        d_lon = pickup_lon + random.uniform(-0.0055, 0.0055)
        d_dist = calculate_haversine_distance(pickup_lat, pickup_lon, d_lat, d_lon)
        d_eta = max(1, int((d_dist / 22.0) * 60) + 1)
        car_info = driver_cars[i % len(driver_cars)]
        nearby_drivers.append({
            "driver_id": f"DEMO-{(int(round(pickup_lat * 10000)) * 1000003 + int(round(pickup_lon * 10000)) * 1009 + i) % 100000000:08d}",
            "name": f"{driver_first_names[i]} {chr(65 + i)}.",
            "rating": round(random.uniform(4.88, 4.99), 2),
            "car_model": car_info[0],
            "car_color": car_info[1],
            "plate": car_info[2],
            "lat": round(d_lat, 6),
            "lon": round(d_lon, 6),
            "distance_km": d_dist,
            "eta_minutes": d_eta,
        })

    nearby_drivers.sort(key=lambda d: d["eta_minutes"])
    best_eta = nearby_drivers[0]["eta_minutes"] if nearby_drivers else wait_time_min

    return {
        "distance_km": distance_km,
        "distance_miles": distance_miles,
        "duration_minutes": duration_mins,
        "pickup_location_code": location_code,
        "prediction_source": prediction_source,
        "prediction_note": prediction_note,
        "fare_currency": "USD equivalent",
        "predicted_hourly_demand": predicted_demand,
        "demand_tier": tier_label,
        "surge_multiplier": surge_mult,
        "probability_finding_car_pct": car_find_probability,
        "availability_status": status_text,
        "status_color": status_color,
        "estimated_wait_time_minutes": best_eta,
        "tier_probabilities": tier_probabilities,
        "fares": {
            "Standard (UberX/Mini)": fare_uberx,
            "Comfort (Prime/Sedan)": fare_comfort,
            "XL / SUV (6-Seater)": fare_uberxl,
        },
        "nearby_drivers": nearby_drivers,
    }


# ── 6. PostgreSQL-backed ride and driver records ────────────────────────────

def save_booking(data: dict, passenger_username: str = None) -> dict:
    try:
        from api.storage import save_booking as persist_booking
    except ImportError:
        from storage import save_booking as persist_booking
    return persist_booking(data, passenger_username or data.get("passenger_name", ""))


def get_bookings(passenger_name: str = None) -> list:
    if not passenger_name:
        return []
    try:
        from api.storage import list_bookings
    except ImportError:
        from storage import list_bookings
    return list_bookings(passenger_name)


def save_quote_drivers(driver_rows: list[dict]) -> None:
    try:
        from api.storage import upsert_quote_drivers
    except ImportError:
        from storage import upsert_quote_drivers
    upsert_quote_drivers(driver_rows)


def get_driver_fleet() -> list[dict]:
    try:
        from api.storage import list_drivers
    except ImportError:
        from storage import list_drivers
    return list_drivers()
