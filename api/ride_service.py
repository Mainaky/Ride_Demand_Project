"""
Ride Intelligence & Booking Service — ZonePulse
================================================
Handles:
  1. Popular NYC Landmarks & Geocoding
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
    """Estimate travel time in minutes based on NYC traffic conditions."""
    # Peak traffic hours in NYC (8-10 AM and 5-8 PM)
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

    location_code = engine.find_nearest_location_code(pickup_lat, pickup_lon)
    is_weekend = 1 if day_of_week in [5, 6] else 0
    hour_bin = hour // 3
    cluster = int(engine.kmeans.predict([[location_code, hour]])[0])

    feature_vector = np.array([[
        location_code, hour, day_of_week, month, is_weekend, hour_bin, cluster
    ]])

    # 1. XGBoost Regression (Exact Demand Count in this Zone)
    raw_demand = float(engine.model.predict(feature_vector)[0])
    predicted_demand = round(max(1.0, raw_demand), 1)

    # 2. XGBoost Classification (Surge Tier)
    pred_class = int(engine.classifier.predict(feature_vector)[0])
    class_labels = ["Low", "Medium", "High", "Very High"]
    tier_label = class_labels[pred_class]

    probs = engine.classifier.predict_proba(feature_vector)[0]
    tier_probabilities = {class_labels[i]: round(float(probs[i]) * 100, 1) for i in range(len(class_labels))}

    # 3. Probability of Finding a Car in that Area & Distance:
    # Based on real-world driver density vs passenger demand ratio:
    if pred_class == 0:  # Low demand zone
        base_prob = 0.76  # Fewer drivers cruising residential/remote areas
        wait_time_min, wait_time_max = 6, 11
        surge_mult = 1.0
        status_text = "Moderate Availability — Drivers are cruising nearby"
        status_color = "amber"
    elif pred_class == 1:  # Medium demand
        base_prob = 0.94  # Optimal driver supply matching demand
        wait_time_min, wait_time_max = 2, 4
        surge_mult = 1.1
        status_text = "High Availability — Fast Match Guaranteed"
        status_color = "green"
    elif pred_class == 2:  # High demand
        base_prob = 0.89  # Active zone, drivers present but high pickup requests
        wait_time_min, wait_time_max = 3, 6
        surge_mult = 1.35
        status_text = "Elevated Demand — High Driver Density"
        status_color = "blue"
    else:  # Very High demand (Surge Zone)
        base_prob = 0.68  # Temporary shortage of idle drivers due to sudden surge
        wait_time_min, wait_time_max = 7, 12
        surge_mult = 1.75
        status_text = "Surge Demand — High Congestion & High Wait Times"
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
        ("Toyota Camry Hybrid", "Black", "TLC #7K82"),
        ("Honda Accord", "Silver", "TLC #4M19"),
        ("Hyundai Sonata", "White", "TLC #9B44"),
        ("Toyota RAV4", "Gray", "TLC #3X71"),
        ("Tesla Model Y", "Blue", "TLC #8Z50"),
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
            "driver_id": f"DRV-{1000 + i}",
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


# ── 6. Persistent Ride Bookings Database (SQLite) ──────────────────────────────
import sqlite3

BOOKING_DB = os.getenv("BOOKINGS_DB_PATH", os.path.join(os.getenv("ZONEPULSE_DATA_DIR", os.path.dirname(__file__)), "bookings.db"))

def init_booking_db():
    with sqlite3.connect(BOOKING_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                booking_id TEXT PRIMARY KEY,
                passenger_name TEXT NOT NULL,
                phone TEXT,
                pickup_address TEXT NOT NULL,
                pickup_lat REAL,
                pickup_lon REAL,
                dropoff_address TEXT NOT NULL,
                dropoff_lat REAL,
                dropoff_lon REAL,
                distance_km REAL,
                duration_mins INTEGER,
                ride_tier TEXT,
                fare REAL,
                driver_name TEXT,
                driver_rating REAL,
                car_model TEXT,
                car_plate TEXT,
                eta_mins INTEGER,
                status TEXT DEFAULT 'Confirmed',
                created_at REAL NOT NULL
            )
        """)
        conn.commit()


def save_booking(data: dict) -> dict:
    init_booking_db()
    booking_id = f"ZP-NYC-{random.randint(10000, 99999)}"
    now = time.time()
    with sqlite3.connect(BOOKING_DB) as conn:
        conn.execute("""
            INSERT INTO bookings (
                booking_id, passenger_name, phone,
                pickup_address, pickup_lat, pickup_lon,
                dropoff_address, dropoff_lat, dropoff_lon,
                distance_km, duration_mins, ride_tier, fare,
                driver_name, driver_rating, car_model, car_plate,
                eta_mins, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            booking_id,
            data.get("passenger_name", "Valued Passenger"),
            data.get("phone", "+1 (212) 555-0199"),
            data.get("pickup_address", "Pickup Location"),
            data.get("pickup_lat", 40.7580),
            data.get("pickup_lon", -73.9855),
            data.get("dropoff_address", "Dropoff Destination"),
            data.get("dropoff_lat", 40.6413),
            data.get("dropoff_lon", -73.7781),
            data.get("distance_km", 10.0),
            data.get("duration_mins", 25),
            data.get("ride_tier", "Standard (UberX)"),
            data.get("fare", 28.50),
            data.get("driver_name", "Carlos M."),
            data.get("driver_rating", 4.93),
            data.get("car_model", "Toyota Camry Hybrid"),
            data.get("car_plate", "TLC #8M12"),
            data.get("eta_mins", 3),
            "Driver En Route",
            now,
        ))
        conn.commit()

    data["booking_id"] = booking_id
    data["status"] = "Driver En Route"
    data["created_at"] = now
    return data


def get_bookings(passenger_name: str = None) -> list:
    init_booking_db()
    with sqlite3.connect(BOOKING_DB) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if passenger_name:
            cursor.execute("SELECT * FROM bookings WHERE LOWER(passenger_name) = ? ORDER BY created_at DESC", (passenger_name.strip().lower(),))
        else:
            cursor.execute("SELECT * FROM bookings ORDER BY created_at DESC LIMIT 50")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
