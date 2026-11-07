"""PostgreSQL persistence for ZonePulse accounts, fleet records, and trips.

Set DATABASE_URL to a PostgreSQL connection string in production. Local development
can use SQLite as a convenience, while Render injects the managed database URL.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, Text,
    create_engine, func, insert, select, update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

_raw_url = os.getenv("DATABASE_URL", "sqlite:///./zonepulse-local.db")
if _raw_url.startswith("postgres://"):
    _raw_url = _raw_url.replace("postgres://", "postgresql+psycopg://", 1)
elif _raw_url.startswith("postgresql://"):
    _raw_url = _raw_url.replace("postgresql://", "postgresql+psycopg://", 1)

connect_args = {"check_same_thread": False} if _raw_url.startswith("sqlite:") else {}
engine = create_engine(
    _raw_url,
    pool_pre_ping=True,
    pool_size=2 if not _raw_url.startswith("sqlite:") else 5,
    max_overflow=2 if not _raw_url.startswith("sqlite:") else 0,
    connect_args=connect_args,
)
metadata = MetaData()

passengers = Table(
    "passengers", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(50), nullable=False, unique=True, index=True),
    Column("email", String(120), nullable=False, unique=True, index=True),
    Column("password_hash", Text, nullable=False),
    Column("role", String(80), nullable=False, server_default="Mobility Operations Analyst"),
    Column("phone", String(40)),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

drivers = Table(
    "drivers", metadata,
    Column("driver_id", String(40), primary_key=True),
    Column("name", String(120), nullable=False),
    Column("rating", Float),
    Column("car_model", String(120)),
    Column("car_color", String(40)),
    Column("license_plate", String(40)),
    Column("phone", String(40)),
    Column("latitude", Float),
    Column("longitude", Float),
    Column("distance_km", Float),
    Column("eta_minutes", Integer),
    Column("status", String(40), nullable=False, server_default="Available"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

rides = Table(
    "rides", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("booking_id", String(32), nullable=False, unique=True, index=True),
    Column("passenger_id", Integer, ForeignKey("passengers.id"), nullable=False, index=True),
    Column("driver_id", String(40), ForeignKey("drivers.driver_id")),
    Column("passenger_name", String(120), nullable=False),
    Column("passenger_email", String(120), nullable=False),
    Column("passenger_phone", String(40)),
    Column("pickup_address", String(240), nullable=False),
    Column("pickup_lat", Float),
    Column("pickup_lon", Float),
    Column("dropoff_address", String(240), nullable=False),
    Column("dropoff_lat", Float),
    Column("dropoff_lon", Float),
    Column("distance_km", Float),
    Column("duration_mins", Integer),
    Column("ride_tier", String(80)),
    Column("fare", Float),
    Column("driver_name", String(120)),
    Column("driver_rating", Float),
    Column("car_model", String(120)),
    Column("car_color", String(40)),
    Column("car_plate", String(40)),
    Column("driver_lat", Float),
    Column("driver_lon", Float),
    Column("driver_distance_km", Float),
    Column("eta_mins", Integer),
    Column("status", String(40), nullable=False, server_default="Confirmed"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now(), index=True),
)


def initialize_schema() -> None:
    """Create missing tables. Existing rows are left untouched."""
    metadata.create_all(engine)


def _driver_payload(driver: dict, status: str = "Available") -> dict:
    return {
        "driver_id": str(driver.get("driver_id") or ""),
        "name": driver.get("name") or "Assigned driver",
        "rating": driver.get("rating"),
        "car_model": driver.get("car_model"),
        "car_color": driver.get("car_color"),
        "license_plate": driver.get("plate") or driver.get("car_plate"),
        "phone": driver.get("phone"),
        "latitude": driver.get("lat") if driver.get("lat") is not None else driver.get("driver_lat"),
        "longitude": driver.get("lon") if driver.get("lon") is not None else driver.get("driver_lon"),
        "distance_km": driver.get("distance_km") if driver.get("distance_km") is not None else driver.get("driver_distance_km"),
        "eta_minutes": driver.get("eta_minutes") if driver.get("eta_minutes") is not None else driver.get("eta_mins"),
        "status": status,
        "updated_at": func.now(),
    }


def upsert_quote_drivers(driver_rows: list[dict]) -> None:
    """Persist the simulated driver profiles returned to a signed-in rider."""
    if not driver_rows:
        return
    insert_fn = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    with engine.begin() as conn:
        for driver in driver_rows:
            payload = _driver_payload(driver)
            if not payload["driver_id"]:
                continue
            stmt = insert_fn(drivers).values(**payload)
            conn.execute(stmt.on_conflict_do_update(
                index_elements=[drivers.c.driver_id],
                set_={key: value for key, value in payload.items() if key != "driver_id"},
            ))


def save_booking(data: dict, passenger_username: str) -> dict:
    """Persist a ride, passenger snapshot, and assigned driver in one transaction."""
    clean_username = passenger_username.strip().lower()
    booking_id = f"ZP-{uuid.uuid4().hex[:10].upper()}"
    with engine.begin() as conn:
        passenger = conn.execute(
            select(passengers).where(passengers.c.username == clean_username)
        ).mappings().first()
        if not passenger:
            raise ValueError("Passenger account was not found")

        phone = (data.get("phone") or "").strip() or passenger.get("phone")
        if phone:
            conn.execute(update(passengers).where(passengers.c.id == passenger["id"]).values(phone=phone))

        driver_id = data.get("driver_id")
        if driver_id:
            insert_fn = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
            driver_payload = _driver_payload(data, status="Available")
            stmt = insert_fn(drivers).values(**driver_payload)
            conn.execute(stmt.on_conflict_do_update(
                index_elements=[drivers.c.driver_id],
                set_={key: value for key, value in driver_payload.items() if key != "driver_id"},
            ))

        ride = {
            "booking_id": booking_id,
            "passenger_id": passenger["id"],
            "driver_id": driver_id,
            "passenger_name": passenger["username"],
            "passenger_email": passenger["email"],
            "passenger_phone": phone,
            "pickup_address": data.get("pickup_address", "Pickup location"),
            "pickup_lat": data.get("pickup_lat"),
            "pickup_lon": data.get("pickup_lon"),
            "dropoff_address": data.get("dropoff_address", "Destination"),
            "dropoff_lat": data.get("dropoff_lat"),
            "dropoff_lon": data.get("dropoff_lon"),
            "distance_km": data.get("distance_km"),
            "duration_mins": data.get("duration_mins"),
            "ride_tier": data.get("ride_tier"),
            "fare": data.get("fare"),
            "driver_name": data.get("driver_name"),
            "driver_rating": data.get("driver_rating"),
            "car_model": data.get("car_model"),
            "car_color": data.get("car_color"),
            "car_plate": data.get("car_plate"),
            "driver_lat": data.get("driver_lat"),
            "driver_lon": data.get("driver_lon"),
            "driver_distance_km": data.get("driver_distance_km"),
            "eta_mins": data.get("eta_mins"),
            "status": "Driver En Route",
        }
        conn.execute(insert(rides).values(**ride))
        row = conn.execute(select(rides).where(rides.c.booking_id == booking_id)).mappings().one()
    return _json_record(dict(row))


def list_bookings(passenger_username: str) -> list[dict]:
    username = passenger_username.strip().lower()
    query = (
        select(rides)
        .select_from(rides.join(passengers, rides.c.passenger_id == passengers.c.id))
        .where(passengers.c.username == username)
        .order_by(rides.c.created_at.desc())
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [_json_record(dict(row)) for row in rows]


def list_drivers() -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(select(drivers).order_by(drivers.c.updated_at.desc(), drivers.c.name)).mappings().all()
    return [_json_record(dict(row)) for row in rows]


def get_passenger(username: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            select(passengers).where(passengers.c.username == username.strip().lower())
        ).mappings().first()
    return dict(row) if row else None


def _json_record(record: dict) -> dict:
    for key, value in record.items():
        if isinstance(value, datetime):
            record[key] = value.isoformat()
    return record
