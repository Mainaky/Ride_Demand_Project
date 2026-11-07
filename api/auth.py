"""User authentication and persistent SQLite storage for ZonePulse Ride Demand API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator

LOGGER = logging.getLogger("zonepulse_auth")
DATA_DIR = Path(os.getenv("ZONEPULSE_DATA_DIR", str(Path(__file__).resolve().parent)))
DB_PATH = Path(os.getenv("USER_DB_PATH", str(DATA_DIR / "users.db")))
SECRET_KEY = os.getenv("JWT_SECRET_KEY") or secrets.token_hex(32)


def init_db() -> None:
    """Create the users table if it does not already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'Mobility Operations Analyst',
                created_at REAL NOT NULL
            )
        """)
        conn.commit()


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a unique salt."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt$hash."""
    try:
        salt, dk_hex = stored_hash.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


def create_token(username: str, role: str) -> str:
    """Create an HMAC-SHA256 signed bearer token with a 7-day expiration."""
    payload = {
        "sub": username,
        "role": role,
        "exp": int(time.time()) + (7 * 86400),
    }
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_part = base64.urlsafe_b64encode(payload_json).decode("ascii").rstrip("=")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
    signature_part = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{payload_part}.{signature_part}"


def decode_token(token: str) -> Optional[dict]:
    """Verify the HMAC signature and expiry before accepting a token."""
    try:
        payload_part, signature_part = token.split(".", 1)
        expected = hmac.new(SECRET_KEY.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
        supplied = base64.urlsafe_b64decode(signature_part + "=" * (-len(signature_part) % 4))
        if not hmac.compare_digest(expected, supplied):
            return None
        payload_json = base64.urlsafe_b64decode(payload_part + "=" * (-len(payload_part) % 4))
        payload = json.loads(payload_json.decode("utf-8"))
        if not isinstance(payload, dict) or not payload.get("sub") or payload.get("exp", 0) < time.time():
            return None
        return payload
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


class SignUpRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(min_length=5, max_length=120)
    password: str = Field(min_length=6, max_length=100)

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", clean):
            raise ValueError("Invalid email address format")
        return clean


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=100)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    email: str
    role: str
    message: str


def register_user(req: SignUpRequest) -> AuthResponse:
    init_db()
    clean_username = req.username.strip().lower()
    clean_email = req.email.strip().lower()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (clean_username, clean_email))
        if cursor.fetchone():
            raise ValueError("Username or email is already registered")

        pwd_hash = hash_password(req.password)
        role = "Mobility Operations Analyst"
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (clean_username, clean_email, pwd_hash, role, time.time()),
        )
        conn.commit()

    token = create_token(clean_username, role)
    return AuthResponse(
        access_token=token,
        username=clean_username,
        email=clean_email,
        role=role,
        message="Account created successfully",
    )


def authenticate_user(req: LoginRequest) -> AuthResponse:
    init_db()
    identifier = req.username.strip().lower()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, email, password_hash, role FROM users WHERE username = ? OR email = ?",
            (identifier, identifier),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError("Invalid username or password")

        username, email, stored_hash, role = row
        if not verify_password(req.password, stored_hash):
            raise ValueError("Invalid username or password")

    token = create_token(username, role)
    return AuthResponse(
        access_token=token,
        username=username,
        email=email,
        role=role,
        message="Authentication successful",
    )
