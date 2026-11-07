"""Authentication helpers backed by the shared PostgreSQL passenger table."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import insert, or_, select
from sqlalchemy.exc import IntegrityError

try:
    from api.storage import engine, initialize_schema, passengers
except ImportError:
    from storage import engine, initialize_schema, passengers

SECRET_KEY = os.getenv("JWT_SECRET_KEY") or secrets.token_hex(32)


def init_db() -> None:
    initialize_schema()


def hash_password(password: str) -> str:
    """Store only a salted PBKDF2-HMAC-SHA256 password digest."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return salt + "$" + digest.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest_hex = stored_hash.split("$", 1)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def create_token(username: str, role: str) -> str:
    payload = {"sub": username, "role": role, "exp": int(time.time()) + 7 * 86400}
    payload_part = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
    signature_part = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{payload_part}.{signature_part}"


def decode_token(token: str) -> Optional[dict]:
    try:
        payload_part, signature_part = token.split(".", 1)
        expected = hmac.new(SECRET_KEY.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
        supplied = base64.urlsafe_b64decode(signature_part + "=" * (-len(signature_part) % 4))
        if not hmac.compare_digest(expected, supplied):
            return None
        payload = json.loads(base64.urlsafe_b64decode(payload_part + "=" * (-len(payload_part) % 4)))
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
    def validate_email_format(cls, value: str) -> str:
        clean = value.strip().lower()
        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", clean):
            raise ValueError("Invalid email address format")
        return clean


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=100)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    full_name: str
    email: str
    role: str
    message: str


def _response(username: str, full_name: str, email: str, role: str, message: str) -> AuthResponse:
    return AuthResponse(
        access_token=create_token(username, role),
        username=username,
        full_name=full_name,
        email=email,
        role=role,
        message=message,
    )


def register_user(req: SignUpRequest) -> AuthResponse:
    full_name = req.username.strip()
    username = full_name.lower()
    email = req.email.strip().lower()
    role = "Mobility Operations Analyst"

    with engine.connect() as conn:
        existing = conn.execute(
            select(passengers).where(or_(passengers.c.username == username, passengers.c.email == email))
        ).mappings().first()

    # A retry after a dropped signup response safely resumes only the exact same account.
    if existing:
        if existing["username"] == username and existing["email"] == email and verify_password(req.password, existing["password_hash"]):
            return _response(existing["username"], existing["full_name"], existing["email"], existing["role"], "Account ready")
        raise ValueError("Username or email is already registered")

    password_hash = hash_password(req.password)
    try:
        with engine.begin() as conn:
            conn.execute(insert(passengers).values(
                username=username,
                full_name=full_name,
                email=email,
                password_hash=password_hash,
                role=role,
            ))
    except IntegrityError:
        raise ValueError("Username or email is already registered")

    return _response(username, full_name, email, role, "Account created successfully")


def authenticate_user(req: LoginRequest) -> AuthResponse:
    identifier = req.username.strip().lower()
    with engine.connect() as conn:
        row = conn.execute(
            select(passengers).where(
                or_(passengers.c.username == identifier, passengers.c.email == identifier)
            )
        ).mappings().first()

    if not row or not verify_password(req.password, row["password_hash"]):
        raise ValueError("Invalid username or password")
    return _response(row["username"], row["full_name"], row["email"], row["role"], "Authentication successful")
