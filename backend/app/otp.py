# backend/app/otp.py
"""
OTP utilities: generation, Redis storage, validation helpers,
and e-mail delivery for the Streamlit-only build.
"""

from __future__ import annotations

import random
import re
from typing import Tuple, Optional

import redis
import streamlit as st

# backend/app/otp.py  – top of file
from .email_utils import send_via_sendgrid


# --------------------------------------------------------------------------- #
# Redis & constants
# --------------------------------------------------------------------------- #

DEFAULT_TTL_SECONDS = 300  # 5 minutes

# Connect to your cloud Redis (e.g., Upstash) via Streamlit secrets
redis_client = redis.from_url(st.secrets["REDIS_URL"], decode_responses=True)

# Regex patterns
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
OTP_REGEX = re.compile(r"\b(\d{4,8})\b")  # 4- to 8-digit number


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #

def generate_otp(length: int = 6) -> str:
    """Return a random numeric OTP (default 6 digits, zero-padded)."""
    return "".join(str(random.randint(0, 9)) for _ in range(length))


def store_otp(email: str, otp: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
    """Store *otp* under key ``otp:<email>`` with a configurable TTL."""
    redis_client.setex(f"otp:{email}", ttl_seconds, otp)


def retrieve_stored_otp(email: str) -> Optional[str]:
    """Return the stored OTP for *email* (or ``None`` if expired/missing)."""
    return redis_client.get(f"otp:{email}")


def delete_otp(email: str) -> None:
    """Remove the OTP for *email* – called after successful verification."""
    redis_client.delete(f"otp:{email}")


def verify_otp(email: str, otp: str) -> bool:
    """
    Return ``True`` if *otp* matches the stored value for *email*.
    The stored OTP is deleted on a successful match.
    """
    stored = retrieve_stored_otp(email)
    if stored and stored == otp:
        delete_otp(email)
        return True
    return False


# --------------------------------------------------------------------------- #
# Validation / extraction helpers
# --------------------------------------------------------------------------- #

def is_valid_email(text: str) -> bool:
    """Return ``True`` if *text* looks exactly like an e-mail address."""
    return EMAIL_REGEX.fullmatch(text.strip()) is not None


def find_email(text: str) -> Optional[str]:
    """Return the first e-mail token inside *text*, or ``None``."""
    match = EMAIL_REGEX.search(text)
    return match.group(0) if match else None


def extract_otp(text: str, digits: int = 6) -> Optional[str]:
    """Return the first *digits*-long number found in *text*, else ``None``."""
    pattern = re.compile(rf"\b(\d{{{digits}}})\b")
    match = pattern.search(text)
    return match.group(1) if match else None


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #

def generate_and_send_otp(
    email: str, ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> Tuple[bool, str]:
    """
    1. Generate an OTP.
    2. Store it in Redis.
    3. Send it to *email* via SendGrid.

    Returns ``(True, "…")`` on success, else ``(False, "error …")``.
    """
    otp_code = generate_otp()
    store_otp(email, otp_code, ttl_seconds=ttl_seconds)

    sent = send_via_sendgrid(
        to_email=email,
        subject="Your OTP Code",
        plain_text=f"Your 6-digit OTP is: {otp_code}\n\nIt expires in five minutes."
    )
    if sent:
        return True, "OTP sent successfully!"
    return False, f"Failed to send OTP to {email}."
