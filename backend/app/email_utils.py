"""
SendGrid-based e-mail helpers for the Streamlit-only build.

Exposed helpers
---------------
• send_via_sendgrid(to_email, subject, plain_text, html_content=None)
• send_otp_email(receiver_email, otp)          – thin wrapper for OTP flow
• send_plain_email(receiver_email, subject, body)

All credentials come from st.secrets:
    SENDGRID_API_KEY   –  your SendGrid API key
    EMAIL_SENDER       –  a verified sender address in SendGrid
"""

from __future__ import annotations

import logging
from typing import Final, Optional

import streamlit as st
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Client & sender pulled from Streamlit secrets
# --------------------------------------------------------------------------- #

SENDGRID_CLIENT: Final = SendGridAPIClient(st.secrets["SENDGRID_API_KEY"])
SENDER_EMAIL:     Final = st.secrets.get("EMAIL_SENDER", "no-reply@example.com")

# --------------------------------------------------------------------------- #
#  Low-level wrapper
# --------------------------------------------------------------------------- #

def _send(mail: Mail) -> bool:
    """
    Actually call SendGrid; return True iff status code is 2xx.
    Logs non-2xx or raised exceptions.
    """
    try:
        resp = SENDGRID_CLIENT.send(mail)
        ok = 200 <= resp.status_code < 300
        if not ok:
            logger.warning(
                "SendGrid returned %s (%s)",
                resp.status_code,
                resp.body.decode() if resp.body else "no body"
            )
        return ok
    except Exception as exc:                           # pragma: no cover
        logger.error("SendGrid exception: %s", exc, exc_info=True)
        return False

# --------------------------------------------------------------------------- #
#  Public helpers
# --------------------------------------------------------------------------- #

def send_via_sendgrid(
    to_email: str,
    subject: str,
    plain_text: str,
    html_content: Optional[str] = None
) -> bool:
    """
    General-purpose one-shot sender.  *html_content* falls back to
    a simple <br>-converted version of *plain_text*.
    """
    mail = Mail(
        from_email=SENDER_EMAIL,
        to_emails=to_email,
        subject=subject,
        plain_text_content=plain_text,
        html_content=html_content or plain_text.replace("\n", "<br>")
    )
    return _send(mail)


def send_otp_email(receiver_email: str, otp: str) -> bool:
    """
    Convenience wrapper for the OTP flow (kept for backward compatibility).
    """
    body = (
        "Hello,\n\n"
        f"Your OTP code is: {otp}\n\n"
        "This code expires in 5 minutes.\n\n"
        "If you didn’t request this, please ignore the e-mail.\n\n"
        "Best regards,\nHistorical Monument Agent"
    )
    return send_via_sendgrid(
        to_email=receiver_email,
        subject="Your OTP Code",
        plain_text=body
    )


def send_plain_email(receiver_email: str, subject: str, body: str) -> bool:
    """
    Generic helper for non-OTP messages (e.g. detailed monument guide).
    """
    return send_via_sendgrid(
        to_email=receiver_email,
        subject=subject,
        plain_text=body
    )
