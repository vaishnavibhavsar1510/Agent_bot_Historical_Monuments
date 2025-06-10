# backend/app/email_otp.py
"""
Utility helpers to generate/send OTPs and verify them.

You can invoke this file directly (`python email_otp.py`) to test the
end-to-end OTP e-mail flow from the command line.
"""

from __future__ import annotations

import logging

from otp import generate_otp, store_otp, verify_otp           # local modules
from email_utils import send_otp_email

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def generate_and_send_otp(email: str) -> tuple[bool, str]:
    """
    1. Generate a 6-digit OTP
    2. Store it in Redis (5-min TTL by default)
    3. Send it via SendGrid

    Returns ``(True, "Success message")`` or ``(False, "Error message")``.
    """
    otp = generate_otp()
    store_otp(email, otp)

    sent = send_otp_email(email, otp)
    if sent:
        return True, "OTP sent successfully!"
    return False, "Failed to send OTP."


def verify_user_otp(email: str, user_input_otp: str) -> bool:
    """
    Check the user-supplied *user_input_otp* against the stored value
    in Redis.  Returns ``True`` on match (and deletes the stored OTP),
    ``False`` otherwise.
    """
    return verify_otp(email, user_input_otp)


# --------------------------------------------------------------------------- #
# Simple CLI tester
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("OTP Email Testing System")
    print("-" * 32)

    recipient = input("Recipient e-mail: ").strip()

    ok, msg = generate_and_send_otp(recipient)
    print(msg)
    if not ok:
        raise SystemExit(1)

    print("\nOTP Verification")
    print("-" * 32)
    user_code = input("Enter the OTP you received: ").strip()

    if verify_user_otp(recipient, user_code):
        print("OTP verified successfully!")
    else:
        print("Invalid or expired OTP.")
