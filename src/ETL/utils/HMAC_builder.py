import hashlib
import hmac
import os


def _secret():
    secret = os.getenv("HMAC_SECRET")
    if not secret:
        raise ValueError("HMAC_SECRET must be set for clinical ETL")
    return secret.encode("utf-8")


def patient_hmac(patient_id):
    return "PAT-" + hmac.new(
        _secret(),
        str(patient_id).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]


def deterministic_uid(original_uid, namespace):
    digest = hmac.new(
        _secret(),
        f"{namespace}:{original_uid}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return f"2.25.{int.from_bytes(digest[:16], 'big')}"
