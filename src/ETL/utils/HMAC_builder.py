import hashlib
import hmac
import os


HMAC_SECRET=os.environ["HMAC_SECRET"]


def patient_hmac(patient_id):
    return "PAT-"+hmac.new(
        HMAC_SECRET.encode("utf-8"),
        str(patient_id).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]

def deterministic_uid(original_uid, namespace):
    digest=hmac.new(
        HMAC_SECRET.encode("utf-8"),
        f"{namespace}:{original_uid}".encode("utf-8"),
        hashlib.sha256,
    ).digest()

    value=int.from_bytes(digest[:16], "big")

    return f"2.25.{value}"