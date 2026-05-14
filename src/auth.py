import hashlib
import os

from src.database import create_user, get_user


def _hash_password(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return salt.hex() + ":" + key.hex()


def _verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, key_hex = stored_hash.split(":")
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return key.hex() == key_hex


def register(username: str, password: str) -> tuple[bool, str]:
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        create_user(username, _hash_password(password))
        return True, "Account created! You can now log in."
    except Exception:
        return False, "Username already taken."


def login(username: str, password: str) -> tuple[bool, int | None, str]:
    row = get_user(username)
    if row is None or not _verify_password(password, row["password_hash"]):
        return False, None, "Invalid username or password."
    return True, row["id"], "Login successful!"
