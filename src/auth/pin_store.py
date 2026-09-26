"""
PIN Store — Per-user 4-digit PIN for 2FA.

PINs are stored as salted SHA-256 hashes (never plain text).
Encrypted with Fernet symmetric encryption if available,
with a base64 fallback for environments without cryptography installed.

Design:
- Each user has exactly one PIN.
- Admins can reset any user's PIN.
- Users set their PIN after face registration.
- 2FA flow: face auth → pending_pin state → PIN verify.
"""

import os
import json
import base64
import hashlib
import secrets
import logging
from typing import Dict, Optional

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


def _hash_pin(pin: str, salt: str) -> str:
    """SHA-256 hash of pin+salt. Never store plain PINs."""
    combined = (salt + pin).encode('utf-8')
    return hashlib.sha256(combined).hexdigest()


class PINStore:
    """
    Encrypted per-user PIN storage.
    PINs are 4–8 digit numeric strings stored as salted hashes.
    """

    def __init__(self, store_path: str, key_path: str):
        self.store_path = store_path
        self.key_path   = key_path
        self._pins: Dict[str, Dict] = {}   # username -> {hash, salt}
        self._fernet = None
        self._load_or_init()

    # ─────────────────────────────────────────────────────────────────────────────
    # Init / persistence
    # ─────────────────────────────────────────────────────────────────────────────

    def _load_or_init(self):
        """Load existing PIN store or create fresh one."""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)

        # Encryption key
        if CRYPTO_AVAILABLE:
            if os.path.exists(self.key_path):
                with open(self.key_path, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(self.key_path, 'wb') as f:
                    f.write(key)
            from cryptography.fernet import Fernet as F
            self._fernet = F(key)

        # Load existing PINs
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'rb') as f:
                    raw = f.read()
                if self._fernet:
                    data = self._fernet.decrypt(raw)
                else:
                    data = base64.b64decode(raw)
                self._pins = json.loads(data.decode('utf-8'))
                logger.info(f"[PINStore] Loaded {len(self._pins)} PINs")
            except Exception as e:
                logger.warning(f"[PINStore] Load error: {e} — starting fresh")
                self._pins = {}

    def _save(self):
        """Persist encrypted PIN store to disk."""
        data = json.dumps(self._pins).encode('utf-8')
        if self._fernet:
            raw = self._fernet.encrypt(data)
        else:
            raw = base64.b64encode(data)
        with open(self.store_path, 'wb') as f:
            f.write(raw)

    # ─────────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────────

    def set_pin(self, username: str, pin: str) -> tuple:
        """
        Set or update a user's PIN.

        Args:
            username: Registered username
            pin: 4–8 digit numeric string

        Returns:
            (success: bool, message: str)
        """
        pin = pin.strip()
        if not pin.isdigit() or not (4 <= len(pin) <= 8):
            return False, "PIN must be 4–8 digits"

        salt = secrets.token_hex(16)
        self._pins[username] = {
            'hash': _hash_pin(pin, salt),
            'salt': salt,
        }
        self._save()
        logger.info(f"[PINStore] PIN set for user: {username}")
        return True, "PIN set successfully"

    def verify_pin(self, username: str, pin: str) -> bool:
        """Verify a user's PIN. Returns True if correct."""
        record = self._pins.get(username)
        if not record:
            return False
        expected = _hash_pin(pin.strip(), record['salt'])
        return expected == record['hash']

    def has_pin(self, username: str) -> bool:
        """Check if a user has a PIN set."""
        return username in self._pins

    def remove_pin(self, username: str) -> bool:
        """Remove a user's PIN (admin reset). Returns True if existed."""
        if username in self._pins:
            del self._pins[username]
            self._save()
            logger.info(f"[PINStore] PIN removed for user: {username}")
            return True
        return False

    def list_users_with_pins(self):
        """Return list of usernames that have PINs set."""
        return list(self._pins.keys())
