"""
Identity Store - Multi-Embedding Centroid-Based Identity Storage.

Phase 1 Upgrade: Stores multiple embeddings per user plus a centroid vector.
Adaptive Identity Upgrade: Per-user adaptive threshold from embedding statistics.
Uses Fernet symmetric encryption for persistence.
No raw face images are stored — only compact 512-d embeddings.

Backward compatible: auto-migrates old single-embedding format on load.
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("[IdentityStore] WARNING: 'cryptography' not installed. "
          "Using base64 obfuscation instead of encryption. "
          "Install with: pip install cryptography")

logger = logging.getLogger(__name__)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector (or return zero vector if norm is 0)."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def _compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute the L2-normalized centroid (mean) of a list of embeddings.

    The centroid is the average direction of all embeddings, re-normalized
    so cosine similarity comparisons remain valid.
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot compute centroid of empty embedding list")
    stacked = np.stack(embeddings, axis=0)
    centroid = np.mean(stacked, axis=0)
    return _l2_normalize(centroid)


def compute_pairwise_stats(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute pairwise cosine similarity statistics across all embeddings.

    For N L2-normalized embeddings, cosine similarity = dot product.
    We compute the upper triangle of the NxN similarity matrix
    (excluding the diagonal, since self-similarity is always 1.0).

    Returns:
        Dict with 'mean_similarity', 'std_similarity', 'num_embeddings'
    """
    n = len(embeddings)
    if n < 2:
        # With only 1 embedding, no pairwise stats are possible.
        # Return defaults that will produce the global threshold as fallback.
        return {
            'mean_similarity': 1.0,
            'std_similarity': 0.0,
            'num_embeddings': n,
        }

    # Stack into matrix: (N, 512)
    emb_matrix = np.stack(embeddings, axis=0)  # Already L2-normalized

    # Cosine similarity matrix via dot product: (N, N)
    sim_matrix = emb_matrix @ emb_matrix.T

    # Extract upper triangle (excluding diagonal) — these are all unique pairs
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    mean_sim = float(np.mean(pairwise_sims))
    std_sim = float(np.std(pairwise_sims))

    return {
        'mean_similarity': mean_sim,
        'std_similarity': std_sim,
        'num_embeddings': n,
    }


class UserRecord:
    """
    In-memory representation of a registered user.

    Attributes:
        embeddings:          List of L2-normalized 512-d face embeddings
        centroid:            L2-normalized mean of all embeddings (used for matching)
        adaptive_threshold:  Per-user identity threshold computed from embedding stats
                             None = use global fallback threshold
        stats:               Pairwise similarity statistics used to compute threshold
    """

    def __init__(
        self,
        embeddings: List[np.ndarray],
        centroid: Optional[np.ndarray] = None,
        adaptive_threshold: Optional[float] = None,
        stats: Optional[Dict[str, float]] = None,
    ):
        self.embeddings = [e.copy() for e in embeddings]
        if centroid is not None:
            self.centroid = centroid.copy()
        else:
            self.centroid = _compute_centroid(self.embeddings)
        self.adaptive_threshold = adaptive_threshold
        self.stats = stats or {}

    @staticmethod
    def from_single_embedding(embedding: np.ndarray) -> 'UserRecord':
        """
        Backward compatibility: create a UserRecord from a single embedding.
        adaptive_threshold is set to None → signals fallback to global threshold.
        """
        emb = _l2_normalize(embedding.copy())
        return UserRecord(
            embeddings=[emb],
            centroid=emb.copy(),
            adaptive_threshold=None,  # Will use global threshold
            stats={'mean_similarity': 1.0, 'std_similarity': 0.0, 'num_embeddings': 1},
        )

    @property
    def embedding_count(self) -> int:
        return len(self.embeddings)


class IdentityStore:
    """
    Manages storage, retrieval, and matching of registered user embeddings.

    Storage format (v3 — adaptive threshold):
        Each user stores:
        {
            "embeddings": [list of base64-encoded 512-d vectors],
            "centroid": base64-encoded 512-d centroid vector,
            "adaptive_threshold": float or null,
            "stats": {mean_similarity, std_similarity, num_embeddings},
            "dtype": "float32",
            "version": 3
        }

    Backward compatibility:
        v1 (single embedding) and v2 (multi-embedding without threshold)
        are auto-migrated on load.
    """

    def __init__(
        self,
        store_path: str = "data/auth/users.enc",
        key_path: str = "data/auth/auth.key",
        max_users: int = 5,
    ):
        """
        Args:
            store_path: Path to encrypted user database file
            key_path: Path to encryption key file
            max_users: Maximum number of registered users
        """
        self.store_path = Path(store_path)
        self.key_path = Path(key_path)
        self.max_users = max_users

        # In-memory database: {username: UserRecord}
        self._users: Dict[str, UserRecord] = {}

        # Ensure directories exist
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or generate encryption key
        self._key = self._load_or_create_key()

        # Load existing users (with auto-migration)
        self.load()

    # =========================================================================
    # Encryption
    # =========================================================================

    def _load_or_create_key(self) -> Optional[bytes]:
        """Load existing key or generate a new one."""
        if not CRYPTO_AVAILABLE:
            return None

        if self.key_path.exists():
            with open(self.key_path, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.key_path, 'wb') as f:
                f.write(key)
            logger.info(f"Generated new encryption key: {self.key_path}")
            return key

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data using Fernet."""
        if CRYPTO_AVAILABLE and self._key:
            f = Fernet(self._key)
            return f.encrypt(data)
        else:
            return base64.b64encode(data)

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data using Fernet."""
        if CRYPTO_AVAILABLE and self._key:
            f = Fernet(self._key)
            return f.decrypt(data)
        else:
            return base64.b64decode(data)

    # =========================================================================
    # Serialization helpers
    # =========================================================================

    @staticmethod
    def _embedding_to_b64(emb: np.ndarray) -> str:
        """Serialize a numpy embedding to base64 string."""
        return base64.b64encode(emb.astype(np.float32).tobytes()).decode('ascii')

    @staticmethod
    def _b64_to_embedding(b64: str) -> np.ndarray:
        """Deserialize a base64 string to numpy embedding."""
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.float32).copy()

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self):
        """Persist users to encrypted file in v3 (adaptive threshold) format."""
        serialized = {}
        for username, record in self._users.items():
            serialized[username] = {
                'version': 3,
                'dtype': 'float32',
                'centroid': self._embedding_to_b64(record.centroid),
                'embeddings': [
                    self._embedding_to_b64(emb) for emb in record.embeddings
                ],
                'adaptive_threshold': record.adaptive_threshold,
                'stats': record.stats,
            }

        json_bytes = json.dumps(serialized).encode('utf-8')
        encrypted = self._encrypt(json_bytes)

        with open(self.store_path, 'wb') as f:
            f.write(encrypted)

        logger.info(f"Saved {len(self._users)} user(s) to {self.store_path}")
        print(f"[IdentityStore] Saved {len(self._users)} user(s) to {self.store_path}")

    def load(self):
        """
        Load users from encrypted file.
        Auto-migrates v1 (single embedding) and v2 (no threshold) to v3 format.
        """
        if not self.store_path.exists():
            self._users = {}
            return

        try:
            with open(self.store_path, 'rb') as f:
                encrypted = f.read()

            json_bytes = self._decrypt(encrypted)
            serialized = json.loads(json_bytes.decode('utf-8'))

            self._users = {}
            migrated = False

            for username, data in serialized.items():
                version = data.get('version', 1)

                if version >= 3:
                    # --- v3 format: multi-embedding + centroid + adaptive threshold ---
                    embeddings = [self._b64_to_embedding(b) for b in data['embeddings']]
                    centroid = self._b64_to_embedding(data['centroid'])
                    self._users[username] = UserRecord(
                        embeddings=embeddings,
                        centroid=centroid,
                        adaptive_threshold=data.get('adaptive_threshold'),
                        stats=data.get('stats', {}),
                    )
                elif version >= 2:
                    # --- v2 format: multi-embedding + centroid (no threshold) ---
                    embeddings = [self._b64_to_embedding(b) for b in data['embeddings']]
                    centroid = self._b64_to_embedding(data['centroid'])
                    # Compute adaptive threshold from existing embeddings
                    stats = compute_pairwise_stats(embeddings)
                    adaptive_t = self._compute_adaptive_threshold(stats)
                    self._users[username] = UserRecord(
                        embeddings=embeddings,
                        centroid=centroid,
                        adaptive_threshold=adaptive_t,
                        stats=stats,
                    )
                    migrated = True
                    logger.info(
                        f"Auto-migrated user '{username}' from v2 → v3 "
                        f"(adaptive_threshold={adaptive_t:.4f})"
                    )
                else:
                    # --- v1 format: single embedding (backward compat) ---
                    emb_bytes = base64.b64decode(data['embedding'])
                    shape = tuple(data['shape'])
                    dtype = np.dtype(data['dtype'])
                    single_emb = np.frombuffer(emb_bytes, dtype=dtype).reshape(shape).copy()
                    single_emb = _l2_normalize(single_emb)
                    self._users[username] = UserRecord.from_single_embedding(single_emb)
                    migrated = True
                    logger.info(f"Auto-migrated user '{username}' from v1 → v3 format")

            if migrated:
                logger.info("Saving migrated user database in v3 format...")
                self.save()

            user_info = ", ".join(
                f"{u}({r.embedding_count}emb, t={r.adaptive_threshold or 'global'})"
                for u, r in self._users.items()
            )
            print(f"[IdentityStore] Loaded {len(self._users)} user(s): {user_info}")

        except Exception as e:
            logger.error(f"Failed to load users: {e}")
            print(f"[IdentityStore] WARNING: Failed to load users: {e}")
            self._users = {}

    # =========================================================================
    # User management
    # =========================================================================

    @staticmethod
    def _compute_adaptive_threshold(
        stats: Dict[str, float],
        std_multiplier: float = 2.0,
        min_threshold: float = 0.75,
        max_threshold: float = 0.90,
    ) -> Optional[float]:
        """
        Compute per-user adaptive identity threshold from embedding statistics.

        Formula: threshold = mean_similarity - (std_multiplier * std_similarity)
        Clamped to [min_threshold, max_threshold] for safety.

        Statistical reasoning:
        - Users with highly consistent embeddings (low std) get a higher
          threshold, meaning they require a tighter match → more secure.
        - Users with more variable embeddings (high std) get a lower
          threshold, reducing false rejections for that individual.
        - The clamp ensures no user ever has an unreasonably loose or
          tight threshold, maintaining a security floor.
        """
        if stats.get('num_embeddings', 0) < 2:
            return None  # Not enough data → use global threshold

        mean_sim = stats['mean_similarity']
        std_sim = stats['std_similarity']

        # threshold = mean - k*std → lower bound of "normal" similarity for this user
        raw_threshold = mean_sim - (std_multiplier * std_sim)

        # Clamp to safe bounds
        clamped = max(min_threshold, min(max_threshold, raw_threshold))

        logger.debug(
            f"Adaptive threshold: mean={mean_sim:.4f}, std={std_sim:.4f}, "
            f"raw={raw_threshold:.4f}, clamped={clamped:.4f}"
        )
        return clamped

    def add_user(
        self,
        username: str,
        embeddings: Union[np.ndarray, List[np.ndarray]],
        adaptive_config: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """
        Register a new user with one or more face embeddings.

        Args:
            username: Unique username
            embeddings: Single 512-d embedding (backward compat) OR
                        list of 512-d embeddings (multi-embedding mode)
            adaptive_config: Optional dict with 'std_multiplier', 'min_threshold',
                             'max_threshold' for adaptive threshold computation

        Returns:
            (success: bool, message: str)
        """
        if len(self._users) >= self.max_users:
            return False, f"User limit reached ({self.max_users}). Cannot register."

        if username in self._users:
            return False, f"Username '{username}' already exists."

        if len(username.strip()) == 0:
            return False, "Username cannot be empty."

        # Handle both single embedding and list of embeddings
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            # Single embedding (backward compatible)
            emb_list = [_l2_normalize(embeddings.copy())]
            logger.info(f"Registering '{username}' with 1 embedding (single mode)")
        elif isinstance(embeddings, list):
            emb_list = [_l2_normalize(e.copy()) for e in embeddings]
            logger.info(f"Registering '{username}' with {len(emb_list)} embeddings (centroid mode)")
        elif isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
            emb_list = [_l2_normalize(embeddings[i]) for i in range(embeddings.shape[0])]
            logger.info(f"Registering '{username}' with {len(emb_list)} embeddings (centroid mode)")
        else:
            return False, f"Invalid embedding format: {type(embeddings)}"

        if len(emb_list) == 0:
            return False, "No embeddings provided."

        # Compute pairwise similarity statistics
        stats = compute_pairwise_stats(emb_list)

        # Compute adaptive threshold from stats
        cfg = adaptive_config or {}
        adaptive_threshold = self._compute_adaptive_threshold(
            stats,
            std_multiplier=cfg.get('std_multiplier', 2.0),
            min_threshold=cfg.get('min_threshold', 0.75),
            max_threshold=cfg.get('max_threshold', 0.90),
        )

        record = UserRecord(
            embeddings=emb_list,
            adaptive_threshold=adaptive_threshold,
            stats=stats,
        )
        self._users[username] = record
        self.save()

        logger.info(
            f"Registered '{username}': {len(emb_list)} embeddings, "
            f"centroid norm={np.linalg.norm(record.centroid):.4f}, "
            f"adaptive_threshold={adaptive_threshold}, "
            f"mean_sim={stats['mean_similarity']:.4f}, "
            f"std_sim={stats['std_similarity']:.4f}"
        )
        print(
            f"[IdentityStore] Registered '{username}': "
            f"{len(emb_list)} embeddings, "
            f"threshold={adaptive_threshold or 'global'}"
        )
        return True, f"User '{username}' registered with {len(emb_list)} embedding(s)."

    def remove_user(self, username: str) -> Tuple[bool, str]:
        """Remove a registered user."""
        if username not in self._users:
            return False, f"User '{username}' not found."

        del self._users[username]
        self.save()
        logger.info(f"Removed user '{username}'")
        return True, f"User '{username}' removed."

    def rename_user(self, old_name: str, new_name: str) -> Tuple[bool, str]:
        """Rename a registered user (preserves all embeddings and centroid)."""
        if old_name not in self._users:
            return False, f"User '{old_name}' not found."

        new_name = new_name.strip()
        if not new_name:
            return False, "New username cannot be empty."

        if new_name in self._users:
            return False, f"Username '{new_name}' already exists."

        self._users[new_name] = self._users.pop(old_name)
        self.save()
        logger.info(f"Renamed user '{old_name}' → '{new_name}'")
        return True, f"User '{old_name}' renamed to '{new_name}'."

    def list_users(self) -> List[str]:
        """Return list of registered usernames."""
        return list(self._users.keys())

    def user_count(self) -> int:
        """Return number of registered users."""
        return len(self._users)

    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get info about a registered user."""
        if username not in self._users:
            return None
        record = self._users[username]
        return {
            'username': username,
            'embedding_count': record.embedding_count,
            'centroid_norm': float(np.linalg.norm(record.centroid)),
            'adaptive_threshold': record.adaptive_threshold,
            'stats': record.stats,
        }

    def get_user_threshold(self, username: str) -> Optional[float]:
        """Get the adaptive threshold for a user, or None if not set."""
        if username not in self._users:
            return None
        return self._users[username].adaptive_threshold

    # =========================================================================
    # Matching
    # =========================================================================

    def match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.65,
        hybrid_alpha: float = 0.6,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Match an embedding against all registered users using a hybrid scorer:

            score = alpha * centroid_sim + (1 - alpha) * max_individual_sim

        Rationale:
            - Centroid gives a stable average direction for the user.
            - Max individual catches frames where the query is close to ONE of
              the stored poses even if the centroid has drifted.
            - Weighting 60/40 gives centroid priority while keeping max_individual
              as a meaningful tie-breaker.

        The threshold is applied to the HYBRID score for each user.
        The user with the highest hybrid score above threshold wins.

        Args:
            embedding:     512-d face embedding to match
            threshold:     Global fallback threshold
            hybrid_alpha:  Weight given to centroid score (0–1). Default 0.6.

        Returns:
            (matched_username or None, hybrid_score, match_details)
            match_details keys: threshold_used, is_adaptive, max_individual_sim,
                                centroid_sim, hybrid_score
        """
        empty_details = {
            'threshold_used': threshold, 'is_adaptive': False,
            'max_individual_sim': 0.0, 'centroid_sim': 0.0, 'hybrid_score': 0.0,
        }
        if not self._users:
            return None, 0.0, empty_details

        embedding = _l2_normalize(embedding)

        best_user           = None
        best_hybrid_score   = -1.0
        best_centroid_sim   = -1.0
        best_max_individual = -1.0
        best_threshold_used = threshold
        best_is_adaptive    = False

        for username, record in self._users.items():
            # ── Component 1: cosine sim vs centroid ──────────────────────
            centroid_sim = float(np.dot(embedding, record.centroid))

            # ── Component 2: max cosine sim vs individual stored frames ──
            # Vectorised: (512,) · (N, 512)^T = (N,) similarities
            emb_matrix = np.stack(record.embeddings, axis=0)   # (N, 512)
            sims = emb_matrix @ embedding                       # (N,)
            max_individual = float(np.max(sims))

            # ── Hybrid score ──────────────────────────────────────────────
            hybrid = hybrid_alpha * centroid_sim + (1.0 - hybrid_alpha) * max_individual

            # ── Per-user threshold ────────────────────────────────────────
            if record.adaptive_threshold is not None:
                user_threshold = record.adaptive_threshold
                is_adaptive = True
            else:
                user_threshold = threshold
                is_adaptive = False

            logger.debug(
                f"  [{username}] centroid={centroid_sim:.4f}, "
                f"max_individual={max_individual:.4f}, "
                f"hybrid={hybrid:.4f}, threshold={user_threshold:.4f}"
            )

            if hybrid > best_hybrid_score:
                best_hybrid_score   = hybrid
                best_centroid_sim   = centroid_sim
                best_max_individual = max_individual
                best_user           = username
                best_threshold_used = user_threshold
                best_is_adaptive    = is_adaptive

        decision = 'MATCH' if best_hybrid_score >= best_threshold_used else 'NO_MATCH'

        logger.debug(
            f"Match result: user={best_user}, "
            f"hybrid={best_hybrid_score:.4f}, "
            f"centroid={best_centroid_sim:.4f}, "
            f"max_individual={best_max_individual:.4f}, "
            f"threshold={best_threshold_used:.4f} "
            f"({'adaptive' if best_is_adaptive else 'global'}), "
            f"decision={decision}"
        )
        print(
            f"[IdentityStore] Match: {best_user} | "
            f"hybrid={best_hybrid_score:.4f} "
            f"(c={best_centroid_sim:.4f}, ind={best_max_individual:.4f}) | "
            f"threshold={best_threshold_used:.4f} "
            f"({'adaptive' if best_is_adaptive else 'global'}) | {decision}"
        )

        details = {
            'threshold_used':    best_threshold_used,
            'is_adaptive':       best_is_adaptive,
            'max_individual_sim': best_max_individual,
            'centroid_sim':      best_centroid_sim,
            'hybrid_score':      best_hybrid_score,
        }

        if best_hybrid_score >= best_threshold_used:
            return best_user, best_hybrid_score, details
        else:
            return None, best_hybrid_score, details


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    store = IdentityStore(
        store_path="data/auth/test_users.enc",
        key_path="data/auth/test_auth.key",
        max_users=3,
    )

    # Test multi-embedding registration
    embeddings = []
    for _ in range(15):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        embeddings.append(emb)

    ok, msg = store.add_user("test_user", embeddings)
    print(f"Add: {ok}, {msg}")

    info = store.get_user_info("test_user")
    print(f"User info: {info}")

    # Test match against centroid
    user, conf = store.match(embeddings[0], threshold=0.3)
    print(f"Match: user={user}, confidence={conf:.4f}")

    # Test backward compat: single embedding
    single_emb = np.random.randn(512).astype(np.float32)
    single_emb /= np.linalg.norm(single_emb)
    ok, msg = store.add_user("single_user", single_emb)
    print(f"Add single: {ok}, {msg}")

    # Cleanup
    store.remove_user("test_user")
    store.remove_user("single_user")
    print("Done")
