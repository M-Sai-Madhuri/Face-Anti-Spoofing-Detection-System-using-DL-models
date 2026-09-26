"""
Registration Manager - Handles user registration with anti-spoof gate.

Phase 1 Upgrade: Supports both single-frame and multi-frame registration.
Multi-frame mode collects multiple face embeddings for centroid-based storage.
"""

import time
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

# ── Quality-gate constants ───────────────────────────────────────────────────
# Frames that fail these gates are dropped BEFORE encoding.
# This prevents blurry / tiny-face captures from corrupting the centroid.
MIN_SHARPNESS      = 80.0   # Laplacian variance threshold — lower = blurrier
MIN_FACE_AREA_FRAC = 0.003  # Face must cover at least 0.3% of frame area
MAX_FRAMES_KEPT    = 30     # Hard cap on frames sent to encoder (avoid OOM)
# ─────────────────────────────────────────────────────────────────────────────


def _frame_quality_score(face_img: np.ndarray) -> Dict[str, float]:
    """
    Compute sharpness and relative face size for a cropped face image.

    Sharpness: Laplacian variance — captures high-frequency energy lost by blur.
    Face area: height*width of the crop relative to expected minimum frame size.

    Returns:
        {'sharpness': float, 'face_area': int, 'passes': bool}
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    face_area = face_img.shape[0] * face_img.shape[1]
    # Approximate frame area (assumes faces are cropped from ≥640×480 frame)
    approx_frame_area = 640 * 480
    area_frac = face_area / approx_frame_area
    passes = sharpness >= MIN_SHARPNESS and area_frac >= MIN_FACE_AREA_FRAC
    return {'sharpness': sharpness, 'face_area': face_area,
            'area_frac': area_frac, 'passes': passes}

from .face_encoder import FaceEncoder
from .identity_store import IdentityStore

logger = logging.getLogger(__name__)


class RegistrationManager:
    """
    Manages user registration flow:
    1. Check anti-spoof result (must be REAL)
    2. Encode face(s) to 512-d embedding(s)
    3. Store in encrypted identity database

    Supports two modes:
    - Single-frame: register(username, face_img, spoof_result)
    - Multi-frame:  register_multi(username, face_imgs, spoof_results)
    """

    def __init__(
        self,
        face_encoder: FaceEncoder,
        identity_store: IdentityStore,
        spoof_threshold: float = 0.50,
        min_embeddings: int = 15,
    ):
        """
        Args:
            face_encoder: FaceEncoder instance for embedding extraction
            identity_store: IdentityStore instance for user persistence
            spoof_threshold: Maximum spoof probability to allow registration
            min_embeddings: Minimum embeddings for multi-frame registration
        """
        self.encoder = face_encoder
        self.store = identity_store
        self.spoof_threshold = spoof_threshold
        self.min_embeddings = min_embeddings

    def register(
        self,
        username: str,
        face_img: np.ndarray,
        spoof_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Register a new user with a single face image.
        Backward compatible with existing callers.

        Args:
            username: Desired username
            face_img: Cropped face image (BGR numpy array)
            spoof_result: Dict from anti-spoof model

        Returns:
            Dict with 'success', 'message', 'username'
        """
        # Stage 1: Anti-spoof gate
        if spoof_result.get('is_spoof', True):
            return {
                'success': False,
                'message': "Spoof Detected - Try Again",
                'username': None,
            }

        # Check user limit
        if self.store.user_count() >= self.store.max_users:
            return {
                'success': False,
                'message': f"User Limit Reached ({self.store.max_users})",
                'username': None,
            }

        # Validate username
        username = username.strip()
        if not username:
            return {
                'success': False,
                'message': "Username cannot be empty",
                'username': None,
            }

        # Stage 2: Encode face
        try:
            embedding = self.encoder.encode(face_img)
        except Exception as e:
            logger.error(f"Failed to encode face: {e}")
            return {
                'success': False,
                'message': f"Failed to encode face: {e}",
                'username': None,
            }

        # Stage 3: Store embedding (single)
        success, msg = self.store.add_user(username, embedding)
        logger.info(f"Single-frame registration: '{username}' -> {msg}")
        return {
            'success': success,
            'message': msg,
            'username': username if success else None,
        }

    def register_multi(
        self,
        username: str,
        face_imgs: List[np.ndarray],
        spoof_results: Optional[List[Dict[str, Any]]] = None,
        adaptive_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a new user with multiple face images (centroid mode).

        Encodes all face images, computes pairwise embedding statistics,
        and stores the full embedding set + computed centroid + adaptive threshold.

        Args:
            username: Desired username
            face_imgs: List of cropped face images (BGR numpy arrays)
            spoof_results: Optional list of spoof results per frame
                           (if provided, only REAL frames are used)
            adaptive_config: Optional dict with 'enabled', 'std_multiplier',
                             'min_threshold', 'max_threshold'

        Returns:
            Dict with 'success', 'message', 'username', 'embedding_count',
                      'adaptive_threshold', 'stats'
        """
        # Validate username
        username = username.strip()
        if not username:
            return {
                'success': False,
                'message': "Username cannot be empty",
                'username': None,
                'embedding_count': 0,
            }

        # Check user limit
        if self.store.user_count() >= self.store.max_users:
            return {
                'success': False,
                'message': f"User Limit Reached ({self.store.max_users})",
                'username': None,
                'embedding_count': 0,
            }

        if username in self.store.list_users():
            return {
                'success': False,
                'message': f"Username '{username}' already exists",
                'username': None,
                'embedding_count': 0,
            }

        # Filter by spoof results if provided
        valid_imgs = []
        if spoof_results is not None:
            for img, sr in zip(face_imgs, spoof_results):
                if not sr.get('is_spoof', True):
                    valid_imgs.append(img)
            logger.info(
                f"Multi-frame registration: {len(valid_imgs)}/{len(face_imgs)} "
                f"frames passed anti-spoof gate"
            )
        else:
            valid_imgs = list(face_imgs)

        if len(valid_imgs) < self.min_embeddings:
            return {
                'success': False,
                'message': (
                    f"Not enough REAL frames ({len(valid_imgs)}/{self.min_embeddings}). "
                    f"Hold face steady longer."
                ),
                'username': None,
                'embedding_count': len(valid_imgs),
            }

        # ── Quality gate ───────────────────────────────────────────────────
        # Discard blurry / tiny-face frames BEFORE encoding.
        # A single low-quality frame dragging the centroid can cause false rejects
        # and false accepts for similar-looking users.
        quality_passed = []
        quality_dropped = 0
        for img in valid_imgs:
            q = _frame_quality_score(img)
            if q['passes']:
                quality_passed.append(img)
            else:
                quality_dropped += 1
                logger.debug(
                    f"Quality gate: dropped frame — "
                    f"sharpness={q['sharpness']:.1f} (min={MIN_SHARPNESS}), "
                    f"area_frac={q['area_frac']:.4f} (min={MIN_FACE_AREA_FRAC})"
                )

        logger.info(
            f"Quality gate: {len(quality_passed)}/{len(valid_imgs)} frames kept "
            f"({quality_dropped} dropped — blur/size)"
        )
        print(
            f"[Registration] Quality gate: {len(quality_passed)}/{len(valid_imgs)} frames "
            f"passed ({quality_dropped} dropped for blur/small face)"
        )

        # If quality gate drops too many frames, fall back to all valid_imgs
        # (ensures registration always succeeds even in poor lighting)
        if len(quality_passed) < self.min_embeddings:
            logger.warning(
                f"Quality gate too strict — only {len(quality_passed)} frames passed. "
                f"Falling back to all {len(valid_imgs)} anti-spoof-passed frames."
            )
            print(
                f"[Registration] WARNING: Quality gate dropped too many frames. "
                f"Using all {len(valid_imgs)} frames (no quality filter)."
            )
            quality_passed = valid_imgs

        # Hard cap to avoid memory issues with very long registrations
        if len(quality_passed) > MAX_FRAMES_KEPT:
            # Keep the sharpest frames, not just the first N
            scored = sorted(
                quality_passed,
                key=lambda img: cv2.Laplacian(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var(),
                reverse=True,
            )
            quality_passed = scored[:MAX_FRAMES_KEPT]
            logger.info(f"Kept top-{MAX_FRAMES_KEPT} sharpest frames (was {len(valid_imgs)})")

        valid_imgs = quality_passed
        # ──────────────────────────────────────────────────────────────────

        # Single forward pass over all N frames is ~5-10× faster than N
        # individual calls, because model-invocation overhead is amortised.
        embeddings = []
        batch_time = None
        try:
            t0 = time.perf_counter()
            embeddings = self.encoder.encode_batch(valid_imgs)
            batch_time = time.perf_counter() - t0

            # Reference estimate: time one frame × N
            t_single = time.perf_counter()
            _ = self.encoder.encode(valid_imgs[0])
            est_sequential = (time.perf_counter() - t_single) * len(valid_imgs)

            speedup = est_sequential / batch_time if batch_time > 0 else 1.0
            logger.info(
                f"Batch encoding: {len(valid_imgs)} frames in {batch_time*1000:.1f}ms "
                f"(est. sequential: {est_sequential*1000:.1f}ms, "
                f"speedup: {speedup:.1f}×)"
            )
            print(
                f"[Registration] Batch encoded {len(valid_imgs)} frames in "
                f"{batch_time*1000:.1f}ms  (≈{speedup:.1f}× faster than sequential)"
            )
        except Exception as e:
            logger.warning(f"Batch encoding failed ({e}), falling back to sequential")
            print(f"[Registration] WARNING: Batch encoding failed, using sequential fallback")
            embeddings = []
            for i, img in enumerate(valid_imgs):
                try:
                    emb = self.encoder.encode(img)
                    embeddings.append(emb)
                except Exception as enc_err:
                    logger.warning(f"Failed to encode frame {i}: {enc_err}")
        # ─────────────────────────────────────────────────────────────────

        if len(embeddings) < self.min_embeddings:
            return {
                'success': False,
                'message': (
                    f"Only {len(embeddings)} frames encoded successfully "
                    f"(need {self.min_embeddings}). Try again."
                ),
                'username': None,
                'embedding_count': len(embeddings),
            }

        # Store with centroid computation + adaptive threshold
        # Pass adaptive_config so add_user can compute per-user threshold
        success, msg = self.store.add_user(
            username, embeddings, adaptive_config=adaptive_config
        )

        # Retrieve the computed threshold for logging
        user_info = self.store.get_user_info(username) if success else None
        adaptive_threshold = user_info.get('adaptive_threshold') if user_info else None
        stats = user_info.get('stats', {}) if user_info else {}

        logger.info(
            f"Multi-frame registration: '{username}' -> {msg} "
            f"({len(embeddings)} embeddings, "
            f"adaptive_threshold={adaptive_threshold})"
        )
        return {
            'success': success,
            'message': msg,
            'username': username if success else None,
            'embedding_count': len(embeddings),
            'adaptive_threshold': adaptive_threshold,
            'stats': stats,
        }
