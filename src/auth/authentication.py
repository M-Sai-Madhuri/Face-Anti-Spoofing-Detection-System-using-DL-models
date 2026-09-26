"""
Authentication Manager - Two-stage identity verification with lockout.

Phase 1 Upgrade: Logs centroid similarity, max individual similarity,
and final decision for enhanced audit trail.
Phase 3 Upgrade: A/B comparison mode logs global vs adaptive threshold
decisions side-by-side for analysis and dashboard display.

Stage 1: Anti-spoof check (must pass before identity check)
Stage 2: Identity matching against registered user centroids
Includes lockout mechanism after consecutive failures.
"""

import time
import logging
import numpy as np
import cv2
from collections import deque
from typing import Dict, Any, Optional

from .face_encoder import FaceEncoder
from .identity_store import IdentityStore
from .attempt_logger import AttemptLogger

logger = logging.getLogger(__name__)


class AuthenticationManager:
    """
    Two-stage biometric authentication:
    1. Anti-spoof validation (existing model, untouched)
    2. Identity verification via centroid matching

    Security features:
    - Lockout after N consecutive failed attempts
    - Attempt logging to CSV
    - Separate thresholds for spoof and identity
    - Centroid + max individual similarity logging
    """

    def __init__(
        self,
        face_encoder,
        identity_store,
        attempt_logger,
        identity_threshold: float = 0.65,
        lockout_attempt_limit: int = 5,
        lockout_duration: int = 30,
        ab_config: Optional[Dict[str, Any]] = None,
        consecutive_required: int = 2,
        auth_blur_threshold: float = 60.0,
    ):
        """
        Args:
            face_encoder:         FaceEncoder instance
            identity_store:       IdentityStore instance
            attempt_logger:       AttemptLogger instance
            identity_threshold:   Minimum cosine similarity for identity match
            lockout_attempt_limit:Consecutive failures before lockout
            lockout_duration:     Lockout duration in seconds
            ab_config:            Optional A/B mode config
            consecutive_required: Number of consecutive successful frame matches
                                  before returning access_granted. Default 2.
                                  Set to 1 to disable temporal consistency.
            auth_blur_threshold:  Laplacian variance below which a live auth
                                  frame is flagged as blurry and skipped.
                                  Default 60.0. Set to 0 to disable.
        """
        self.encoder = face_encoder
        self.store = identity_store
        self.logger = attempt_logger
        self.identity_threshold = identity_threshold
        self.lockout_attempt_limit = lockout_attempt_limit
        self.lockout_duration = lockout_duration

        # Phase 3: A/B comparison config
        self.ab_enabled = (ab_config or {}).get('enabled', False)
        self.ab_global_threshold = (ab_config or {}).get('global_threshold', 0.85)

        # Temporal consistency: require N consecutive matching frames
        self.consecutive_required = max(1, consecutive_required)
        self._consecutive_match_user: Optional[str] = None  # who is matching
        self._consecutive_match_count: int = 0              # frames so far

        # Auth-time quality gate
        self.auth_blur_threshold = auth_blur_threshold

        # Lockout state
        self._consecutive_failures = 0
        self._lockout_start_time: Optional[float] = None

        # Phase 3: cumulative A/B stats (in-memory counters)
        self._ab_stats = {
            'false_reject_prevented': 0,
            'security_tightened':     0,
            'agreement':              0,
            'total_ab_attempts':      0,
        }

    def is_locked_out(self) -> bool:
        """Check if the system is currently locked out."""
        if self._lockout_start_time is None:
            return False

        elapsed = time.time() - self._lockout_start_time
        if elapsed >= self.lockout_duration:
            # Lockout expired
            self._lockout_start_time = None
            self._consecutive_failures = 0
            return False

        return True

    def get_lockout_remaining(self) -> int:
        """Get remaining lockout time in seconds."""
        if self._lockout_start_time is None:
            return 0

        elapsed = time.time() - self._lockout_start_time
        remaining = self.lockout_duration - elapsed
        return max(0, int(remaining))

    def authenticate(
        self,
        face_img: np.ndarray,
        spoof_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform two-stage authentication.

        Args:
            face_img: Cropped face image (BGR numpy array)
            spoof_result: Dict from anti-spoof model with:
                'is_spoof' (bool), 'spoof_prob' (float), 'label' (str), 'confidence' (float)

        Returns:
            Dict with:
                'status': one of 'locked_out', 'spoof_denied', 'access_granted',
                         'unauthorized', 'no_users'
                'message': Human-readable message
                'user': matched username or None
                'confidence': match confidence (centroid similarity)
                'lockout_remaining': seconds remaining if locked out
        """
        # Check lockout first
        if self.is_locked_out():
            remaining = self.get_lockout_remaining()
            return {
                'status': 'locked_out',
                'message': f"System Temporarily Locked ({remaining}s)",
                'user': None,
                'confidence': 0.0,
                'lockout_remaining': remaining,
            }

        # Check if any users are registered
        if self.store.user_count() == 0:
            return {
                'status': 'no_users',
                'message': "No registered users. Please register first.",
                'user': None,
                'confidence': 0.0,
                'lockout_remaining': 0,
            }

        # Stage 1: Anti-spoof check
        spoof_label = spoof_result.get('label', 'SPOOF')
        spoof_confidence = spoof_result.get('confidence', 0.0)
        is_spoof = spoof_result.get('is_spoof', True)

        if is_spoof:
            # Log the spoof attempt
            self.logger.log(
                spoof_label=spoof_label,
                spoof_confidence=spoof_confidence,
                identity_match=False,
                identity_confidence=0.0,
                matched_user=None,
            )

            self._consecutive_failures += 1
            self._check_lockout()

            logger.info(
                f"Auth DENIED (spoof): label={spoof_label}, "
                f"confidence={spoof_confidence:.4f}, "
                f"consecutive_failures={self._consecutive_failures}"
            )

            return {
                'status': 'spoof_denied',
                'message': "Spoof Detected - Access Denied",
                'user': None,
                'confidence': 0.0,
                'lockout_remaining': 0,
            }

        # Stage 2: Identity verification (centroid matching + adaptive threshold)
        # Auth-time quality gate: skip blurry frames before expensive encoding
        if self.auth_blur_threshold > 0:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharpness < self.auth_blur_threshold:
                logger.debug(
                    f"Auth-time blur gate: frame skipped "
                    f"(sharpness={sharpness:.1f} < {self.auth_blur_threshold})"
                )
                return {
                    'status': 'scanning',
                    'message': 'Frame too blurry — hold still',
                    'user': None,
                    'confidence': 0.0,
                    'lockout_remaining': 0,
                }

        try:
            embedding = self.encoder.encode(face_img)
            # match() returns (user, score, details) with per-user adaptive threshold
            matched_user, match_confidence, match_details = self.store.match(
                embedding, threshold=self.identity_threshold
            )
            threshold_used = match_details.get('threshold_used', self.identity_threshold)
            is_adaptive = match_details.get('is_adaptive', False)
        except Exception as e:
            logger.error(f"Identity encoding error: {e}")
            print(f"[Auth] Identity encoding error: {e}")
            matched_user = None
            match_confidence = 0.0
            threshold_used = self.identity_threshold
            is_adaptive = False

        if matched_user is not None:
            # ── Temporal consistency gate ─────────────────────────────────
            # Require N consecutive frames of the SAME user before granting.
            # Resets if a different user (or no user) is matched.
            if self.consecutive_required > 1:
                if matched_user == self._consecutive_match_user:
                    self._consecutive_match_count += 1
                else:
                    # Different user matched — reset counter
                    self._consecutive_match_user  = matched_user
                    self._consecutive_match_count = 1

                if self._consecutive_match_count < self.consecutive_required:
                    remaining = self.consecutive_required - self._consecutive_match_count
                    logger.debug(
                        f"Temporal gate: {self._consecutive_match_count}/{self.consecutive_required} "
                        f"frames for '{matched_user}' — need {remaining} more"
                    )
                    return {
                        'status': 'scanning',
                        'message': f'Verifying identity… ({self._consecutive_match_count}/{self.consecutive_required})',
                        'user': None,
                        'confidence': match_confidence,
                        'lockout_remaining': 0,
                    }

            # N consecutive frames confirmed — reset temporal counter
            self._consecutive_match_count = 0
            self._consecutive_match_user  = None
            # Access granted
            self.logger.log(
                spoof_label=spoof_label,
                spoof_confidence=spoof_confidence,
                identity_match=True,
                identity_confidence=match_confidence,
                matched_user=matched_user,
            )
            self._consecutive_failures = 0
            self._lockout_start_time = None

            # Phase 3: A/B comparison logging
            ab_data = self._run_ab_comparison(match_confidence, threshold_used)

            logger.info(
                f"Auth GRANTED: user={matched_user}, "
                f"centroid_similarity={match_confidence:.4f}, "
                f"threshold={threshold_used:.4f} "
                f"({'adaptive' if is_adaptive else 'global'})"
            )

            result = {
                'status': 'access_granted',
                'message': f"Access Granted: {matched_user}",
                'user': matched_user,
                'confidence': match_confidence,
                'threshold_used': threshold_used,
                'is_adaptive_threshold': is_adaptive,
                'lockout_remaining': 0,
            }
            if ab_data:
                result['ab_comparison'] = ab_data
            return result
        else:
            # No match — reset temporal consistency counter
            self._consecutive_match_user  = None
            self._consecutive_match_count = 0

            # Unauthorized
            self.logger.log(
                spoof_label=spoof_label,
                spoof_confidence=spoof_confidence,
                identity_match=False,
                identity_confidence=match_confidence,
                matched_user=None,
            )

            self._consecutive_failures += 1
            self._check_lockout()

            logger.info(
                f"Auth DENIED (unauthorized): "
                f"best_centroid_similarity={match_confidence:.4f}, "
                f"threshold={threshold_used:.4f} "
                f"({'adaptive' if is_adaptive else 'global'}), "
                f"consecutive_failures={self._consecutive_failures}"
            )

            return {
                'status': 'unauthorized',
                'message': "Unauthorized User",
                'user': None,
                'confidence': match_confidence,
                'lockout_remaining': 0,
            }

    def _check_lockout(self):
        """Trigger lockout if consecutive failures exceed limit."""
        if self._consecutive_failures >= self.lockout_attempt_limit:
            self._lockout_start_time = time.time()
            logger.warning(
                f"LOCKOUT triggered after {self._consecutive_failures} "
                f"consecutive failures. Duration: {self.lockout_duration}s"
            )
            print(f"[Auth] LOCKOUT triggered after {self._consecutive_failures} "
                  f"consecutive failures. Duration: {self.lockout_duration}s")

    def _run_ab_comparison(
        self,
        similarity: float,
        adaptive_threshold: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Phase 3: Compare adaptive vs global threshold decisions.

        Records which threshold would have made a different decision,
        updating in-memory cumulative counters for dashboard display.

        Returns:
            Dict with comparison details, or None if A/B mode disabled.
        """
        if not self.ab_enabled:
            return None

        adaptive_pass = similarity >= adaptive_threshold
        global_pass   = similarity >= self.ab_global_threshold
        same_decision = adaptive_pass == global_pass

        self._ab_stats['total_ab_attempts'] += 1
        if same_decision:
            self._ab_stats['agreement'] += 1
        elif adaptive_pass and not global_pass:
            # Adaptive is LESS strict → prevented a false reject
            self._ab_stats['false_reject_prevented'] += 1
        else:
            # Adaptive is MORE strict → tightened security
            self._ab_stats['security_tightened'] += 1

        ab_data = {
            'similarity': round(similarity, 4),
            'adaptive_threshold': round(adaptive_threshold, 4),
            'global_threshold': round(self.ab_global_threshold, 4),
            'adaptive_decision': 'PASS' if adaptive_pass else 'FAIL',
            'global_decision':   'PASS' if global_pass   else 'FAIL',
            'difference': not same_decision,
        }

        logger.info(
            f"A/B comparison: similarity={similarity:.4f}, "
            f"adaptive={ab_data['adaptive_decision']} (t={adaptive_threshold:.4f}), "
            f"global={ab_data['global_decision']} (t={self.ab_global_threshold:.4f}), "
            f"differs={not same_decision}"
        )
        return ab_data

    def get_ab_stats(self) -> Dict[str, Any]:
        """Return cumulative A/B comparison statistics for dashboard."""
        total = self._ab_stats['total_ab_attempts']
        agreement_rate = (
            round(self._ab_stats['agreement'] / total * 100, 1) if total else 0
        )
        return {
            **self._ab_stats,
            'agreement_rate': agreement_rate,
            'ab_enabled': self.ab_enabled,
        }

    def reset_lockout(self):
        """Manually reset lockout state."""
        self._consecutive_failures = 0
        self._lockout_start_time = None
        logger.info("Lockout manually reset")
