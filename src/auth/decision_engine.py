"""
Risk-Based Decision Engine — Phase 2

Replaces fixed spoof threshold with weighted multi-signal risk scoring.
Combines anti-spoof confidence, frame stability, brightness, and motion
into a single risk score with configurable bands.

Security decision:
    PASS: All signals strong, proceed to identity check
    REQUIRE_CHALLENGE: Marginal signals, trigger blink challenge (Phase 3)
    FAIL: High risk of spoofing attempt

Falls back to original threshold mode if 'risk_engine_enabled: false' in config.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque

logger = logging.getLogger(__name__)


class RiskDecisionEngine:
    """
    Multi-signal risk scoring engine for face authentication.

    Combines four independent signals into a weighted risk score:
    1. Anti-spoof confidence (primary signal from model)
    2. Frame stability (face position stability over time)
    3. Brightness (lighting condition quality)
    4. Motion consistency (optical flow natural movement check)

    Configurable via config_auth.yaml under 'risk_engine' section.
    """

    # Risk score bands
    BAND_PASS = 'PASS'
    BAND_CHALLENGE = 'REQUIRE_CHALLENGE'
    BAND_FAIL = 'FAIL'

    def __init__(
        self,
        enabled: bool = True,
        weight_spoof: float = 0.50,
        weight_stability: float = 0.20,
        weight_brightness: float = 0.15,
        weight_motion: float = 0.15,
        threshold_pass: float = 0.80,
        threshold_challenge: float = 0.60,
        stability_window: int = 10,
        brightness_min: float = 60.0,
        brightness_max: float = 220.0,
        motion_min: float = 0.2,
        motion_max: float = 8.0,
    ):
        """
        Args:
            enabled: If False, engine is bypassed (fallback mode)
            weight_spoof: Weight for anti-spoof confidence signal
            weight_stability: Weight for frame stability signal
            weight_brightness: Weight for brightness signal
            weight_motion: Weight for motion consistency signal
            threshold_pass: Minimum score for PASS band
            threshold_challenge: Minimum score for REQUIRE_CHALLENGE band
            stability_window: Number of frames to track for stability
            brightness_min: Minimum acceptable brightness (mean grayscale)
            brightness_max: Maximum acceptable brightness
            motion_min: Minimum optical flow magnitude for natural motion
            motion_max: Maximum optical flow magnitude (beyond = unstable)
        """
        self.enabled = enabled

        # Weights (must sum to 1.0)
        self.weight_spoof = weight_spoof
        self.weight_stability = weight_stability
        self.weight_brightness = weight_brightness
        self.weight_motion = weight_motion

        # Band thresholds
        self.threshold_pass = threshold_pass
        self.threshold_challenge = threshold_challenge

        # Scoring parameters
        self.stability_window = stability_window
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.motion_min = motion_min
        self.motion_max = motion_max

        # Internal state for tracking
        self._face_positions: deque = deque(maxlen=stability_window)
        self._prev_gray_frame: Optional[np.ndarray] = None

        logger.info(
            f"RiskDecisionEngine initialized: enabled={enabled}, "
            f"weights=[spoof={weight_spoof}, stability={weight_stability}, "
            f"brightness={weight_brightness}, motion={weight_motion}], "
            f"thresholds=[pass={threshold_pass}, challenge={threshold_challenge}]"
        )

    @classmethod
    def from_config(cls, config: dict) -> 'RiskDecisionEngine':
        """
        Create engine from config dict.

        Expected config structure:
            risk_engine:
                enabled: true
                weights:
                    spoof: 0.50
                    stability: 0.20
                    brightness: 0.15
                    motion: 0.15
                thresholds:
                    pass: 0.80
                    challenge: 0.60
        """
        risk_cfg = config.get('risk_engine', {})
        weights = risk_cfg.get('weights', {})
        thresholds = risk_cfg.get('thresholds', {})

        return cls(
            enabled=risk_cfg.get('enabled', False),
            weight_spoof=weights.get('spoof', 0.50),
            weight_stability=weights.get('stability', 0.20),
            weight_brightness=weights.get('brightness', 0.15),
            weight_motion=weights.get('motion', 0.15),
            threshold_pass=thresholds.get('pass', 0.80),
            threshold_challenge=thresholds.get('challenge', 0.60),
            stability_window=risk_cfg.get('stability_window', 10),
            brightness_min=risk_cfg.get('brightness_min', 60.0),
            brightness_max=risk_cfg.get('brightness_max', 220.0),
            motion_min=risk_cfg.get('motion_min', 0.2),
            motion_max=risk_cfg.get('motion_max', 8.0),
        )

    # =========================================================================
    # Signal scoring functions (each returns 0.0 to 1.0)
    # =========================================================================

    def compute_spoof_score(self, spoof_result: Dict[str, Any]) -> float:
        """
        Convert anti-spoof probability to a confidence score.
        Higher = more likely REAL.

        spoof_prob near 0 = likely real → score near 1.0
        spoof_prob near 1 = likely spoof → score near 0.0
        """
        spoof_prob = spoof_result.get('spoof_prob', 0.5)
        # Invert: low spoof prob = high confidence in realness
        score = 1.0 - spoof_prob
        return max(0.0, min(1.0, score))

    def compute_stability_score(self, face_bbox: tuple) -> float:
        """
        Compute face position stability over recent frames.
        A real face has natural micro-movements; a perfectly still
        image or rapidly moving projection is suspicious.

        Args:
            face_bbox: (x, y, w, h) tuple of current face bounding box

        Returns:
            Score 0.0-1.0 where 1.0 = stable real face movement
        """
        x, y, w, h = face_bbox
        center = (x + w / 2, y + h / 2)
        self._face_positions.append(center)

        if len(self._face_positions) < 3:
            return 0.5  # Not enough data yet

        # Compute standard deviation of face center positions
        positions = np.array(list(self._face_positions))
        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])
        avg_std = (std_x + std_y) / 2.0

        # Sweet spot: natural micro-movement (std ~2-15 pixels)
        # Too still (< 1 px std) = likely static image attack
        # Too jittery (> 30 px std) = likely video/projection attack
        if avg_std < 1.0:
            score = 0.3  # Suspiciously still
        elif avg_std < 3.0:
            score = 0.7  # Slightly too still
        elif avg_std <= 15.0:
            score = 1.0  # Natural movement
        elif avg_std <= 30.0:
            score = 0.6  # Somewhat jittery
        else:
            score = 0.2  # Very jittery / projection

        return score

    def compute_brightness_score(self, face_img: np.ndarray) -> float:
        """
        Score based on face region brightness (mean grayscale).
        Very dark or very bright faces are harder to authenticate reliably.

        Args:
            face_img: Cropped face image (BGR)

        Returns:
            Score 0.0-1.0 where 1.0 = ideal lighting
        """
        import cv2
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        # Ideal range: brightness_min to brightness_max
        if mean_brightness < self.brightness_min:
            # Too dark — linear falloff
            score = max(0.0, mean_brightness / self.brightness_min)
        elif mean_brightness > self.brightness_max:
            # Too bright — linear falloff
            score = max(0.0, 1.0 - (mean_brightness - self.brightness_max) / (255 - self.brightness_max))
        else:
            # In ideal range
            score = 1.0

        return score

    def compute_motion_score(self, frame_gray: np.ndarray) -> float:
        """
        Compute optical flow magnitude to detect natural motion.
        Static images produce zero flow; screen replays can show unusual patterns.

        Args:
            frame_gray: Current frame in grayscale (full frame, not just face)

        Returns:
            Score 0.0-1.0 where 1.0 = natural motion detected
        """
        import cv2

        if self._prev_gray_frame is None:
            self._prev_gray_frame = frame_gray.copy()
            return 0.5  # No data yet

        try:
            # Compute dense optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray_frame, frame_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_motion = float(np.mean(magnitude))
        except Exception:
            avg_motion = 0.0

        self._prev_gray_frame = frame_gray.copy()

        # Score based on motion magnitude
        if avg_motion < self.motion_min:
            score = 0.3  # Too little motion (static image?)
        elif avg_motion <= self.motion_max:
            # Linear scale in acceptable range
            score = 0.7 + 0.3 * (avg_motion - self.motion_min) / (self.motion_max - self.motion_min)
            score = min(1.0, score)
        else:
            # Too much motion (unstable)
            score = max(0.2, 1.0 - (avg_motion - self.motion_max) / 20.0)

        return score

    # =========================================================================
    # Main evaluation
    # =========================================================================

    def evaluate(
        self,
        spoof_result: Dict[str, Any],
        face_bbox: tuple,
        face_img: np.ndarray,
        frame_gray: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate all risk signals and produce a final decision.

        Args:
            spoof_result: Dict from anti-spoof model
            face_bbox: (x, y, w, h) face bounding box
            face_img: Cropped face image (BGR)
            frame_gray: Full frame in grayscale

        Returns:
            Dict with:
                'score': Final weighted score (0.0-1.0)
                'band': 'PASS', 'REQUIRE_CHALLENGE', or 'FAIL'
                'components': {spoof, stability, brightness, motion scores}
                'enabled': Whether engine is active
        """
        if not self.enabled:
            # Fallback mode: use original binary spoof check
            is_spoof = spoof_result.get('is_spoof', True)
            return {
                'score': 0.0 if is_spoof else 1.0,
                'band': self.BAND_FAIL if is_spoof else self.BAND_PASS,
                'components': {
                    'spoof': 0.0 if is_spoof else 1.0,
                    'stability': 1.0,
                    'brightness': 1.0,
                    'motion': 1.0,
                },
                'enabled': False,
            }

        # Compute individual scores
        spoof_score = self.compute_spoof_score(spoof_result)
        stability_score = self.compute_stability_score(face_bbox)
        brightness_score = self.compute_brightness_score(face_img)
        motion_score = self.compute_motion_score(frame_gray)

        # Weighted combination
        final_score = (
            self.weight_spoof * spoof_score +
            self.weight_stability * stability_score +
            self.weight_brightness * brightness_score +
            self.weight_motion * motion_score
        )

        # Determine band
        if final_score >= self.threshold_pass:
            band = self.BAND_PASS
        elif final_score >= self.threshold_challenge:
            band = self.BAND_CHALLENGE
        else:
            band = self.BAND_FAIL

        result = {
            'score': round(final_score, 4),
            'band': band,
            'components': {
                'spoof': round(spoof_score, 4),
                'stability': round(stability_score, 4),
                'brightness': round(brightness_score, 4),
                'motion': round(motion_score, 4),
            },
            'enabled': True,
        }

        logger.info(
            f"Risk evaluation: score={final_score:.4f}, band={band}, "
            f"spoof={spoof_score:.3f}, stability={stability_score:.3f}, "
            f"brightness={brightness_score:.3f}, motion={motion_score:.3f}"
        )

        return result

    def reset(self):
        """Reset internal tracking state (e.g., when face is lost)."""
        self._face_positions.clear()
        self._prev_gray_frame = None
