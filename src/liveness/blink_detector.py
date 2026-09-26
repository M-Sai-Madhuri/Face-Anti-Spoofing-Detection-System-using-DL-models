"""
Blink Detector — Phase 3: Active Liveness Challenge

Uses OpenCV Haar Cascade eye detector for blink detection.
Provides a BlinkChallenge class that requires N blinks within a timeout
to pass the liveness check.

Security rationale:
    - Static photos cannot blink
    - Video replays have predictable blink patterns
    - Debounce logic prevents false positives from noise

Does NOT modify the anti-spoof model — operates as an independent gate.
"""

import cv2
import time
import logging
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class BlinkDetector:
    """
    Detects eye blinks using OpenCV Haar Cascade eye detector.

    Tracks whether eyes are visible frame-to-frame. A blink is
    detected when eyes disappear and then reappear (closed → open).
    No MediaPipe or external ML dependencies required.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,   # kept for API compatibility, unused
        debounce_frames: int = 3,
    ):
        self.debounce_frames = debounce_frames

        # Load bundled OpenCV eye cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self._eye_cascade = cv2.CascadeClassifier(cascade_path)
        if self._eye_cascade.empty():
            raise RuntimeError(f"Failed to load eye cascade from {cascade_path}")

        # State
        self._eyes_were_closed = False
        self._frames_since_blink = 0
        self._blink_count = 0
        self._blink_timestamps: List[float] = []
        self._last_ear = 0.3  # placeholder for API compat

        logger.info("BlinkDetector initialised with OpenCV Haar cascade (no MediaPipe)")

    def process_frame(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and detect blinks.

        Args:
            frame_rgb: Frame in RGB format (H, W, 3)

        Returns:
            Dict with:
                'ear': placeholder (always 0.3 open / 0.1 closed)
                'blink_detected': Whether a blink just happened
                'blink_count': Total blinks detected since reset
                'eye_closed': Whether eyes appear closed
        """
        self._frames_since_blink += 1

        result = {
            'ear': self._last_ear,
            'blink_detected': False,
            'blink_count': self._blink_count,
            'eye_closed': False,
        }

        # Convert to BGR for OpenCV cascade
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces first to narrow the ROI
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) == 0:
            return result

        # Use only the largest face
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        # Look for eyes only in the upper half of the face
        eye_roi = gray[fy:fy + fh // 2, fx:fx + fw]
        eyes = self._eye_cascade.detectMultiScale(eye_roi, 1.1, 6, minSize=(20, 20))

        eyes_visible = len(eyes) >= 2
        eye_closed = not eyes_visible
        self._last_ear = 0.1 if eye_closed else 0.3
        result['eye_closed'] = eye_closed
        result['ear'] = self._last_ear

        if eye_closed and not self._eyes_were_closed:
            # Transition: open → closed
            self._eyes_were_closed = True

        elif not eye_closed and self._eyes_were_closed:
            # Transition: closed → open = BLINK
            if self._frames_since_blink >= self.debounce_frames:
                self._blink_count += 1
                self._blink_timestamps.append(time.time())
                self._frames_since_blink = 0
                result['blink_detected'] = True
                logger.debug(f"Blink #{self._blink_count} detected")
            self._eyes_were_closed = False

        result['blink_count'] = self._blink_count
        return result

    def reset(self):
        """Reset blink counter and state."""
        self._eyes_were_closed = False
        self._frames_since_blink = 0
        self._blink_count = 0
        self._blink_timestamps.clear()
        self._last_ear = 0.3

    @property
    def blink_count(self) -> int:
        return self._blink_count

    @property
    def blink_timestamps(self) -> List[float]:
        return list(self._blink_timestamps)

    @property
    def last_ear(self) -> float:
        return self._last_ear


class BlinkChallenge:
    """
    Active liveness challenge: user must blink N times within a timeout.

    Usage:
        challenge = BlinkChallenge(required_blinks=2, timeout=5.0)
        challenge.start()

        while challenge.is_active():
            result = detector.process_frame(frame)
            outcome = challenge.update(result)
            if outcome is not None:
                # outcome['passed'] = True/False
                break
    """

    def __init__(
        self,
        required_blinks: int = 2,
        timeout: float = 5.0,
        ear_threshold: float = 0.21,
        debounce_frames: int = 3,
    ):
        """
        Args:
            required_blinks: Number of blinks needed to pass
            timeout: Seconds allowed to complete challenge
            ear_threshold: EAR threshold for blink detection
            debounce_frames: Minimum frames between blinks
        """
        self.required_blinks = required_blinks
        self.timeout = timeout

        self.detector = BlinkDetector(
            ear_threshold=ear_threshold,
            debounce_frames=debounce_frames,
        )

        self._active = False
        self._start_time: Optional[float] = None
        self._outcome: Optional[Dict[str, Any]] = None

        logger.info(
            f"BlinkChallenge created: required={required_blinks} blinks, "
            f"timeout={timeout}s"
        )

    @classmethod
    def from_config(cls, config: dict) -> 'BlinkChallenge':
        """Create from config dict (liveness section)."""
        liveness = config.get('liveness', {})
        return cls(
            required_blinks=liveness.get('blink_count', 2),
            timeout=liveness.get('challenge_timeout', 5.0),
            ear_threshold=liveness.get('ear_threshold', 0.21),
            debounce_frames=liveness.get('debounce_frames', 3),
        )

    def start(self):
        """Start the challenge (reset state and begin timer)."""
        self.detector.reset()
        self._active = True
        self._start_time = time.time()
        self._outcome = None
        logger.info("Blink challenge started")

    def is_active(self) -> bool:
        """Check if challenge is still in progress."""
        return self._active

    def get_remaining_time(self) -> float:
        """Seconds remaining in the challenge."""
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        return max(0.0, self.timeout - elapsed)

    def update(self, frame_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a frame during the challenge.

        Args:
            frame_rgb: Frame in RGB format

        Returns:
            None if challenge still active, or Dict with:
                'passed': True/False
                'blinks_detected': int
                'message': str
                'ear_values': list of EAR samples
                'blink_timestamps': list
        """
        if not self._active:
            return self._outcome

        # Check timeout
        elapsed = time.time() - self._start_time
        if elapsed >= self.timeout:
            self._active = False
            self._outcome = {
                'passed': False,
                'blinks_detected': self.detector.blink_count,
                'required_blinks': self.required_blinks,
                'message': (
                    f"Challenge failed: {self.detector.blink_count}/"
                    f"{self.required_blinks} blinks in {self.timeout}s"
                ),
                'blink_timestamps': self.detector.blink_timestamps,
            }
            logger.warning(f"Blink challenge FAILED (timeout): {self._outcome['message']}")
            return self._outcome

        # Process frame
        blink_result = self.detector.process_frame(frame_rgb)

        # Check if enough blinks
        if self.detector.blink_count >= self.required_blinks:
            self._active = False
            self._outcome = {
                'passed': True,
                'blinks_detected': self.detector.blink_count,
                'required_blinks': self.required_blinks,
                'message': (
                    f"Challenge passed: {self.detector.blink_count} blinks "
                    f"in {elapsed:.1f}s"
                ),
                'blink_timestamps': self.detector.blink_timestamps,
            }
            logger.info(f"Blink challenge PASSED: {self._outcome['message']}")
            return self._outcome

        return None  # Still in progress

    @property
    def outcome(self) -> Optional[Dict[str, Any]]:
        return self._outcome
