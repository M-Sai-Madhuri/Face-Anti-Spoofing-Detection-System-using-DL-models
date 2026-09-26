# src/liveness/__init__.py
"""Active Liveness Detection Module (Phase 3)"""

from .blink_detector import BlinkDetector, BlinkChallenge

__all__ = ['BlinkDetector', 'BlinkChallenge']
