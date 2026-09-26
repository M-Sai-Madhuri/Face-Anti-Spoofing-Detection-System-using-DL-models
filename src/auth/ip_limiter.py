"""
IP Rate Limiter — in-memory, no external dependencies.

Tracks failed authentication attempts per IP address.
After exceeding the limit within the window, the IP is blocked
for the configured duration. Admins can manually unblock IPs
via the admin dashboard.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IPRateLimiter:
    """
    Thread-safe in-memory IP rate limiter for authentication endpoints.

    Each IP gets its own sliding window of failed attempt timestamps.
    If fails_in_window >= max_fails, the IP is blocked for block_duration_s.
    Blocked IPs can be:
      - Auto-unblocked after block_duration_s (if auto_expire=True)
      - Manually unblocked by admin via unblock()
    """

    def __init__(
        self,
        max_fails: int = 5,
        window_s: int = 60,
        block_duration_s: int = 300,
        auto_expire: bool = True,
    ):
        """
        Args:
            max_fails:        Number of failed attempts allowed in window_s before blocking.
            window_s:         Rolling window in seconds for counting failures.
            block_duration_s: How long to block a breaching IP (0 = permanent until manual unblock).
            auto_expire:      If True, blocks auto-expire after block_duration_s.
        """
        self.max_fails        = max_fails
        self.window_s         = window_s
        self.block_duration_s = block_duration_s
        self.auto_expire      = auto_expire

        # ip -> list of failure timestamps (unix seconds)
        self._fails: Dict[str, List[float]] = {}
        # ip -> block_start_timestamp (None = not blocked)
        self._blocked: Dict[str, Optional[float]] = {}
        # ip -> reason string
        self._block_reason: Dict[str, str] = {}

        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def is_blocked(self, ip: str) -> Tuple[bool, int]:
        """
        Check if the IP is currently blocked.

        Returns:
            (blocked: bool, seconds_remaining: int)
        """
        with self._lock:
            return self._check_blocked(ip)

    def record_failure(self, ip: str) -> Tuple[bool, int]:
        """
        Record a failed auth attempt from this IP.
        If the failure pushes the IP over the limit, it gets blocked.

        Returns:
            (now_blocked: bool, fail_count_in_window: int)
        """
        now = time.time()
        with self._lock:
            # Prune old failures outside the window
            self._fails.setdefault(ip, [])
            self._fails[ip] = [t for t in self._fails[ip] if now - t < self.window_s]
            self._fails[ip].append(now)

            count = len(self._fails[ip])
            logger.debug(f"[IPRateLimiter] {ip}: {count}/{self.max_fails} fails in window")

            if count >= self.max_fails:
                if ip not in self._blocked or self._blocked[ip] is None:
                    self._blocked[ip] = now
                    self._block_reason[ip] = f"{count} failed attempts in {self.window_s}s"
                    logger.warning(
                        f"[IPRateLimiter] IP BLOCKED: {ip} "
                        f"({count} fails in {self.window_s}s)"
                    )
                return True, count

            return False, count

    def record_success(self, ip: str) -> None:
        """
        Record a successful auth — reset the failure window for this IP.
        Does NOT unblock a manually-blocked IP.
        """
        with self._lock:
            self._fails[ip] = []
            # Auto-unblock on success if it was rate-blocked (not admin-blocked)
            blocked, _ = self._check_blocked(ip)
            if blocked and self.auto_expire:
                # Only clear auto-expired blocks; manual blocks stay
                pass
        logger.debug(f"[IPRateLimiter] {ip}: success — failure window reset")

    def unblock(self, ip: str) -> bool:
        """Manually unblock an IP. Returns True if it was blocked."""
        with self._lock:
            if ip in self._blocked:
                self._blocked[ip] = None
                self._fails[ip] = []
                logger.info(f"[IPRateLimiter] Admin manually unblocked: {ip}")
                return True
            return False

    def get_blocked_ips(self) -> List[Dict]:
        """Return list of currently blocked IPs with block info."""
        now = time.time()
        result = []
        with self._lock:
            for ip, start in self._blocked.items():
                if start is None:
                    continue
                elapsed = now - start
                remaining = max(0, int(self.block_duration_s - elapsed))
                still_blocked = not self.auto_expire or elapsed < self.block_duration_s
                if still_blocked:
                    result.append({
                        'ip': ip,
                        'blocked_at': start,
                        'blocked_since_s': int(elapsed),
                        'remaining_s': remaining,
                        'reason': self._block_reason.get(ip, 'rate limit'),
                        'fail_count': len(self._fails.get(ip, [])),
                        'auto_expires': self.auto_expire,
                    })
        return sorted(result, key=lambda x: x['blocked_at'], reverse=True)

    def get_stats(self) -> Dict:
        """Return summary stats for the admin panel."""
        now = time.time()
        with self._lock:
            total_tracked = len(self._fails)
            total_blocked = sum(
                1 for ip, start in self._blocked.items()
                if start is not None and (
                    not self.auto_expire or now - start < self.block_duration_s
                )
            )
            total_fails = sum(
                len([t for t in ts if now - t < self.window_s])
                for ts in self._fails.values()
            )
        return {
            'tracked_ips': total_tracked,
            'blocked_ips': total_blocked,
            'fails_in_window': total_fails,
            'max_fails': self.max_fails,
            'window_s': self.window_s,
            'block_duration_s': self.block_duration_s,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers (must be called with lock held)
    # ─────────────────────────────────────────────────────────────────────────

    def _check_blocked(self, ip: str) -> Tuple[bool, int]:
        """Internal: check block status (lock must be held by caller)."""
        start = self._blocked.get(ip)
        if start is None:
            return False, 0
        if self.auto_expire:
            elapsed = time.time() - start
            if elapsed >= self.block_duration_s:
                # Auto-expired
                self._blocked[ip] = None
                self._fails[ip] = []
                return False, 0
            return True, max(0, int(self.block_duration_s - elapsed))
        return True, -1   # -1 = permanent (manual unblock required)
