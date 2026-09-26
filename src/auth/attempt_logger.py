"""
Attempt Logger - Logs all authentication attempts to CSV.
Each entry records the timestamp, spoof detection result,
identity match result, and confidence scores.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class AttemptLogger:
    """
    Logs authentication attempts to a CSV file.

    Columns:
        timestamp, spoof_label, spoof_confidence,
        identity_match, identity_confidence, matched_user
    """

    FIELDNAMES = [
        'timestamp',
        'spoof_label',
        'spoof_confidence',
        'identity_match',
        'identity_confidence',
        'matched_user',
    ]

    def __init__(self, log_path: str = "data/auth/attempts.csv"):
        """
        Args:
            log_path: Path to the CSV log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with header if it doesn't exist
        if not self.log_path.exists():
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
            print(f"[AttemptLogger] Created log file: {self.log_path}")

    def log(
        self,
        spoof_label: str,
        spoof_confidence: float,
        identity_match: bool = False,
        identity_confidence: float = 0.0,
        matched_user: Optional[str] = None,
    ):
        """
        Log a single authentication attempt.

        Args:
            spoof_label: 'REAL' or 'SPOOF'
            spoof_confidence: Confidence of spoof detection
            identity_match: Whether identity was matched
            identity_confidence: Confidence of identity match
            matched_user: Username if matched, else None
        """
        row = {
            'timestamp': datetime.now().isoformat(),
            'spoof_label': spoof_label,
            'spoof_confidence': f"{spoof_confidence:.4f}",
            'identity_match': str(identity_match),
            'identity_confidence': f"{identity_confidence:.4f}",
            'matched_user': matched_user or "",
        }

        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row)
        except Exception as e:
            print(f"[AttemptLogger] WARNING: Failed to log attempt: {e}")


if __name__ == "__main__":
    logger = AttemptLogger("data/auth/test_attempts.csv")
    logger.log("REAL", 0.15, True, 0.89, "john")
    logger.log("SPOOF", 0.92, False, 0.0, None)
    logger.log("REAL", 0.20, False, 0.45, None)
    print(f"Logged 3 test attempts to {logger.log_path}")
