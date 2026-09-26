"""
Integrity Checker — Phase 5: Tamper Detection

Computes SHA256 hashes of critical system files on startup and compares
them against a trusted manifest (config/integrity_manifest.json).

Security rationale:
    - Prevents execution if model weights, configs, or source code
      have been tampered with after deployment.
    - Detects unauthorized modifications to anti-spoof models,
      authentication logic, or configuration files.
    - Logs all mismatches as potential intrusion attempts.

The manifest is generated once via generate_manifest.py and should
be treated as read-only in production.
"""

import hashlib
import json
import logging
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntegrityChecker:
    """
    Verifies file integrity by comparing SHA256 hashes against a trusted manifest.

    Usage:
        checker = IntegrityChecker(project_root="/path/to/project")
        passed, report = checker.verify()
        if not passed:
            print("INTEGRITY COMPROMISED")
            for item in report['mismatches']:
                print(f"  TAMPERED: {item['file']}")
    """

    # Files and patterns to check
    DEFAULT_PATTERNS = [
        'checkpoints/**/*.pth',
        'config/config_auth.yaml',
        'src/models/*.py',
        'src/auth/*.py',
        'src/liveness/*.py',
        'src/security/*.py',
        'web_app.py',
    ]

    MANIFEST_FILENAME = 'config/integrity_manifest.json'

    def __init__(
        self,
        project_root: str,
        manifest_path: Optional[str] = None,
        patterns: Optional[List[str]] = None,
    ):
        """
        Args:
            project_root: Absolute path to the project root directory
            manifest_path: Path to the manifest JSON file (default: config/integrity_manifest.json)
            patterns: Glob patterns for files to check (default: DEFAULT_PATTERNS)
        """
        self.project_root = Path(project_root)
        self.manifest_path = Path(
            manifest_path or self.project_root / self.MANIFEST_FILENAME
        )
        self.patterns = patterns or self.DEFAULT_PATTERNS

        self._manifest: Optional[Dict] = None
        self._integrity_passed: Optional[bool] = None

        logger.info(
            f"IntegrityChecker initialized: root={self.project_root}, "
            f"manifest={self.manifest_path}"
        )

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def collect_files(self) -> List[str]:
        """
        Collect all files matching the configured patterns.

        Returns:
            List of relative file paths (using forward slashes)
        """
        files = set()
        for pattern in self.patterns:
            full_pattern = str(self.project_root / pattern)
            for match in glob.glob(full_pattern, recursive=True):
                if os.path.isfile(match):
                    rel = os.path.relpath(match, self.project_root)
                    files.add(rel.replace('\\', '/'))
        return sorted(files)

    def generate_manifest(self) -> Dict:
        """
        Generate a fresh integrity manifest from current files.

        Returns:
            Dict with file hashes and metadata
        """
        files = self.collect_files()
        hashes = {}

        for rel_path in files:
            abs_path = str(self.project_root / rel_path)
            try:
                file_hash = self.compute_sha256(abs_path)
                hashes[rel_path] = file_hash
                logger.debug(f"Hash: {rel_path} -> {file_hash[:16]}...")
            except Exception as e:
                logger.warning(f"Could not hash {rel_path}: {e}")

        from datetime import datetime
        manifest = {
            'version': 1,
            'generated_at': datetime.now().isoformat(),
            'file_count': len(hashes),
            'patterns': self.patterns,
            'hashes': hashes,
        }

        logger.info(f"Generated manifest with {len(hashes)} file hashes")
        return manifest

    def save_manifest(self, manifest: Optional[Dict] = None) -> str:
        """
        Generate and save the integrity manifest.

        Args:
            manifest: Pre-computed manifest dict (if None, generates fresh)

        Returns:
            Path to saved manifest file
        """
        if manifest is None:
            manifest = self.generate_manifest()

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Manifest saved to {self.manifest_path}")
        print(f"[Integrity] Manifest saved: {self.manifest_path} "
              f"({manifest['file_count']} files)")
        return str(self.manifest_path)

    def load_manifest(self) -> Optional[Dict]:
        """Load the trusted manifest from disk."""
        if not self.manifest_path.exists():
            logger.warning(f"Manifest not found: {self.manifest_path}")
            print(f"[Integrity] WARNING: Manifest not found at {self.manifest_path}. "
                  f"Run 'python generate_manifest.py' to create it.")
            return None

        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            self._manifest = manifest
            logger.info(
                f"Loaded manifest: {manifest.get('file_count', 0)} files, "
                f"generated {manifest.get('generated_at', 'unknown')}"
            )
            return manifest
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return None

    def verify(self) -> Tuple[bool, Dict]:
        """
        Verify file integrity against the trusted manifest.

        Returns:
            (passed: bool, report: Dict)

            report contains:
                'passed': bool
                'checked': int (files checked)
                'mismatches': list of {file, expected, actual, status}
                'missing_files': list of files in manifest but not on disk
                'new_files': list of files on disk but not in manifest
                'message': str
        """
        manifest = self.load_manifest()

        if manifest is None:
            report = {
                'passed': True,  # No manifest = skip check (first run)
                'checked': 0,
                'mismatches': [],
                'missing_files': [],
                'new_files': [],
                'message': 'No manifest found — integrity check skipped. '
                           'Run generate_manifest.py to enable.',
            }
            self._integrity_passed = True
            return True, report

        stored_hashes = manifest.get('hashes', {})
        current_files = self.collect_files()

        mismatches = []
        missing_files = []
        checked = 0

        # Check each file in manifest
        for rel_path, expected_hash in stored_hashes.items():
            abs_path = str(self.project_root / rel_path)

            if not os.path.exists(abs_path):
                missing_files.append(rel_path)
                logger.warning(f"MISSING: {rel_path}")
                continue

            actual_hash = self.compute_sha256(abs_path)
            checked += 1

            if actual_hash != expected_hash:
                mismatches.append({
                    'file': rel_path,
                    'expected': expected_hash[:16] + '...',
                    'actual': actual_hash[:16] + '...',
                    'status': 'TAMPERED',
                })
                logger.error(
                    f"INTEGRITY MISMATCH: {rel_path} "
                    f"expected={expected_hash[:16]}... "
                    f"actual={actual_hash[:16]}..."
                )

        # Check for new files not in manifest
        new_files = [f for f in current_files if f not in stored_hashes]

        passed = len(mismatches) == 0 and len(missing_files) == 0
        self._integrity_passed = passed

        if passed:
            message = f"Integrity check PASSED ({checked} files verified)"
            logger.info(message)
            print(f"[Integrity] [OK] {message}")
        else:
            message = (
                f"INTEGRITY CHECK FAILED: "
                f"{len(mismatches)} tampered, "
                f"{len(missing_files)} missing"
            )
            logger.critical(message)
            print(f"[Integrity] [FAIL] {message}")
            for m in mismatches:
                print(f"  TAMPERED: {m['file']}")
            for f in missing_files:
                print(f"  MISSING:  {f}")

        report = {
            'passed': passed,
            'checked': checked,
            'mismatches': mismatches,
            'missing_files': missing_files,
            'new_files': new_files,
            'message': message,
        }

        return passed, report

    @property
    def integrity_passed(self) -> Optional[bool]:
        """Returns the result of the last verify() call, or None if not yet run."""
        return self._integrity_passed
