#!/usr/bin/env python
"""
Generate Integrity Manifest — Phase 5 CLI Tool

Computes SHA256 hashes of all critical files and saves them to
config/integrity_manifest.json. This manifest is used on startup
to detect file tampering.

Usage:
    python generate_manifest.py
    python generate_manifest.py --root /path/to/project
    python generate_manifest.py --verify   # Verify instead of generate
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.security import IntegrityChecker


def main():
    parser = argparse.ArgumentParser(
        description='Generate or verify file integrity manifest'
    )
    parser.add_argument(
        '--root', default=PROJECT_ROOT,
        help='Project root directory (default: script directory)'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify integrity instead of generating manifest'
    )
    parser.add_argument(
        '--output', default=None,
        help='Output manifest path (default: config/integrity_manifest.json)'
    )
    args = parser.parse_args()

    checker = IntegrityChecker(
        project_root=args.root,
        manifest_path=args.output,
    )

    if args.verify:
        print("=" * 60)
        print("  Verifying File Integrity")
        print("=" * 60)

        passed, report = checker.verify()

        print(f"\nFiles checked: {report['checked']}")
        print(f"Mismatches:    {len(report['mismatches'])}")
        print(f"Missing:       {len(report['missing_files'])}")
        print(f"New files:     {len(report['new_files'])}")
        print(f"\nResult: {'PASSED [OK]' if passed else 'FAILED [FAIL]'}")

        if report['new_files']:
            print("\nNew files (not in manifest):")
            for f in report['new_files']:
                print(f"  + {f}")

        sys.exit(0 if passed else 1)
    else:
        print("=" * 60)
        print("  Generating Integrity Manifest")
        print("=" * 60)

        manifest = checker.generate_manifest()
        manifest_path = checker.save_manifest(manifest)

        print(f"\nFiles hashed: {manifest['file_count']}")
        print(f"Manifest:     {manifest_path}")
        print(f"\nHashed files:")
        for filepath in sorted(manifest['hashes'].keys()):
            h = manifest['hashes'][filepath]
            print(f"  {filepath}: {h[:16]}...")

        print(f"\n[OK] Manifest generated successfully")
        print(f"  To verify: python generate_manifest.py --verify")


if __name__ == '__main__':
    main()
