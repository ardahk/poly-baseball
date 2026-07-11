"""Shared experiment-provenance helpers: canonical JSON, config/code identity.

Every checksum that ties a run, artifact, or preregistration to its inputs
must use the same canonicalization, or provenance comparisons silently break.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path


def canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: dict) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def config_hash(cfg) -> str:
    return digest(asdict(cfg))


def code_revision(root: Path | None = None) -> str:
    override = os.environ.get("POLYBOT_CODE_REVISION")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root or Path(__file__).resolve().parent.parent,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_identity() -> tuple[str, str]:
    """(revision, sha256 of main.py + polybot sources)."""
    root = Path(__file__).resolve().parent.parent
    hasher = hashlib.sha256()
    for path in sorted([root / "main.py", *root.joinpath("polybot").glob("*.py")]):
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())
    return code_revision(root), hasher.hexdigest()
