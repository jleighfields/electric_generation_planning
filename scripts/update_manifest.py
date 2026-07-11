#!/usr/bin/env python
"""Regenerate the Posit Connect Cloud deploy artifacts.

Rebuilds both:
  - requirements.txt  (uv export of the runtime dependency set)
  - manifest.json     (rsconnect, trimmed to the runtime file set)

rsconnect bundles every tracked file and ignores .gitignore, so the
``EXCLUDES`` list below is what keeps dev tooling, docs, and the reference
PDFs out of the deploy bundle. This script is the single home for that
exclude list — edit it here, not in CLAUDE.md / README.

Run from anywhere:  uv run python scripts/update_manifest.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

# Repo root (this file lives in scripts/).
ROOT = Path(__file__).resolve().parent.parent

# Files/dirs to keep OUT of the Connect Cloud bundle — dev tooling, docs, and
# reference material the running app never touches. Directory names exclude
# everything beneath them.
EXCLUDES = [
    "refs",
    ".claude",
    "tests",
    "docker",
    "scripts",
    "CLAUDE.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
]

# Cache dirs rsconnect would otherwise bundle if left on disk.
CACHE_DIRS = ("__pycache__", ".ruff_cache", ".pytest_cache")


def run(cmd: list[str]) -> None:
    """Run a command from the repo root, raising on a non-zero exit.

    Args:
        cmd: The command and its arguments.
    """
    print("==>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def clean_caches() -> None:
    """Delete build/test cache directories so they don't land in the bundle."""
    for name in CACHE_DIRS:
        for path in ROOT.rglob(name):
            if ".venv" not in path.parts and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    """Regenerate requirements.txt and manifest.json, then print the file list."""
    clean_caches()

    run(
        [
            "uv",
            "export",
            "--no-hashes",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "-o",
            "requirements.txt",
        ]
    )

    exclude_args = [arg for pattern in EXCLUDES for arg in ("-x", pattern)]
    run(
        [
            "uv",
            "run",
            "rsconnect",
            "write-manifest",
            "shiny",
            ".",
            "--entrypoint",
            "app",
            "--overwrite",
            *exclude_args,
        ]
    )

    files = sorted(json.loads((ROOT / "manifest.json").read_text())["files"])
    print(f"\nmanifest.json lists {len(files)} files:")
    for name in files:
        print("  ", name)


if __name__ == "__main__":
    main()
