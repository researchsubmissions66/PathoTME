#!/usr/bin/env python3
"""Download the licensed OpenTME snapshot from Hugging Face.

The script never stores a token in the project. It uses the token configured by
``huggingface_hub`` for the current user and pins the dataset revision so a
resumed transfer cannot silently mix releases. By default it downloads the
quantitative CSVs, settings, and documentation. Pass ``--include-thumbnails``
to additionally fetch the roughly 28,000 visualization files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPOSITORY = "Aignostics/OpenTME"
DEFAULT_REVISION = "9262bc0cd0cd7774d0f7bcbfe6ae5f4665898b25"
DEFAULT_OUTPUT = Path("/path/to/shared/PathoTME-data/OpenTME")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a reproducible snapshot transfer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--include-thumbnails",
        action="store_true",
        help="also download all per-slide PNG thumbnails",
    )
    return parser.parse_args()


def main() -> None:
    """Download or resume the pinned gated dataset snapshot."""

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    allow_patterns = None
    if not args.include_thumbnails:
        allow_patterns = [
            "*.csv",
            "settings/*",
            "README.md",
            "CHANGELOG.md",
            "OpenTME_User_Guide.pdf",
        ]
    path = snapshot_download(
        repo_id=DEFAULT_REPOSITORY,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        max_workers=args.max_workers,
        token=True,
        allow_patterns=allow_patterns,
    )
    print(f"OpenTME snapshot available at {path}")


if __name__ == "__main__":
    main()
