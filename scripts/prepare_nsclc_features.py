#!/usr/bin/env python3
"""Prepare the pinned, non-redistributable OpenTME NSCLC core panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathotme.features import write_nsclc_feature_table


PINNED_OPENTME_REVISION = "9262bc0cd0cd7774d0f7bcbfe6ae5f4665898b25"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opentme-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--revision", default=PINNED_OPENTME_REVISION,
        help="pinned Hugging Face dataset revision")
    args = parser.parse_args()
    metadata = write_nsclc_feature_table(
        args.opentme_root, args.output, source_revision=args.revision)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
