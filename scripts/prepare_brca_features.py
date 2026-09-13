#!/usr/bin/env python3
"""Build a breast-only panel with explicit provenance; default is read-only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathotme.brca_features import PANELS, panel_spec, selected_rows, write_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentme-root", required=True, type=Path)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        if args.output is None:
            parser.error("--execute requires --output")
        result = write_panel(args.opentme_root, args.output, args.panel)
    else:
        spec = panel_spec(args.panel)
        rows = selected_rows(args.opentme_root, args.panel)
        result = {"status": "dry_run", "panel": args.panel, "slides": len(rows),
                  "features": len(spec["feature_names"]), "schema_sha256": spec["schema_sha256"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
