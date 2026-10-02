#!/usr/bin/env python3
"""Prepare immutable private core62 tables for all four TCGA cohorts on CPU."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathotme.shared_panel import COHORT_SOURCES, common_spec, write_panel


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opentme-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        print(json.dumps(dict(panel=common_spec(), cohorts=list(COHORT_SOURCES), status='plan')))
    else:
        records = []
        for cohort in COHORT_SOURCES:
            metadata = write_panel(a.opentme_root, cohort, a.output_root/cohort/'core62.csv')
            records.append(dict(cohort=cohort, slides=metadata['slide_count'],
                feature_schema_sha256=metadata['spec']['feature_schema_sha256'],
                selected_table_sha256=metadata['selected_table_sha256']))
        report = dict(status='materialized_not_training_integrated', common_spec=common_spec(), records=records)
        with (a.output_root/'MANIFEST.json').open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        print(json.dumps(report))
