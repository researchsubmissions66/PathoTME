#!/usr/bin/env python3
"""Aggregate matched held-out PathoTME attribution exports without a model."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathotme.attribution_io import aggregate_results, digest, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    pgvl = Path(os.environ.get('PGVL_ROOT', ROOT.parent / 'PGVL-Gym')).resolve()
    if any(output == root or root in output.parents for root in (ROOT, pgvl)):
        parser.error('Write patient artifacts outside the source repository')
    documents = [json.loads(p.read_text()) for p in args.inputs]
    result = aggregate_results(documents)
    result['source_sha256'] = {str(p): digest(p) for p in args.inputs}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(output / 'patient_scores.csv', result['patient_scores'])
    write_csv(output / 'ranking.csv', result['ranking'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('ranking','patient_scores','source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
