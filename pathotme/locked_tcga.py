"""Dependency-light immutable contracts for the two-cohort TCGA study."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/path/to/PathoTME')
PGVL = Path('/path/to/PGVL-Gym')
RESULTS = Path('/path/to/shared/PathoTME-results')
DATA = Path('/path/to/shared/PathoTME-data')
PYTHON = '/path/to/shared/envs/pgvl-gym/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def membership(values):
    return [(r['slide_id'], r['case_id'], int(r.get('label_id', r.get('label')))) for r in values]


def check_phases(phases, covered):
    patients = {}
    for phase, values in phases.items():
        slides = [r['slide_id'] for r in values]
        if len(set(slides)) != len(slides) or set(slides) - set(covered):
            raise ValueError(f'{phase}: duplicate or uncovered slides')
        patients[phase] = {r['case_id'] for r in values}
        if {int(r['label_id']) for r in values} != {0, 1}:
            raise ValueError('two ordered classes required')
        if phase != 'test' and any(sum(int(r['label_id']) == c for r in values) != 16 for c in (0, 1)):
            raise ValueError('exactly 16 training/validation slides per class required')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if patients[a] & patients[b]:
            raise ValueError('patient leakage')


def verify_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f'locked file changed: {path}')


def load_launch(path):
    launch = json.loads(Path(path).read_text())
    expected = launch.pop('identity')
    if identity(launch) != expected:
        raise ValueError('launch identity mismatch')
    launch['identity'] = expected
    verify_files(launch['file_sha256'])
    return launch


def verify_completion(path, expected):
    record = json.loads(Path(path).read_text())
    if record.get('identity') != expected or record.get('status') != 'completed':
        raise ValueError('completion identity/status mismatch')
    verify_files(record['artifact_sha256'])
    return record
