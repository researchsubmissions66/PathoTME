"""Exact, isolated 32/64-shot bindings; no full-shot or old campaign mutation."""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/path/to/PathoTME')
PGVL = Path('/path/to/PGVL-Gym')
sys.path[:0] = [str(ROOT), str(PGVL)]
from pathotme.locked_tcga import atomic_json, identity, sha, rows, membership, verify_files
from pathotme.focus_contract import validate_donors

OUTPUT = Path('/path/to/shared/PathoTME-results/tcga_32_64shot_20260915_v1')
RUNTIME = ROOT / 'campaigns/highshot_20260915'
METHODS = ['vila_mil', 'mgpath', 'focus', 'muse', 'hive_mil', 'dyko', 'mscpt']
ENCODERS = ['plip', 'clip-rn50']
SCOPE = {'nsclc': [32, 64], 'brca': [32, 64], 'blca': [32]}
BASELINE = OUTPUT.parent / 'baselines_16shot_20260914_v1/manifest.json'
ALLSHOTS = OUTPUT.parent / 'feature_ablation_allshots_20260915_v1/manifest.json'

def read_config(path):
    import yaml
    path = Path(path)
    return json.loads(path.read_text()) if path.suffix == '.json' else yaml.safe_load(path.read_text())

def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(values)

def check_phases(phases, shots):
    if set(phases) != {'train', 'val', 'test'}: raise ValueError('Missing phase')
    patients = {}
    for phase, values in phases.items():
        members = membership(values)
        if len({r[0] for r in members}) != len(values): raise ValueError('Duplicate slide')
        patients[phase] = {r[1] for r in members}
        if phase != 'test' and (Counter(r[2] for r in members) != {0: shots, 1: shots}
                               or len(patients[phase]) != 2*shots):
            raise ValueError('Wrong patient shots')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if patients[a] & patients[b]: raise ValueError('Patient leakage')

def load_campaign(path, verify=True):
    data = json.loads(Path(path).read_text()); digest = data.pop('identity')
    if identity(data) != digest: raise ValueError('Campaign identity changed')
    data['identity'] = digest
    if data['scope'] != SCOPE or len(data['plans']) != 350 or len(data['groups']) != 42:
        raise ValueError('Unauthorized scope')
    if verify: verify_files(data['file_sha256'])
    return data

def adapter_config(plan, cfg, arm, smoke=False):
    spec = plan['adapter']
    out = Path(plan['smoke_output'] if smoke else plan['output']) / arm
    result = {**cfg, '_fold_index': plan['fold'], 'results_dir': str(out),
        'base_checkpoint_dir': plan['smoke_checkpoint_dir'] if smoke else plan['native_dir'],
        'tme_feature_csv': plan['tme_csv'], 'tme_panel': plan['panel'],
        'tme_mode': 'zero' if arm == 'zero' else 'actual',
        **{'tme_'+k: spec[k] for k in ['hidden_dim', 'attention_heads', 'dropout', 'initial_gate']},
        'optimizer': 'adam', 'lr': spec['lr'], 'weight_decay': spec['weight_decay'], 'lr_scheduler': None}
    if plan['cohort'] == 'blca': result['pathotme_arm'] = arm
    return result

def validate_config(cfg):
    path = cfg.get('highshot_contract_path')
    if not path or sha(path) != cfg.get('highshot_contract_sha256'):
        raise ValueError('High-shot contract changed or missing')
    contract = json.loads(Path(path).read_text())
    expected = contract['native_config']
    plan = contract['plan']
    expected = {**expected, 'highshot_contract_path': path,
                'highshot_contract_sha256': cfg['highshot_contract_sha256']}
    candidates = [expected, {**expected, '_fold_index': plan['fold']}]
    for smoke in [False, True]:
        for arm in ['zero', 'actual', 'shuffled']:
            candidates.append(adapter_config(plan, expected, arm, smoke))
    if cfg not in candidates:
        closest = min(candidates, key=lambda c: sum(c.get(k) != cfg.get(k) for k in set(c)|set(cfg)))
        drift = [k for k in set(closest)|set(cfg) if closest.get(k) != cfg.get(k)]
        raise ValueError(f'Frozen high-shot recipe drift: {sorted(drift)}')
    if cfg['shots'] not in SCOPE[cfg['task']]: raise ValueError('Unsupported shots')
    verify_files(contract['assets'])

def check_features(plan, cfg):
    validate_config(cfg)
    phases = {p: rows(Path(cfg['split_dir'])/f"fold{plan['fold']}/{p}.csv") for p in ['train','val','test']}
    check_phases(phases, plan['shots'])
    validate_donors(phases, json.loads(Path(plan['donor_maps']).read_text()))
    inventory = json.loads(Path(plan['feature_inventory']).read_text())
    columns = {v for k,v in cfg.items() if k.startswith('feature_path_column')}
    for values in phases.values():
        for row in values:
            bound = inventory['slides'][row['slide_id']]['files'] if 'slides' in inventory else inventory
            for col in columns:
                path = os.path.expandvars(row[col]); stat = Path(path).stat(); item = bound[path]
                if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
                    raise ValueError(f'Changed audited feature: {path}')
