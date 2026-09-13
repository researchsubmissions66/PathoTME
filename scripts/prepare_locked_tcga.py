#!/usr/bin/env python3
"""Generate only the bounded two-cohort 16-shot study; never submits jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym']
from common.configuration import load_dotenv, load_yaml_config
from common.run_state import validate_resume_state
from pathotme.locked_tcga import ROOT, PGVL, RESULTS, DATA, sha, rows, membership, check_phases, atomic_json, identity
from pathotme.controlled_vila import donor_maps


def inspect_feature(item):
    """Inspect one immutable feature header; safe to run in a bounded worker."""
    import h5py
    path, encoder, width, scale, source = item
    path = Path(path); stat = path.stat()
    with h5py.File(path, 'r') as h:
        f, c = h['features'], h['coords']
        if (f.ndim != 2 or f.shape[1] != width or f.shape[0] < 1
                or c.shape != (f.shape[0], 2)
                or f.attrs.get('encoder') != encoder
                or int(c.attrs.get('patch_size', -1)) != 224
                or float(c.attrs.get('target_magnification', -1)) != scale):
            raise ValueError(f'feature identity/geometry mismatch: {path}')
        return str(path), {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'shape': list(f.shape), 'encoder': encoder, 'magnification': scale,
            'patch_size': 224, 'registry_feature_space_id': source['feature_space_id'],
            'registry_representation': source.get('feature_representation', 'native_shared_embedding'),
            'checkpoint_attestation': 'historical_extraction_digest_absent; registry_binding_only'}


def feature_inventory(frame, protocol):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    tasks = []
    for encoder, width in [('clip_rn50', 1024), ('plip', 768)]:
        for scale in (5, 10):
            source = protocol['feature_sources'][f'{encoder}_{scale}x']
            for row in frame.to_dict('records'):
                path = Path(os.path.expandvars(row[f'feature__{encoder}_{scale}x']))
                tasks.append((str(path), encoder, width, scale, source))
    inventory = {}
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('fork')) as pool:
        for i, (path, result) in enumerate(pool.map(inspect_feature, tasks, chunksize=16), 1):
            inventory[path] = result
            if i % 500 == 0 or i == len(tasks):
                print(f'feature headers {i}/{len(tasks)} passed', flush=True)
    return inventory


def prepare(config, output):
    import pandas as pd
    import numpy as np
    from scripts.tcga_benchmark import build_splits
    from common.preflight import preflight
    from pathotme.brca_features import panel_spec, transform_rows
    from pathotme.features import transform_nsclc_features, validate_feature_table
    spec = json.loads(config.read_text())
    if spec['cohorts'] != ['nsclc', 'brca'] or spec['methods'] != ['vila_mil', 'mgpath'] or spec['folds'] != list(range(5)):
        raise ValueError('only the explicitly bounded study is supported')
    if output.exists():
        raise FileExistsError(f'preserve existing preparation: {output}')
    output.mkdir(parents=True)
    bound = {str(config): sha(config)}
    plans = []
    coverage = {}
    for cohort in spec['cohorts']:
        original_protocol = PGVL / f'benchmarks/tcga_{cohort}/protocol.yaml'
        protocol = load_yaml_config(original_protocol)
        original_manifest = PGVL / f'benchmarks/tcga_{cohort}/data/{cohort}/manifest.csv'
        tme = DATA / 'processed' / ('opentme_nsclc_core62.csv' if cohort == 'nsclc' else 'opentme_brca_morph64_v1.csv')
        table = pd.read_csv(tme)
        if table.slide_id.duplicated().any():
            raise ValueError('duplicate TME rows')
        if cohort == 'brca':
            metadata = tme.with_suffix('.csv.metadata.json')
            meta = json.loads(metadata.read_text()); panel = panel_spec('brca_morph64_v1')
            if (meta['selected_table_sha256'] != sha(tme) or meta['schema_sha256'] != panel['schema_sha256']
                    or meta['builder_sha256'] != sha(ROOT / 'pathotme/brca_features.py')
                    or meta['revision'] != panel['revision']):
                raise ValueError('BRCA TME provenance mismatch')
            bound[str(metadata)] = sha(metadata)
            transformed = pd.DataFrame(transform_rows(rows(tme), 'brca_morph64_v1'), index=table.slide_id)
        else:
            transformed = transform_nsclc_features(validate_feature_table(table))
            transformed.index = table.slide_id
        original = pd.read_csv(original_manifest)
        frame = original[original.slide_id.isin(table.slide_id)].copy()
        coverage[cohort] = {'slides': len(frame), 'patients': frame.case_id.nunique(),
            'excluded_slide_ids': sorted(set(original.slide_id) - set(frame.slide_id))}
        manifest = output / f'data/{cohort}/manifest.csv'
        manifest.parent.mkdir(parents=True)
        frame.to_csv(manifest, index=False)
        split_protocol = {**protocol, 'shots': spec['split_sampling_reference_shots'],
                          'cohorts': {cohort: protocol['cohorts'][cohort]}}
        with tempfile.TemporaryDirectory(prefix='pathotme_splits_') as td:
            temp = Path(td); (temp / f'data/{cohort}').mkdir(parents=True)
            shutil.copy2(manifest, temp / f'data/{cohort}/manifest.csv')
            build_splits(split_protocol, temp)
            shutil.copytree(temp / f'splits/{cohort}/16shot', output / f'splits/{cohort}/16shot')
        inventory = output / f'{cohort}_feature_inventory.json'
        atomic_json(inventory, feature_inventory(frame, protocol))
        for p in [original_protocol, original_manifest, tme, manifest, inventory]:
            bound[str(p)] = sha(p)
        test_ids = []
        for fold in spec['folds']:
            split_dir = output / f'splits/{cohort}/16shot'
            phases = {phase: rows(split_dir / f'fold{fold}/{phase}.csv') for phase in ['train', 'val', 'test']}
            check_phases(phases, table.slide_id)
            values = transformed.loc[[r['slide_id'] for r in phases['train']]].to_numpy(dtype=float)
            if np.isinf(values).any() or np.isnan(values).all(axis=0).any():
                raise ValueError('invalid train-only imputation inputs')
            test_ids.extend(r['slide_id'] for r in phases['test'])
            maps = output / f'splits/{cohort}/16shot/fold{fold}/donors.json'
            atomic_json(maps, donor_maps(phases, spec['seed'], fold))
            for method in spec['methods']:
                base_path = PGVL / f'benchmarks/tcga_{cohort}/configs/{method}/{cohort}_16shot.yaml'
                base = load_yaml_config(base_path)
                baseline = Path(base['results_dir'])
                native_state = json.loads((baseline / 'metrics.json').read_text())
                valid = validate_resume_state(native_state, baseline / 'config.json', method, base)
                record = next((r for r in valid if r['fold'] == fold), None)
                if record is None or any(record.get('sample_failures', {}).values()):
                    raise ValueError('historical native fixture invalid')
                reusable = all(membership(phases[p]) == membership(rows(Path(base['split_dir']) / f'fold{fold}/{p}.csv')) for p in phases)
                cfg = {**base, 'dataset_csv': str(manifest), 'split_dir': str(split_dir),
                       'experiment': f"{spec['study']}_{cohort}_{method}_native",
                       'benchmark': spec['study'], 'k_start': fold, 'k_end': fold + 1,
                       'results_dir': str(output / f'runs/{cohort}/{method}/fold{fold}/native')}
                native_dir = baseline if reusable else Path(cfg['results_dir'])
                cfg_path = output / f'configs/{cohort}_{method}_fold{fold}.json'
                atomic_json(cfg_path, cfg)
                report = preflight(cfg, check_features=True)
                if not report.ok:
                    raise ValueError(f'{cohort}/{method}/{fold}: {report.as_dict()}')
                reuse_files = [baseline / f'fold{fold}_best.pt']
                if reusable:
                    prediction = baseline / f'fold{fold}_predictions.csv'
                    if sorted(membership(rows(prediction))) != sorted(membership(phases['test'])):
                        raise ValueError('reused predictions do not cover identical test slides')
                    reuse_files += [prediction, baseline / 'config.json', baseline / 'metrics.json']
                for p in [base_path, Path(base['text_prompt_path']), *reuse_files]:
                    if str(p) not in bound:
                        bound[str(p)] = sha(p)
                plans.append({'cohort': cohort, 'method': method, 'fold': fold,
                    'config': str(cfg_path), 'native_reused': reusable, 'native_dir': str(native_dir),
                    'smoke_checkpoint_dir': str(baseline), 'tme_csv': str(tme), 'donor_maps': str(maps),
                    'feature_inventory': str(inventory), 'output': str(output / f'runs/{cohort}/{method}/fold{fold}'),
                    'historical_native_provenance': 'config/artifact bound; historical producer source revision not independently reattested',
                    'split_counts': {p: len(v) for p, v in phases.items()}})
                print(f'prepared {cohort}/{method}/fold{fold} native_reused={reusable}', flush=True)
        if len(test_ids) != len(set(test_ids)) or set(test_ids) != set(frame.slide_id):
            raise ValueError('outer folds must cover every eligible slide exactly once')
    # Runtime hashes include both projects without modifying either native source tree.
    source_paths = [PGVL / 'train.py', PGVL / 'scripts/tcga_benchmark.py']
    for directory in [ROOT / 'pathotme', ROOT / 'scripts', PGVL / 'common', PGVL / 'methods/vila_mil', PGVL / 'methods/mgpath']:
        source_paths += list(directory.rglob('*.py'))
    source_paths += [PGVL / 'methods/base.py', PGVL / 'methods/__init__.py']
    for p in source_paths:
        bound[str(p)] = sha(p)
    for p in output.rglob('*'):
        if p.is_file():
            bound[str(p)] = sha(p)
    launch = {'protocol': spec, 'protocol_path': str(config), 'output': str(output),
              'coverage': coverage, 'plans': plans, 'file_sha256': bound,
              'counts': {'fold_jobs': len(plans), 'smoke_jobs': 4, 'comparisons': len(plans)*4,
                         'native_reused': sum(p['native_reused'] for p in plans),
                         'native_new': sum(not p['native_reused'] for p in plans), 'adapter_fits': len(plans)*3}}
    launch['identity'] = identity(launch)
    atomic_json(output / 'launch.json', launch)
    print(json.dumps({'launch': str(output / 'launch.json'), 'counts': launch['counts'], 'coverage': coverage}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/tcga_locked_16shot_20260909.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    load_dotenv(PGVL / '.env')
    os.environ.setdefault('PGVL_REPO_ROOT', str(PGVL))
    if args.prepare:
        prepare(args.config.resolve(), args.output.resolve())
    else:
        print(json.dumps({'would_prepare': str(args.output), 'protocol': json.loads(args.config.read_text())}))
