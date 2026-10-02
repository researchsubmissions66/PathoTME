#!/usr/bin/env python3
"""Within-panel subset and biological-group LR refits; no GPU allocation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(os.environ.get('PATHOTME_ROOT', '/path/to/PathoTME'))
PGVL = Path(os.environ.get('PGVL_ROOT', '/path/to/PGVL-Gym'))
BASE = ROOT/'campaigns/baselines_20260914'
sys.path[:0] = [str(BASE), str(ROOT), str(PGVL)]
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from pathotme.locked_tcga import atomic_json, identity, sha, verify_files
from pathotme.shared_panel import common_spec
from pathotme.brca_features import panel_spec
from baseline_core import (POLICY as BASE_POLICY, read_splits, feature_matrix,
    fit_scaler, scale_values, metrics, selection_key, checked_predictions, csv_write)

POLICY = {
    'version': 'within_panel_feature_ablation_20260915_v1',
    'variant': 'PathoTME-LR', 'shots': 16, 'subset_sizes': [8, 16, 32],
    'random_repeats': 10,
    'selector': 'training_only_recursive_feature_elimination_L2_logistic',
    'selector_C': 1.0, 'selector_step': 1,
    'selector_tie': 'remove_lexicographically_first_feature_among_equal_absolute_coefficients',
    'random_masks': 'nested_hash_order_per_panel_and_repeat_shared_across_folds_and_matching_cohorts',
    'c_grid': BASE_POLICY['c_grid'], 'selection': BASE_POLICY['selection'],
    'preprocessing': BASE_POLICY['preprocessing'], 'fit': BASE_POLICY['fit'],
    'patient_aggregation': BASE_POLICY['patient_aggregation'],
    'group_design': ['group_only', 'leave_group_out'],
    'no_train_val_refit': True, 'no_test_subset_selection': True,
    'scope': 'existing_core62_and_historical_BRCA_morph64_only',
    'exploratory_after_prior_results_inspected': True,
    'interpretation': 'predictive_sufficiency_and_redundancy_within_panel_not_original_panel_optimality',
}


def load(path):
    return json.loads(Path(path).read_text())


def groups_for(task):
    spec = panel_spec(task['panel']) if task['panel'].startswith('brca_') else common_spec()
    names = list(spec['feature_names'])
    if names != task['feature_names']:
        raise ValueError('Feature order differs from parent panel')
    groups = {'tissue_architecture': [], 'cell_composition': [], 'spatial_interactions': []}
    groups['tissue_geometry' if task['panel'] == 'brca_morph64_v1' else 'tls'] = []
    for name in names:
        if name.startswith(('CELL_PERCENTAGE_', 'CELL_DENSITY_')):
            group = 'cell_composition'
        elif name.startswith(('RATIO_OF_', 'AVG_MIN_DISTANCE_OF_')):
            group = 'spatial_interactions'
        elif name.startswith(('TLS_', 'LOG1P_COUNT_TLS_')):
            group = 'tls'
        elif name.startswith(('AVG_ECCENTRICITY_', 'AVG_SOLIDICITY_', 'LARGEST_ROUNDNESS_', 'REGIONS_PER_MM2_')):
            group = 'tissue_geometry'
        elif name.startswith('RELATIVE_AREA_'):
            group = 'tissue_architecture'
        else:
            raise ValueError('Unclassified feature: '+name)
        groups[group].append(name)
    flattened = [n for group in groups.values() for n in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(names) or not all(groups.values()):
        raise ValueError('Biological groups must partition this entire panel')
    return groups


def fixed_conditions(task):
    names = task['feature_names']
    rows = [{'name': 'full', 'kind': 'full', 'k': len(names), 'group': '', 'repeat': -1,
             'feature_names': names}]
    for repeat in range(POLICY['random_repeats']):
        order = sorted(names, key=lambda n: identity({'version': POLICY['version'],
                       'feature_names': names, 'repeat': repeat, 'feature': n}))
        for k in POLICY['subset_sizes']:
            retained = set(order[:k])
            rows.append({'name': f'random_k{k}_r{repeat:02d}', 'kind': 'random', 'k': k,
                         'group': '', 'repeat': repeat, 'feature_names': [n for n in names if n in retained]})
    for group, members in groups_for(task).items():
        for kind in POLICY['group_design']:
            selected = set(members) if kind == 'group_only' else set(names)-set(members)
            rows.append({'name': kind+'_'+group, 'kind': kind, 'k': len(selected), 'group': group,
                         'repeat': -1, 'feature_names': [n for n in names if n in selected]})
    return rows


def fit(x, y, c, seed):
    model = LogisticRegression(C=c, solver='lbfgs', max_iter=5000, tol=1e-8, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    if list(model.classes_) != [0, 1] or not np.isfinite(model.coef_).all():
        raise ValueError('Invalid fitted classifier')
    return model


def training_subsets(train, labels, names, seed):
    """No validation/test arrays or parent fitted model enter this selector."""
    active = list(range(len(names)))
    subsets, elimination = {}, []
    while True:
        if len(active) in POLICY['subset_sizes']:
            subsets[len(active)] = [names[i] for i in active]
        if len(active) <= min(POLICY['subset_sizes']):
            break
        model = fit(train[:, active], labels, POLICY['selector_C'], seed)
        local = min(range(len(active)), key=lambda j: (abs(float(model.coef_[0, j])), names[active[j]]))
        elimination.append({'remaining_before': len(active), 'removed': names[active[local]],
                            'absolute_coefficient': abs(float(model.coef_[0, local]))})
        active.pop(local)
    return [{'name': f'rfe_k{k}', 'kind': 'rfe', 'k': k, 'group': '', 'repeat': -1,
             'feature_names': subsets[k]} for k in POLICY['subset_sizes']], elimination


def private_output(path):
    out = Path(path).expanduser().resolve()
    if any(out == r or r in out.parents for r in [ROOT, PGVL]):
        raise ValueError('Keep patient-level experiment outputs outside code repositories')
    return out


def prepare(parent_path, output):
    parent = load(parent_path)
    if identity({k:v for k,v in parent.items() if k != 'identity'}) != parent['identity']:
        raise ValueError('Invalid parent manifest')
    verify_files(parent['source_sha256'])
    out = private_output(output)
    if (out/'manifest.json').exists():
        raise FileExistsError('Use the existing manifest; do not regenerate silently')
    tasks = parent['tme_tasks']
    if Counter(t['cohort'] for t in tasks) != Counter({c:5 for c in ['nsclc','brca','crc','blca']}):
        raise ValueError('Expected the exact four-cohort five-fold study')
    test_owners = {}
    audits = []
    for task in tasks:
        verify_files(task['input_sha256'])
        frames = read_splits(task)
        for phase, frame in frames.items():
            if frame.to_dict('records') != task['memberships'][phase]:
                raise ValueError('Frozen membership mismatch')
            if phase != 'test' and frame.label.value_counts().to_dict() != {0:16, 1:16}:
                raise ValueError('Changed 16-shot budget')
        for case in frames['test'].case_id.unique():
            key = (task['cohort'], case)
            if test_owners.setdefault(key, task['fold']) != task['fold']:
                raise ValueError('Test patient overlaps folds')
        groups = groups_for(task)
        audits.append({'cohort':task['cohort'], 'fold':task['fold'], 'panel':task['panel'],
                       'group_sizes':{k:len(v) for k,v in groups.items()},
                       'memberships':{p:{'slides':len(f),'patients':int(f.case_id.nunique())} for p,f in frames.items()}})
    sources = {str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    sources.update(parent['source_sha256'])
    manifest = {'policy': POLICY, 'output':str(out), 'parent_manifest':str(Path(parent_path).resolve()),
                'parent_sha256':sha(parent_path), 'parent_identity':parent['identity'], 'tasks':tasks,
                'source_sha256':sources, 'sklearn_version':sklearn.__version__,
                'groups':{t['cohort']:groups_for(t) for t in tasks},
                'fixed_conditions':{t['cohort']:fixed_conditions(t) for t in tasks},
                'counts':{'cohort_folds':20, 'condition_folds':20*(39+3),
                          'selected_refits':20*42, 'C_candidate_fits':20*42*5, 'gpu_jobs':0},
                'subset_scope_note':'Learned masks vary by fold and must not be promoted to a test-selected global panel.',
                'neural_scope':'not_executed_or_submitted_by_this_CPU_runner',
                'created_at_utc':datetime.now(timezone.utc).isoformat()}
    manifest['identity'] = identity(manifest)
    atomic_json(out/'manifest.json', manifest)
    atomic_json(out/'preflight.json', {'status':'passed', 'folds':audits,
                'original_source_bindings_verified':len(parent['source_sha256']),
                'manifest_identity':manifest['identity'], 'training':False})
    print(json.dumps({'manifest':str(out/'manifest.json'), 'identity':manifest['identity'],
                      'counts':manifest['counts']}), flush=True)


def run_fold(manifest, task):
    root = Path(manifest['output'])/'folds'/f"{task['cohort']}_f{task['fold']}"
    root.mkdir(parents=True, exist_ok=True)
    verify_files(task['input_sha256'])
    frames = read_splits(task)
    names = task['feature_names']
    train_raw = feature_matrix(task, frames['train'])
    scaler = fit_scaler(train_raw)
    train = scale_values(train_raw, scaler)
    learned, elimination = training_subsets(train, frames['train'].label.to_numpy(), names, task['seed'])
    design = {'manifest_identity':manifest['identity'], 'task_id':task['id'], 'scaler':scaler,
              'selector_policy':POLICY, 'elimination':elimination,
              'conditions':fixed_conditions(task)+learned}
    design['identity'] = identity(design)
    design_path = root/'design.json'
    if design_path.exists() and load(design_path) != design:
        raise ValueError('Existing frozen feature design differs')
    atomic_json(design_path, design)
    # Feature masks are frozen before even transforming validation values.
    val = scale_values(feature_matrix(task, frames['val']), scaler)
    values_test = None
    for condition in design['conditions']:
        folder = root/condition['name']
        result_path = folder/'metrics.json'
        if result_path.exists():
            previous = load(result_path)
            if previous['design_identity'] != design['identity'] or previous['status'] != 'completed':
                raise ValueError('Existing condition identity/status mismatch')
            verify_files(previous['artifact_sha256'])
            continue
        start = time.monotonic()
        indices = [names.index(n) for n in condition['feature_names']]
        candidates, models = [], {}
        for c in POLICY['c_grid']:
            model = fit(train[:,indices], frames['train'].label.to_numpy(), c, task['seed'])
            candidate = {'C':c, 'metrics':metrics(frames['val'], model.predict_proba(val[:,indices])[:,1])}
            candidates.append(candidate); models[c] = model
        chosen = min(candidates, key=lambda r:selection_key(r['metrics'], r['C']))
        model = models[chosen['C']]
        folder.mkdir(exist_ok=True)
        atomic_json(folder/'selection.json', {'selected_C':chosen['C'], 'candidates':candidates,
                    'condition':condition, 'design_identity':design['identity']})
        atomic_json(folder/'model.json', {'feature_names':condition['feature_names'], 'indices':indices,
                    'coef':model.coef_[0].tolist(), 'intercept':float(model.intercept_[0]), 'C':chosen['C'],
                    'scaler':{k:[v[i] for i in indices] for k,v in scaler.items()}, 'classes':[0,1]})
        # This condition's mask and C are committed before test values are used.
        if values_test is None:
            values_test = scale_values(feature_matrix(task, frames['test']), scaler)
        phase_scores, probabilities = {}, {}
        for phase, values in [('val',val), ('test',values_test)]:
            p = model.predict_proba(values[:,indices])[:,1]
            probabilities[phase] = p
            csv_write(folder/f'{phase}_predictions.csv', frames[phase].assign(probability_0=1-p, probability_1=p))
            phase_scores[phase] = metrics(frames[phase], p)
        parity = None
        parent_bindings = {}
        if condition['kind'] == 'full':
            parent_folder = Path(manifest['parent_manifest']).parent/'tme_only'/task['id']
            saved = load(parent_folder/'metrics.json'); verify_files(saved['artifact_sha256'])
            if saved['selection']['C'] != chosen['C']:
                raise ValueError('Full-panel selected C differs from matching baseline')
            parity = {}
            for phase in ['val','test']:
                pred = checked_predictions(parent_folder/f'{phase}_predictions.csv', frames[phase])
                error = float(np.max(np.abs(pred.probability_1.to_numpy()-probabilities[phase])))
                if error > 1e-9:
                    raise ValueError('Full-panel refit fails original baseline probability parity')
                parity[phase] = error
            parent_bindings = {str(parent_folder/'metrics.json'):sha(parent_folder/'metrics.json')}
        result = {'status':'completed', 'manifest_identity':manifest['identity'], 'design_identity':design['identity'],
                  'variant':'PathoTME-LR', 'cohort':task['cohort'], 'panel':task['panel'], 'fold':task['fold'],
                  'condition':condition, 'selected_C':chosen['C'], 'metrics':phase_scores,
                  'full_panel_parent_parity_max_error':parity, 'parent_artifact_sha256':parent_bindings,
                  'wall_seconds':time.monotonic()-start,
                  'artifact_sha256':{str(folder/n):sha(folder/n) for n in
                     ['selection.json','model.json','val_predictions.csv','test_predictions.csv']}}
        atomic_json(result_path, result)
    return {'status':'completed', 'conditions':len(design['conditions']), 'design_identity':design['identity']}


def execute(manifest, cohort=None):
    root = Path(manifest['output'])
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_files(manifest['source_sha256'])
        if sha(manifest['parent_manifest']) != manifest['parent_sha256']:
            raise ValueError('Parent manifest changed')
        status_path = root/'status.json'
        status = load(status_path) if status_path.exists() else {'identity':manifest['identity'], 'folds':{}}
        if status['identity'] != manifest['identity']:
            raise ValueError('Status identity mismatch')
        status.update(pid=os.getpid(), node=os.uname().nodename)
        for task in manifest['tasks']:
            if cohort and task['cohort'] != cohort:
                continue
            name = f"{task['cohort']}_f{task['fold']}"
            status['folds'][name] = {'status':'running', 'started_at':time.time()}
            atomic_json(status_path, status)
            try:
                outcome = run_fold(manifest, task)
            except Exception as error:
                status['folds'][name] = {'status':'failed', 'error':str(error)}
                atomic_json(status_path, status)
                raise
            status['folds'][name] = outcome
            atomic_json(status_path, status)
            print(json.dumps({'fold':name, **outcome}), flush=True)
        verify_files(manifest['source_sha256'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--parent-manifest', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--cohort', choices=['nsclc','brca','crc','blca'])
    a = p.parse_args()
    if a.prepare:
        if not a.parent_manifest or not a.output:
            p.error('--prepare requires --parent-manifest and --output')
        prepare(a.parent_manifest, a.output)
        return
    if not a.manifest:
        p.error('--manifest is required')
    m = load(a.manifest)
    if identity({k:v for k,v in m.items() if k != 'identity'}) != m['identity'] or m['policy'] != POLICY:
        raise ValueError('Manifest identity or policy mismatch')
    if a.execute:
        execute(m, a.cohort)
    else:
        print(json.dumps({'counts':m['counts'], 'policy':m['policy']}, indent=2))


if __name__ == '__main__':
    main()
