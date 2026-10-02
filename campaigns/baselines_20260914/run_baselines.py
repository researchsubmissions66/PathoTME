#!/usr/bin/env python3
"""Prepare or run matched baselines on login CPU; never allocate a GPU."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PGVL = Path(os.environ.get('PGVL_ROOT', str(ROOT.parent/'PGVL-Gym')))
sys.path[:0] = [str(Path(__file__).parent), str(ROOT), str(PGVL)]

from pathotme.locked_tcga import atomic_json, identity, sha, verify_files


def prepare(paths, output):
    from common.configuration import load_yaml_config
    from baseline_core import POLICY, split_frame, read_splits
    from pathotme.brca_features import panel_spec as breast_spec
    from pathotme.shared_panel import common_spec
    selected = {}
    for path in paths:
        launch = json.loads(Path(path).read_text())
        if identity({k: v for k, v in launch.items() if k != 'identity'}) != launch['identity']:
            raise ValueError('Invalid parent launch identity')
        for plan in launch['plans']:
            cfg = load_yaml_config(plan['config'])
            if cfg['shots'] != 16:
                raise ValueError('This addition is bounded to the main 16-shot study')
            key = (plan['cohort'], plan['method'], cfg['backbone'], plan['fold'])
            selected[key] = (path, launch, plan, cfg)
    tme_tasks, fusion_tasks = {}, []
    for key, (launch_path, launch, plan, cfg) in sorted(selected.items()):
        cohort, method, encoder, fold = key
        panel = 'brca_morph64_v1' if cohort == 'brca' else 'shared_core62_v1'
        spec = breast_spec(panel) if cohort == 'brca' else common_spec()
        splits = {p: str(Path(cfg['split_dir'])/f'fold{fold}/{p}.csv') for p in ['train', 'val', 'test']}
        # Only identical panel bytes, class binding, patient/slide membership and
        # preprocessing/selection policy may share one fitted TME comparator.
        split_membership = {p: split_frame(path, cfg['label_dict']).to_dict('records') for p, path in splits.items()}
        files = [plan['tme_csv'], *splits.values(), plan['config']]
        for path in files:
            if launch['file_sha256'].get(path) != sha(path):
                raise ValueError(f'Input differs from its parent launch: {path}')
        tme_payload = {'cohort': cohort, 'fold': fold, 'panel': panel, 'feature_names': list(spec['feature_names']),
            'label_dict': cfg['label_dict'], 'seed': cfg.get('seed', 1)+fold,
            'tme_sha256': sha(plan['tme_csv']), 'memberships': split_membership, 'policy': POLICY}
        tme_id = identity(tme_payload)
        tme_task = {**tme_payload, 'id': tme_id, 'tme_csv': plan['tme_csv'], 'splits': splits,
                    'input_sha256': {p: sha(p) for p in [plan['tme_csv'], *splits.values()]}}
        read_splits(tme_task)
        tme_tasks.setdefault(tme_id, tme_task)
        task = {'name': '_'.join(map(str, key)), 'cohort': cohort, 'method': method, 'encoder': encoder,
            'fold': fold, 'tme_id': tme_id, 'plan': plan, 'launch': str(launch_path),
            'launch_identity': launch['identity'], 'label_dict': cfg['label_dict'], 'splits': splits,
            'input_sha256': {p: sha(p) for p in files}}
        task['id'] = identity(task)
        fusion_tasks.append(task)
    output = Path(output)
    if (output/'manifest.json').exists():
        raise FileExistsError('Baseline manifest exists; use it instead of silently regenerating')
    # Source snapshots bind this new analysis, without changing any parent runtime.
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += list((ROOT/'pathotme').glob('*.py'))
    sources += list((PGVL/'common').rglob('*.py'))+[PGVL/'train.py']
    for method in ('vila_mil', 'mgpath', 'focus', 'muse', 'hive_mil', 'dyko', 'mscpt'):
        sources += list((PGVL/'methods'/method).rglob('*.py'))
    sources += list((ROOT/'campaigns/crc_blca_core62_20260914').glob('*.py'))
    report = {'output': str(output), 'policy': POLICY, 'scope': 'four_cohorts_seven_architectures_two_encoders_16shot_five_folds',
        'tme_tasks': list(tme_tasks.values()), 'fusion_tasks': fusion_tasks,
        'source_sha256': {str(p): sha(p) for p in sorted(set(sources))},
        'counts': {'unique_tme_fits': len(tme_tasks), 'fusion_fold_evaluations': len(fusion_tasks), 'gpu_jobs': 0},
        'panel_note': 'Historical BRCA remains morph64; a common62 BRCA result is a separate experiment.',
        'precision_note': 'Same preprocessing equations; the linear baseline fits in CPU float64, neural buffers use float32.'}
    report['identity'] = identity(report)
    atomic_json(output/'manifest.json', report)
    print(json.dumps({'manifest': str(output/'manifest.json'), 'identity': report['identity'], 'counts': report['counts']}), flush=True)


def run_one(manifest, name):
    from baseline_core import run_tme, run_fusion
    from native_export import NativePending
    tme = {t['id']: t for t in manifest['tme_tasks']}
    if name.startswith('tme:'):
        return run_tme(tme[name[4:]], manifest['output'])
    task = next(t for t in manifest['fusion_tasks'] if t['name'] == name)
    verify_files(task['input_sha256'])
    try:
        return run_fusion(task, tme[task['tme_id']], manifest['output'])
    except NativePending as error:
        return {'status': 'waiting_native', 'reason': str(error)}


def dispatch(manifest, args):
    root = Path(manifest['output'])
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report_path = root/'status.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {'identity': manifest['identity'], 'tasks': {}}
        if report['identity'] != manifest['identity']:
            raise ValueError('Controller report belongs to a different manifest')
        report.update(pid=os.getpid(), node=os.uname().nodename, started_at=time.time())
        # Cheap fits first. Native replay then runs serially at the declared CPU cap.
        names = ['tme:'+t['id'] for t in manifest['tme_tasks']]
        if not args.tme_only:
            names += [t['name'] for t in manifest['fusion_tasks']]
        (root/'logs').mkdir(exist_ok=True)
        while True:
            verify_files(manifest['source_sha256'])
            for name in names:
                previous = report['tasks'].get(name, {})
                if previous.get('status') in ('completed', 'failed'):
                    continue
                if not name.startswith('tme:'):
                    task = next(t for t in manifest['fusion_tasks'] if t['name'] == name)
                    native = Path(task['plan']['native_dir'])
                    needed = [native/'metrics.json', native/f"fold{task['fold']}_best.pt", native/f"fold{task['fold']}_predictions.csv"]
                    if not all(p.exists() for p in needed):
                        report['tasks'][name] = {'status': 'waiting_native'}
                        continue
                log = root/'logs'/(name.replace(':', '_')+'.log')
                report['tasks'][name] = {'status': 'running', 'log': str(log), 'started_at': time.time()}
                atomic_json(report_path, report)
                command = [sys.executable, '-u', __file__, '--manifest', str(args.manifest), '--execute', '--one', name]
                with log.open('a') as handle:
                    result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
                status_path = root/'task_status'/(name.replace(':', '_')+'.json')
                status = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'failed', 'error': 'Worker exited without a status report'}
                if result.returncode:
                    status = {**status, 'status': 'failed', 'error': status.get('error', 'Worker returned a nonzero exit; inspect its log')}
                report['tasks'][name] = {**report['tasks'][name], **status, 'returncode': result.returncode}
                report['counts'] = dict(Counter(x['status'] for x in report['tasks'].values()))
                atomic_json(report_path, report)
                print(json.dumps({'task': name, 'status': status['status']}), flush=True)
            report['counts'] = dict(Counter(x['status'] for x in report['tasks'].values()))
            report['updated_at'] = time.time()
            atomic_json(report_path, report)
            if not args.watch or not any(x['status'] == 'waiting_native' for x in report['tasks'].values()):
                return
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, action='append')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--one')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--tme-only', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if not args.launch or not args.output:
            parser.error('--prepare requires --launch and --output')
        prepare(args.launch, args.output)
        return
    if not args.manifest:
        parser.error('--manifest required')
    manifest = json.loads(args.manifest.read_text())
    if identity({k: v for k, v in manifest.items() if k != 'identity'}) != manifest['identity']:
        raise ValueError('Baseline manifest identity mismatch')
    if not args.execute:
        print(json.dumps({'counts': manifest['counts'], 'policy': manifest['policy'], 'output': manifest['output']}, indent=2))
        return
    verify_files(manifest['source_sha256'])
    if args.one:
        try:
            result = run_one(manifest, args.one)
            status = {k: v for k, v in result.items() if k in ('status', 'identity', 'reason', 'wall_seconds')}
        except Exception as error:
            traceback.print_exc()
            status = {'status': 'failed', 'error': str(error)}
        atomic_json(Path(manifest['output'])/'task_status'/(args.one.replace(':', '_')+'.json'), status)
        print(json.dumps(status), flush=True)
        if status['status'] == 'failed':
            raise SystemExit(1)
    else:
        dispatch(manifest, args)


if __name__ == '__main__':
    main()
