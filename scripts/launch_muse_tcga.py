#!/usr/bin/env python3
"""Dry-run by default; submit exactly four smokes and twenty dependent folds."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym']
from pathotme.locked_tcga import ROOT, PGVL, PYTHON, atomic_json, load_launch, sha, verify_files


def commands(launch, path):
    resource = launch['protocol']['resources']; output = Path(launch['output'])
    plans = []
    env = ['env', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
           'HF_HOME=/path/to/shared/.cache_huggingface', 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1',
           'OMP_NUM_THREADS=8', 'MKL_NUM_THREADS=8', 'OPENBLAS_NUM_THREADS=8',
           'LD_LIBRARY_PATH=/path/to/shared/envs/pgvl-gym/lib']
    for smoke in [True, False]:
        for plan in launch['plans']:
            if smoke and plan['fold'] != 0:
                continue
            cohort, method, fold = plan['cohort'], plan['method'], plan['fold']
            encoder = plan['encoder']
            key = f'{cohort}_{encoder}'
            name = f"ptme-muse-{cohort}-{encoder}-{'smk' if smoke else 'f'+str(fold)}"
            runtime = [PYTHON, '-u', str(ROOT / 'scripts/run_muse_tcga.py'), '--launch', str(path),
                       '--cohort', cohort, '--encoder', encoder, '--fold', str(fold), '--execute']
            if smoke:
                runtime.append('--smoke')
            command = ['sbatch', '--parsable', f"--account={resource['account']}",
                f"--partition={resource['partition']}", f"--gres=gpu:{resource['gpus']}", f"--cpus-per-task={resource['cpus']}",
                f"--mem={resource['mem']}",
                f"--time={resource['smoke_time'] if smoke else resource['fold_time'][method]}",
                f'--job-name={name}', f'--chdir={PGVL}', f'--output={output}/logs/{name}-%j.out',
                '--wrap', shlex.join(env + runtime)]
            plans.append({'name': name, 'pair': key, 'cohort': cohort, 'method': method,
                          'encoder': encoder, 'fold': fold, 'smoke': smoke, 'command': command,
                          'depends_on_smoke_pair': None if smoke else key})
    if len(plans) != 24 or sum(p['smoke'] for p in plans) != 4:
        raise ValueError('job count exceeded frozen 24-job boundary')
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    launch = load_launch(args.launch)
    validation = json.loads(args.validation.read_text())
    if validation['status'] != 'passed' or validation.get('launch_identity') != launch['identity']:
        raise ValueError('passing museed validation required')
    verify_files(validation['source_sha256'])
    plans = commands(launch, args.launch.resolve())
    output = Path(launch['output']); ledger_path = output / 'submission.json'
    if not args.submit and not args.test_only:
        print(json.dumps({'launch_identity': launch['identity'], 'counts': launch['counts'], 'plans': plans}, indent=2))
        return
    if args.submit and args.test_only:
        raise ValueError('test-only and submit are mutually exclusive')
    if args.test_only:
        reports = []
        for plan in plans:
            result = subprocess.run([*plan['command'][:1], '--test-only', *plan['command'][1:]], capture_output=True, text=True, timeout=60)
            reports.append({'name': plan['name'], 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
            if result.returncode:
                raise RuntimeError(reports[-1])
        atomic_json(output / 'slurm_test_only.json', {'launch_identity': launch['identity'], 'reports': reports})
        print(json.dumps({'test_only_passed': len(reports)})); return
    resource_check = json.loads((output / 'slurm_test_only.json').read_text())
    if resource_check['launch_identity'] != launch['identity'] or len(resource_check['reports']) != 24 or any(r['returncode'] for r in resource_check['reports']):
        raise ValueError('exact 24-job Slurm test-only gate required')
    (output / 'logs').mkdir(exist_ok=True)
    with (output / 'submission.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger_path.exists():
            raise FileExistsError('submission ledger exists; inspect it before any continuation')
        queue = subprocess.run(['squeue', '-h', '-u', 'anonymous', '-o', '%i|%j|%T|%R'], capture_output=True, text=True, check=True, timeout=30).stdout
        if any('ptme-muse-' in line for line in queue.splitlines()):
            raise ValueError('existing locked-study jobs found; refusing duplicates')
        ledger = {'created_at': datetime.now(timezone.utc).isoformat(), 'launch': str(args.launch),
            'launch_identity': launch['identity'], 'validation_sha256': sha(args.validation),
            'queue_before': queue, 'plans': plans}
        atomic_json(ledger_path, ledger)
        smokes = {}
        for plan in plans:
            command = list(plan['command'])
            if not plan['smoke']:
                dependency = smokes[plan['pair']]
                command.insert(1, f'--dependency=afterok:{dependency}')
                plan['smoke_job_id'] = dependency
            plan['submitted_command'] = command
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            plan.update({'returncode': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()})
            if not result.returncode:
                job_id = result.stdout.strip().split(';')[0]
                if not job_id.isdigit():
                    atomic_json(ledger_path, ledger); raise ValueError('unrecognized Slurm submission result')
                plan['job_id'] = job_id
                if plan['smoke']:
                    smokes[plan['pair']] = job_id
            atomic_json(ledger_path, ledger)
            print(json.dumps({k: plan[k] for k in ['name', 'job_id', 'smoke_job_id'] if k in plan}), flush=True)
            if result.returncode:
                raise RuntimeError('partial submission saved; inspect ledger before continuing')
        ledger['queue_after'] = subprocess.run(['squeue', '-h', '-j', ','.join(p['job_id'] for p in plans),
            '-o', '%i|%j|%T|%R|%E'], capture_output=True, text=True, check=True, timeout=30).stdout
        atomic_json(ledger_path, ledger)


if __name__ == '__main__':
    main()
