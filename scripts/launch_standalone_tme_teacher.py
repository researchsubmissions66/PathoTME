#!/usr/bin/env python3
"""Plan ten new folds plus one smoke; never resubmit the ten completed controls."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess

from run_standalone_tme_teacher import (ROOT, PGVL, CONDITIONS, NEW_CONDITIONS,
    environment, load_yaml_config, preflight, sha, verify_completed)
from launch_controlled_vila import persist


def build_plan(config, with_external=False):
    """Return config-bound single-GPU commands without external writes."""
    config = Path(config).resolve(); contract = load_yaml_config(config)
    matched = load_yaml_config(contract['matched_contract'])
    python = str(Path(os.environ['PGVL_CONDA_ENV'])/'bin/python')
    runner = str(ROOT/'scripts/run_standalone_tme_teacher.py')
    env = ['env', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
        'HF_HOME=/path/to/huggingface-cache', 'PYTHONNOUSERSITE=1',
        'TOKENIZERS_PARALLELISM=false', 'OMP_NUM_THREADS=6', 'MKL_NUM_THREADS=6',
        'LD_LIBRARY_PATH='+str(Path(python).parent.parent/'lib')+
        (':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')]
    reused = []; plans = []

    def component(fold, condition, smoke=False):
        _, _, payload, identity, output = preflight(contract, fold, condition, smoke)
        completed = bool(verify_completed(output, identity, smoke))
        if not completed and output.exists() and any(p.name != '.run.lock' for p in output.iterdir()):
            raise ValueError(f'partial result preserved: {output}')
        command = [python, '-u', runner, '--config', str(config), '--fold', str(fold),
                   '--condition', condition, '--expected-identity', identity]
        if smoke:
            command.append('--smoke-only')
        return {'fold': fold, 'condition': condition, 'identity': identity, 'output': str(output),
                'completed': completed, 'inputs': payload, 'command': command}

    for fold in contract['folds']:
        for condition in CONDITIONS[:2]:
            c = component(fold, condition)
            if not c['completed']:
                raise ValueError('matched controls must already be completed')
            c.pop('command'); reused.append(c)
    smokes = [component(contract['folds'][0], c, True) for c in NEW_CONDITIONS]
    plans.append({'key': 'smoke', 'time': '00:20:00', 'components': smokes, 'depends_on': []})
    for fold in contract['folds']:
        for condition in NEW_CONDITIONS:
            plans.append({'key': f'{condition}-f{fold}', 'time': '00:45:00',
                          'components': [component(fold, condition)], 'depends_on': ['smoke']})
    target_path = Path(matched['external']['contract'])
    if with_external:
        from eval_wsi_only import table, validate_target
        target = load_yaml_config(target_path)
        if target['status'] != 'ready':
            raise ValueError('external target is not ready')
        for file_key, hash_key in [('manifest','manifest_sha256'), ('label_source','label_source_sha256'),
                                   ('feature_audit','feature_audit_sha256')]:
            if sha(target[file_key]) != target[hash_key]:
                raise ValueError('frozen external assets changed')
        cfg = load_yaml_config(matched['base_config'])
        validate_target(target, table(target['manifest']),
                        {r['case_id'] for r in table(cfg['dataset_csv'])}, cfg)
        audit = json.loads(Path(target['feature_audit']).read_text())
        if audit['status'] != 'passed' or audit['manifest_sha256'] != target['manifest_sha256']:
            raise ValueError('external header audit is not valid for this manifest')
        for path, checked in audit['files'].items():
            stat = Path(path).stat()
            if stat.st_size != checked['size'] or stat.st_mtime_ns != checked['mtime_ns']:
                raise ValueError('external features changed since audited headers')
    for plan in plans:
        commands = [c['command'] for c in plan['components'] if not c['completed']]
        plan['external_components'] = []
        if with_external and plan['key'] != 'smoke':
            c = plan['components'][0]
            output = Path(contract['results_root'])/'external/cptac_brca'/c['condition']/f"fold{c['fold']}"
            command = [python, '-u', str(ROOT/'scripts/eval_wsi_only.py'), '--source-result', c['output'],
                '--target-contract', str(target_path), '--output', str(output), '--execute',
                '--expected-source-identity', c['identity'], '--expected-code-sha256', sha(ROOT/'scripts/eval_wsi_only.py'),
                '--expected-target-sha256', sha(target_path)]
            complete = False
            if c['completed']:
                from eval_wsi_only import preflight as external_preflight
                *_, identity = external_preflight(c['output'], target_path)
                complete = bool(verify_completed(output, identity))
            if not complete:
                if output.exists() and any(p.name != '.run.lock' for p in output.iterdir()):
                    raise ValueError('partial external output preserved')
                commands.append(command)
            plan['external_components'].append({'output': str(output), 'command': command, 'completed': complete})
        name = 'ptme-stt-v1-'+plan['key']
        logs = Path(os.environ['PATHOTME_RESULTS_ROOT'])/'logs'
        plan.update(job_name=name, skip_completed=not commands)
        wrap = 'set -eu\n'+'\n'.join(shlex.join(env+cmd) for cmd in commands)
        plan['sbatch'] = ['sbatch', '--parsable', '--account=YOUR_SLURM_ACCOUNT', '--partition=gpuA100x4',
            '--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=6', '--mem=24G',
            f"--time={plan['time']}", f'--job-name={name}', f'--chdir={PGVL}',
            f'--output={logs}/{name}-%j.out', '--comment=pathotme-standalone-tme-teacher-v1', '--wrap', wrap]
    return reused, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/vila_conch_brca_tme_teacher_16shot.yaml')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--with-external', action='store_true'); parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(); environment()
    if args.report.exists():
        raise FileExistsError('preserve previous ledger')
    # Dry-run never creates locks or output directories on shared storage.
    lock = None
    try:
        if args.submit:
            lock = (ROOT/'.standalone_teacher_launch.lock').open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reused, plans = build_plan(args.config, args.with_external)
        if args.submit:
            names = subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%j'], text=True).splitlines()
            if any(p['job_name'] in names for p in plans):
                raise ValueError('campaign already active; do not duplicate jobs')
            (Path(os.environ['PATHOTME_RESULTS_ROOT'])/'logs').mkdir(parents=True, exist_ok=True)
        ledger = {'schema': 'pathotme.standalone_teacher.v1', 'dry_run': not args.submit,
                  'created_at': datetime.datetime.now().astimezone().isoformat(), 'launcher_sha256': sha(__file__),
                  'reused_controls': reused, 'plans': plans, 'external_requested': args.with_external}
        with args.report.open('x') as handle:
            json.dump(ledger, handle, indent=2)
        submitted = {}; completed = set()
        for row in plans:
            if row['skip_completed']:
                row['status'] = 'skipped_completed'; completed.add(row['key'])
            elif not args.submit:
                row['status'] = 'dry_run_ready'
            else:
                command = row['sbatch'].copy(); dependencies = []
                for key in row['depends_on']:
                    if key in submitted:
                        dependencies.append(submitted[key])
                    elif key not in completed:
                        raise ValueError('unresolved smoke dependency')
                if dependencies:
                    command.insert(1, '--dependency=afterok:'+':'.join(dependencies))
                row.update(status='submitting', submitted_command=command); persist(args.report, ledger)
                proc = subprocess.run(command, capture_output=True, text=True)
                row.update(returncode=proc.returncode, stdout=proc.stdout.strip(), stderr=proc.stderr.strip())
                job = proc.stdout.strip().split(';')[0]
                if proc.returncode or not job.isdigit():
                    row['status'] = 'submission_failed_or_ambiguous'; persist(args.report, ledger)
                    raise RuntimeError('inspect ledger/Slurm before retrying')
                row.update(status='submitted', job_id=job); submitted[row['key']] = job
            persist(args.report, ledger)
            print(json.dumps({k: row[k] for k in ('key','status','time','job_id') if k in row}), flush=True)
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
