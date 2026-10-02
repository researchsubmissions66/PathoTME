"""Retry the unfinished 8-shot smoke, preserving the completed 4-shot report."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys

sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym',
               '/path/to/PathoTME/scripts',
               '/path/to/PathoTME/campaigns/tcga_4_8shot_20260912']
from pathotme.locked_tcga import atomic_json, identity, load_launch, sha, verify_files
from shot_contract import load_campaign


def load_amendment(path):
    amendment = json.loads(Path(path).read_text())
    expected = amendment.pop('identity')
    if identity(amendment) != expected:
        raise ValueError('amendment identity mismatch')
    amendment['identity'] = expected
    verify_files(amendment['file_sha256'])
    campaign = load_campaign(amendment['campaign'])
    if campaign['identity'] != amendment['campaign_identity']:
        raise ValueError('campaign changed')
    launches = {}
    for shots in (4, 8):
        group = next(g for g in campaign['groups'] if (g['tag'], g['shots']) == ('native', shots))
        launch = load_launch(group['launch'])
        if launch['identity'] != group['identity']:
            raise ValueError('child launch changed')
        launches[shots] = launch
    retained = json.loads(Path(amendment['retained_4shot_report']).read_text())
    if (retained['status'] != 'smoke_passed' or retained['launch_identity'] != launches[4]['identity']
            or retained['shots'] != 4 or retained['slurm_job_id'] != '22017480'
            or set(retained['arms']) != {'zero', 'actual', 'shuffled'}):
        raise ValueError('retained 4-shot smoke provenance mismatch')
    verify_files(retained['artifact_sha256'])
    for arm in retained['arms'].values():
        if arm['status'] != 'completed':
            raise ValueError('incomplete retained arm')
        verify_files(arm['artifact_sha256'])
    return amendment, launches


def execute(args, amendment, launch, plan):
    import torch
    from common.configuration import load_dotenv, load_yaml_config
    from run_locked_tcga import check_features
    from repaired_shot_smoke import arm_smoke
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('allocated GPU job required')
    load_dotenv('/path/to/PGVL-Gym/.env')
    cfg = load_yaml_config(plan['config'])
    check_features(plan, cfg)
    checkpoint = Path(plan['smoke_checkpoint_dir']) / 'fold0_best.pt'
    native_hashes = {str(checkpoint): plan['smoke_checkpoint_sha256']}
    verify_files(native_hashes)
    canonical = Path(launch['output']) / 'smokes/nsclc/mgpath/smoke_report.json'
    root = Path(amendment['output']) / 'smokes/nsclc/mgpath'
    if canonical.exists() or root.exists():
        raise FileExistsError('inspect prior smoke output before another attempt')
    private_launch = {**launch, 'output': amendment['output'], 'amendment_identity': amendment['identity']}
    results = {}
    for arm in ['zero', 'actual', 'shuffled']:
        results[arm] = arm_smoke(private_launch, plan, cfg, arm, args.device, native_hashes)
        gc.collect()
        torch.cuda.empty_cache()
    if len({r['initial_conditioner_sha256'] for r in results.values()}) != 1:
        raise ValueError('initial conditioners differ')
    # Verify preserved 4-shot evidence again before making either shot runnable.
    load_amendment(args.amendment)
    report = {'status': 'smoke_passed', 'launch_identity': launch['identity'],
              'amendment_identity': amendment['identity'], 'amendment_path': str(args.amendment.resolve()),
              'cohort': 'nsclc', 'method': 'mgpath', 'encoder': 'plip', 'shots': 8,
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'arms': results,
              'retained_4shot_report': amendment['retained_4shot_report'],
              'note': 'Smoke assertion repair only; unchanged training recipe. 16-shot checkpoint is a structural fixture.',
              'artifact_sha256': {**native_hashes, **amendment['file_sha256'],
                  str(args.amendment.resolve()): sha(args.amendment),
                  **{str(root / arm / 'metrics.json'): sha(root / arm / 'metrics.json') for arm in results},
                  **{p: h for result in results.values() for p, h in result['artifact_sha256'].items()}}}
    atomic_json(root / 'smoke_report.json', report)
    atomic_json(canonical, report)
    print(json.dumps({'status': 'smoke_passed', 'shots_retried': 8,
                      'shots_retained': 4, 'canonical_report': str(canonical)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    amendment, launches = load_amendment(args.amendment)
    launch = launches[8]
    plan = next(p for p in launch['plans'] if (p['cohort'], p['method'], p['fold']) == ('nsclc', 'mgpath', 0))
    if plan['encoder'] != 'plip' or plan['shots'] != 8:
        raise ValueError('wrong repair scope')
    if not args.execute:
        print(json.dumps({'amendment_identity': amendment['identity'], 'retry_shots': 8,
                          'retained_shots': 4, 'plan': plan}))
        return
    with (Path(launch['output']) / 'locks/nsclc_mgpath_smoke.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, amendment, launch, plan)


if __name__ == '__main__':
    main()
