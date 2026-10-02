#!/usr/bin/env python3
"""Retry only the 25 unstarted folds, with type-preserving runtime YAML."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym']
from pathotme.locked_tcga import ROOT, PGVL, RESULTS, PYTHON, atomic_json, identity, sha, load_launch
from pathotme.yaml_runtime_config import write_runtime_config

OUTPUT = RESULTS / 'tcga_numeric_retry_20260910_v1'
PARENTS = {'native': RESULTS / 'tcga_locked_16shot_20260909_v1',
           'cross': RESULTS / 'tcga_cross_encoders_16shot_20260909_v1'}
BLOCKED = ['21934327','21934329','21934331','21934333','21934335',
           '21934337','21934339','21934341','21934344','21934346']


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def prepare():
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing retry; inspect its ledger')
    OUTPUT.mkdir(parents=True)
    groups = []
    for tag, parent_root in PARENTS.items():
        parent_path = parent_root / 'launch.sealed.json'
        parent = load_launch(parent_path)
        submission = json.loads((parent_root / 'submission.json').read_text())
        lookup = {(p['cohort'],p['method'],p['fold']):p for p in submission['plans'] if not p['smoke']}
        launch = copy.deepcopy(parent); launch.pop('identity')
        out = OUTPUT / tag; launch['output'] = str(out)
        launch['plans'] = []
        for old in parent['plans']:
            # Completed BRCA ViLa folds and repaired BRCA PLIP MGPATH are excluded.
            if old['cohort'] == 'brca' and (tag == 'native' or old['method'] == 'vila_mil'):
                continue
            before = Path(old['output'])
            progress = [*before.rglob('*.pt'), *before.rglob('metrics.json'), *before.rglob('fold_complete.json')]
            if progress:
                raise ValueError(f'Unexpected saved progress; preserve and inspect: {progress}')
            cfg = json.loads(Path(old['config']).read_text())
            assert not old['native_reused'] and type(cfg['lr']) is float and type(cfg['weight_decay']) is float
            plan = copy.deepcopy(old)
            plan['output'] = str(out / f"runs/{old['cohort']}/{old['method']}/fold{old['fold']}")
            cfg['results_dir'] = str(Path(plan['output']) / 'native')
            config_path = out / f"configs/{old['cohort']}_{old['method']}_fold{old['fold']}.yaml"
            write_runtime_config(config_path, cfg)
            plan.update(config=str(config_path), native_dir=cfg['results_dir'],
                        retry_of_job=lookup[old['cohort'],old['method'],old['fold']]['job_id'],
                        retry_of_output=old['output'], retry_of_config=old['config'])
            if tag == 'cross' and old['method'] == 'mgpath':
                plan['smoke_checkpoint_dir'] = str(out / f"smokes/{old['cohort']}/mgpath/native_fixture")
            launch['plans'].append(plan)
            launch['file_sha256'][str(config_path)] = sha(config_path)
        launch['runtime_amendment'] = {
            'kind':'yaml_numeric_serialization_repair', 'parent_launch':str(parent_path),
            'parent_launch_identity':parent['identity'],
            'change':'Generate runtime YAML from the original typed JSON snapshot. Only results_dir changes; numeric values, model recipe, split, features and prompts are identical.',
            'progress':'All selected attempts have no checkpoint, metrics or completed fold; completed conditions excluded.',
            'authorization':'User: Fix and launch again. Preserve completed folds.'}
        sources = [parent_path, parent_root/'submission.json', Path(__file__).resolve(),
                   ROOT/'pathotme/yaml_runtime_config.py', ROOT/'scripts/check_tcga_numeric_retry.py',
                   ROOT/'TCGA_NUMERIC_RETRY.md']
        for path in sources: launch['file_sha256'][str(path)] = sha(path)
        n = len(launch['plans']); nsmokes = sum(p['fold'] == 0 for p in launch['plans'])
        launch['counts'] = {'fold_jobs':n, 'smoke_jobs':nsmokes, 'native_new':n,
                            'adapter_fits':3*n, 'native_reused':0}
        launch['identity'] = identity(launch)
        path = out/'launch.json'; atomic_json(path,launch)
        groups.append({'tag':tag,'launch':str(path),'identity':launch['identity']})
    manifest = {'created_at':datetime.now(timezone.utc).isoformat(),'groups':groups,
                'fold_jobs':25,'smoke_jobs':5,'preserved_completed_folds':15,
                'blocked_predecessors':BLOCKED}
    atomic_json(OUTPUT/'campaign.json',manifest)
    print(json.dumps(manifest),flush=True)


def commands(campaign):
    jobs=[]
    for g in campaign['groups']:
        launch=load_launch(g['launch']);out=Path(launch['output']);r=launch['protocol']['resources']
        env=['env','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
             'HF_HOME=/path/to/shared/.cache_huggingface','PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1',
             'OMP_NUM_THREADS=8','MKL_NUM_THREADS=8','OPENBLAS_NUM_THREADS=8',
             'LD_LIBRARY_PATH=/path/to/shared/envs/pgvl-gym/lib']
        runner=ROOT/('scripts/run_locked_tcga.py' if g['tag']=='native' else 'scripts/run_cross_encoder_tcga.py')
        for smoke in (True,False):
            for p in launch['plans']:
                if smoke and p['fold']!=0:continue
                pair=f"{g['tag']}_{p['cohort']}_{p['method']}"
                name=f"ptme-num-{g['tag'][0]}-{p['cohort']}-{'vila' if p['method']=='vila_mil' else 'mgp'}-"+('smk' if smoke else f"f{p['fold']}")
                runtime=[PYTHON,'-u',str(runner),'--launch',g['launch'],'--cohort',p['cohort'],
                         '--method',p['method'],'--fold',str(p['fold']),'--execute']+(['--smoke'] if smoke else [])
                cmd=['sbatch','--parsable',f"--account={r['account']}",f"--partition={r['partition']}",
                     '--gres=gpu:1','--cpus-per-task=8',f"--mem={r['mem']}",
                     f"--time={r['smoke_time'] if smoke else r['fold_time'][p['method']]}",
                     f'--job-name={name}',f'--chdir={PGVL}',f'--output={out}/logs/{name}-%j.out',
                     '--wrap',shlex.join(env+runtime)]
                jobs.append({'name':name,'pair':pair,'tag':g['tag'],'cohort':p['cohort'],'method':p['method'],
                             'fold':p['fold'],'smoke':smoke,'command':cmd,'launch':g['launch'],
                             'launch_identity':launch['identity'],'config':p['config'],
                             'retry_of_job':None if smoke else p['retry_of_job']})
    assert len(jobs)==30 and sum(j['smoke'] for j in jobs)==5
    return sorted(jobs,key=lambda j:not j['smoke'])


def submit(campaign,jobs):
    validation=json.loads((OUTPUT/'validation.json').read_text())
    assert validation['status']=='passed'
    assert validation['launch_identities']==[g['identity'] for g in campaign['groups']]
    if (OUTPUT/'submission.json').exists():raise FileExistsError('Inspect existing submission before retry')
    reports=[]
    for j in jobs:
        result=subprocess.run([j['command'][0],'--test-only',*j['command'][1:]],capture_output=True,text=True)
        reports.append({'name':j['name'],'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:raise RuntimeError(reports[-1])
    atomic_json(OUTPUT/'slurm_test_only.json',reports)
    queue=run(['squeue','-h','-u','anonymous','-o','%i|%u|%j|%T|%R|%E'])
    if 'ptme-num-' in queue:raise RuntimeError('Numeric retry already queued')
    lookup={r.split('|')[0]:r.split('|') for r in queue.splitlines()}
    for jid in BLOCKED:
        r=lookup[jid]
        assert r[1]=='anonymous' and r[2].startswith('ptme-cross-') and r[3]=='PENDING' and 'DependencyNeverSatisfied' in r[4] and '(failed)' in r[5],r
    failed=[j['retry_of_job'] for j in jobs if j['retry_of_job'] and j['retry_of_job'] not in BLOCKED]
    states=run(['sacct','-X','-n','-j',','.join(failed),'--format=JobIDRaw,State','-P'])
    assert {r.split('|')[0] for r in states.splitlines()}==set(failed)
    assert all(r.split('|')[1]=='FAILED' for r in states.splitlines())
    for g in campaign['groups']:(Path(g['launch']).parent/'logs').mkdir(exist_ok=True)
    ledger={'created_at':datetime.now(timezone.utc).isoformat(),'campaign':campaign,
            'validation_sha256':sha(OUTPUT/'validation.json'),'queue_before':queue,
            'failed_predecessor_states':states,'jobs':jobs}
    atomic_json(OUTPUT/'submission.json',ledger)
    cancel=['scancel',*BLOCKED];run(cancel)
    ledger['cancelled_blocked_predecessors']={'command':cancel,'ids':BLOCKED}
    atomic_json(OUTPUT/'submission.json',ledger)
    smokes={}
    for j in jobs:
        cmd=list(j['command'])
        if not j['smoke']:
            j['smoke_job_id']=smokes[j['pair']];cmd.insert(1,f"--dependency=afterok:{j['smoke_job_id']}")
        result=subprocess.run(cmd,capture_output=True,text=True)
        j.update(submitted_command=cmd,returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
        if result.returncode==0:
            jid=result.stdout.strip().split(';')[0];assert jid.isdigit();j['job_id']=jid
            if j['smoke']:smokes[j['pair']]=jid
        atomic_json(OUTPUT/'submission.json',ledger)
        print(json.dumps({k:j[k] for k in ['name','job_id','smoke_job_id','retry_of_job'] if k in j}),flush=True)
        if result.returncode:raise RuntimeError('Partial submission saved; inspect ledger')
    ledger['queue_after']=run(['squeue','-h','-j',','.join(j['job_id'] for j in jobs),'-o','%i|%j|%T|%R|%E|%C|%m|%l|%b'])
    atomic_json(OUTPUT/'submission.json',ledger)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    if args.prepare and args.submit:raise ValueError('Prepare, check, then submit')
    if args.prepare:prepare()
    else:
        campaign=json.loads((OUTPUT/'campaign.json').read_text());jobs=commands(campaign)
        atomic_json(OUTPUT/'dry_run.json',{'campaign':campaign,'jobs':jobs})
        if args.submit:
            with (OUTPUT/'submission.lock').open('a') as h:
                fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);submit(campaign,jobs)
        else:print(json.dumps({'jobs':len(jobs),'smokes':sum(j['smoke'] for j in jobs)}))
