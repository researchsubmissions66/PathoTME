"""Plan by default; submit exactly eight 4/8-shot smokes and eighty folds."""

import getpass
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
from datetime import datetime,timezone

from shot_contract import load_campaign,SHOTS,COHORTS,METHODS,ENCODERS,tag_for
from pathotme.locked_tcga import atomic_json,load_launch,PYTHON,PGVL,sha


def commands(campaign,path):
    output=Path(campaign['output']);worker=Path(__file__).parent/'run_additional_shots.py'
    children={(g['tag'],g['shots']):json.loads(Path(g['launch']).read_text()) for g in campaign['groups']}
    result=[]
    for smoke in [True,False]:
        for cohort in COHORTS:
            for method in METHODS:
                for encoder in ENCODERS:
                    pair=f'{cohort}_{method}_{encoder}'
                    for shot in [None] if smoke else SHOTS:
                        for fold in [0] if smoke else range(5):
                            launch=children[tag_for(method,encoder),4 if smoke else shot]
                            plan=next(p for p in launch['plans'] if (p['cohort'],p['method'],p['fold'])==(cohort,method,fold))
                            seconds=7200 if smoke else plan['time_limit_seconds']
                            hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
                            time_limit=f'{hours:02d}:{minutes:02d}:{seconds:02d}'
                            name=f"ptme-s48-{cohort}-{'vila' if method=='vila_mil' else 'mgp'}-{encoder}-"+('smk' if smoke else f's{shot}-f{fold}')
                            env=['env','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
                                'HF_HOME=/path/to/huggingface-cache','PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1',
                                'OMP_NUM_THREADS=8','MKL_NUM_THREADS=8','OPENBLAS_NUM_THREADS=8',
                                'LD_LIBRARY_PATH=/path/to/shared/envs/pgvl-gym/lib']
                            runtime=[PYTHON,'-u',str(worker),'--campaign',str(path.resolve()),'--cohort',cohort,
                                '--method',method,'--encoder',encoder,'--fold',str(fold),'--execute']
                            runtime+=['--smoke'] if smoke else ['--shots',str(shot)]
                            command=['sbatch','--parsable','--account=YOUR_SLURM_ACCOUNT','--partition=gpuA100x4',
                                '--gres=gpu:1','--cpus-per-task=8','--mem=48G',f'--time={time_limit}',
                                f'--job-name={name}',f'--chdir={PGVL}',f'--output={output}/logs/{name}-%j.out',
                                '--wrap',shlex.join(env+runtime)]
                            result.append({'name':name,'pair':pair,'cohort':cohort,'method':method,'encoder':encoder,
                                'shots':SHOTS if smoke else shot,'fold':fold,'smoke':smoke,'command':command,
                                'config':None if smoke else plan['config'],'time_limit':time_limit})
    if len(result)!=88 or sum(p['smoke'] for p in result)!=8:raise ValueError('only the authorized 88 allocations may launch')
    return result


def queue():
    return subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%j|%T|%R|%E|%Q'],
                          capture_output=True,text=True,check=True,timeout=30).stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    if args.submit and args.test_only:raise ValueError('choose one execution mode')
    campaign=load_campaign(args.campaign);plans=commands(campaign,args.campaign)
    output=Path(campaign['output'])
    if not args.submit and not args.test_only:
        print(json.dumps({'campaign_identity':campaign['identity'],'counts':campaign['counts'],'plans':plans},indent=2));return
    validation=json.loads(Path(campaign['validation']).read_text())
    if validation['status']!='passed':raise ValueError('focused validation required')
    if args.test_only:
        reports=[]
        for plan in plans:
            r=subprocess.run([plan['command'][0],'--test-only',*plan['command'][1:]],capture_output=True,text=True,timeout=60)
            reports.append({'name':plan['name'],'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
            if r.returncode:raise RuntimeError(reports[-1])
        atomic_json(output/'slurm_test_only.json',{'campaign_identity':campaign['identity'],'reports':reports})
        print(json.dumps({'test_only_passed':88}));return
    checks=json.loads((output/'slurm_test_only.json').read_text())
    if checks['campaign_identity']!=campaign['identity'] or len(checks['reports'])!=88 or any(r['returncode'] for r in checks['reports']):
        raise ValueError('exact 88 resource checks required')
    # All child runtime/source/asset bindings are rechecked before submission.
    for group in campaign['groups']:load_launch(group['launch'])
    ledger_path=output/'submission.json'
    with (output/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ledger_path.exists():raise FileExistsError('inspect existing submission ledger before any continuation')
        before=queue()
        if 'ptme-s48-' in before:raise ValueError('existing shot campaign found; refusing duplicates')
        held=json.loads((PGVL/'benchmarks/top_user_hold_20260910T231752Z.json').read_text())['target_job_ids']
        by_id={line.split('|')[0]:line.split('|') for line in before.splitlines()}
        assert len(held)==72
        for job in held:assert by_id[job][2]=='PENDING' and by_id[job][3].strip('()')=='JobHeldUser' and by_id[job][5]=='0'
        (output/'logs').mkdir(exist_ok=True)
        ledger={'created_at':datetime.now(timezone.utc).isoformat(),'campaign_identity':campaign['identity'],
            'campaign_path':str(args.campaign),'campaign_sha256':sha(args.campaign),
            'queue_before':before,'plans':plans,'top_held_ids':held}
        atomic_json(ledger_path,ledger);smokes={}
        for plan in plans:
            command=list(plan['command'])
            if not plan['smoke']:
                plan['smoke_job_id']=smokes[plan['pair']]
                command.insert(1,f"--dependency=afterok:{plan['smoke_job_id']}")
            plan['submitted_command']=command
            r=subprocess.run(command,capture_output=True,text=True,timeout=60)
            plan.update(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
            if r.returncode==0:
                job=r.stdout.strip().split(';')[0]
                if not job.isdigit():
                    atomic_json(ledger_path,ledger);raise ValueError('unexpected Slurm job ID')
                plan['job_id']=job
                if plan['smoke']:smokes[plan['pair']]=job
            atomic_json(ledger_path,ledger)
            print(json.dumps({k:plan[k] for k in ['name','job_id','smoke_job_id'] if k in plan}),flush=True)
            if r.returncode:raise RuntimeError('partial submission saved; inspect ledger')
        ledger['queue_after']=queue();ledger['status']='submitted'
        atomic_json(ledger_path,ledger)


if __name__=='__main__':main()
