"""Test and submit the bounded 392-job campaign, with durable duplicate protection."""
import argparse
import fcntl
import json
import math
import os
import shlex
import subprocess
from pathlib import Path
from contract import *

PYTHON='/path/to/shared/envs/pgvl-gym/bin/python'

def run(args):return subprocess.run(args,text=True,capture_output=True,check=True).stdout.strip()
def hms(seconds):return f'{seconds//3600:02d}:{seconds%3600//60:02d}:00'
def queue():
    text=run(['squeue','-u','anonymous','-h','-o','%i|%j|%T|%r|%Q|%E'])
    return [dict(zip(['id','name','state','reason','priority','dependency'],line.split('|'))) for line in text.splitlines()]
def holds(snapshot):
    top=[r for r in snapshot if 'top' in r['name'].lower()]
    if len(top)!=76 or any(r['state']!='PENDING' or r['reason']!='JobHeldUser' or r['priority']!='0' for r in top):
        raise ValueError('TOP hold invariant changed; do not mutate TOP')
    return sorted(r['id'] for r in top)

def requests(campaign):
    result=[]
    from common.configuration import load_dotenv
    load_dotenv(PGVL/'.env')
    ram={}
    for plan in campaign['plans']:
        if plan['method']!='hive_mil':continue
        cfg=read_config(plan['config']);inventory=json.loads(Path(plan['feature_inventory']).read_text())
        cache_bytes=0
        for phase in ['train','val']:
            for row in rows(Path(cfg['split_dir'])/f"fold{plan['fold']}/{phase}.csv"):
                bound=inventory['slides'][row['slide_id']]['files'] if 'slides' in inventory else inventory
                n,d=bound[os.path.expandvars(row[cfg['feature_path_column_l']])]['shape']
                children=cfg.get('max_children',16)
                cache_bytes+=n*((1+children)*d*4+children)
        gib=cache_bytes/1024**3
        ram[plan['id']]={'mem_gib':max(plan['resources']['mem_gib'],math.ceil((24+1.25*gib)/8)*8),
            'hive_cached_hierarchy_upper_gib':gib,
            'hive_ram_policy':'All low parents retained upper bound for train+val float32 tensors and masks; 25% cache margin plus24GiB process/transient allowance.'}
    for group in campaign['groups']:
        plans=[p for p in campaign['plans'] if p['group']==group['id']]
        result.append({'key':group['id']+'_smoke','group':group['id'],'smoke':True,'shots':None,'fold':0,
            'mem_gib':max(p['resources']['mem_gib'] for p in plans),
            'time_min_seconds':3600,'time_limit_seconds':5400,'cpus':8})
    for plan in campaign['plans']:
        result.append({'key':plan['id'],'group':plan['group'],'smoke':False,
                       'shots':plan['shots'],'fold':plan['fold'],**plan['resources'],**ram.get(plan['id'],{})})
    assert len(result)==392
    return result

def command(request,dependency=None):
    key=request['key'];script=OUTPUT/'sbatch'/f'{key}.sh'
    args=[PYTHON,'-u',str(RUNTIME/'runtime.py'),'--campaign',str(OUTPUT/'campaign.json'),
          '--group',request['group'],'--fold',str(request['fold']),'--execute']
    args+=['--smoke'] if request['smoke'] else ['--shots',str(request['shots'])]
    content='#!/bin/bash\nset -euo pipefail\ncd /path/to/PGVL-Gym\n'
    env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/path/to/shared/.cache_huggingface',
         'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
         'OPENBLAS_NUM_THREADS':'8','LD_LIBRARY_PATH':'/path/to/shared/envs/pgvl-gym/lib'}
    content+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())+'\nexec '+shlex.join(args)+'\n'
    script.parent.mkdir(exist_ok=True);(OUTPUT/'logs').mkdir(exist_ok=True)
    if script.exists() and script.read_text()!=content:raise ValueError('Job script changed')
    script.write_text(content)
    cmd=['sbatch','--parsable','--partition=gpuA100x4','--account=shared-delta-gpu','--nodes=1','--ntasks=1',
         '--gpus-per-node=1',f'--cpus-per-task={request["cpus"]}',f'--mem={request["mem_gib"]}G',
         '--time='+hms(request['time_limit_seconds']),'--time-min='+hms(request['time_min_seconds']),
         '--job-name=ptme3264_'+key,'--comment=PathoTME 32/64-shot matched native and adapters',
         '--output='+str(OUTPUT/'logs'/f'{key}_%j.out'),'--error='+str(OUTPUT/'logs'/f'{key}_%j.err')]
    if dependency:cmd.append('--dependency=afterok:'+str(dependency))
    return cmd+[str(script)]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    campaign=load_campaign(OUTPUT/'campaign.json')
    validation=json.loads((OUTPUT/'validation.json').read_text())
    if validation['status']!='passed' or validation['campaign_identity']!=campaign['identity']:raise ValueError('CPU preflight required')
    with (OUTPUT/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initial=queue();top=holds(initial)
        ledger_path=OUTPUT/'launch.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'campaign_identity':campaign['identity'],'jobs':[],
            'top_held_ids':top,'queue_before':initial,'validation_sha256':sha(OUTPUT/'validation.json')}
        if ledger['campaign_identity']!=campaign['identity'] or ledger['top_held_ids']!=top:raise ValueError('Launch identity mismatch')
        known={str(r['job_id']) for r in ledger['jobs']}
        if any(r['name'].startswith('ptme3264_') and r['id'] not in known for r in initial):raise ValueError('Unrecorded existing high-shot job')
        requests_list=requests(campaign)
        atomic_json(OUTPUT/'scheduling_plan.json',{'campaign_identity':campaign['identity'],'requests':requests_list,
            'note':'HiVE cache-aware RAM supersedes initial measured-RSS estimates; all scientific fields unchanged.'})
        # Complete all exact scheduling dry-runs before any submission.
        prior=OUTPUT/'slurm_resource_checks.json'
        cached=json.loads(prior.read_text())['checks'] if prior.exists() else []
        checks=[]
        for r in requests_list:
            cmd=command(r)
            match=next((c for c in cached if c['key']==r['key'] and c['command']==cmd
                        and c.get('script_sha256')==sha(cmd[-1])),None)
            if match:checks.append(match)
            else:
                output=run([cmd[0],'--test-only',*cmd[1:]])
                checks.append({'key':r['key'],'command':cmd,'output':output,'script_sha256':sha(cmd[-1])})
        atomic_json(OUTPUT/'slurm_preflight.json',{'campaign_identity':campaign['identity'],'checks':checks})
        print(json.dumps({'slurm_test_only_passed':len(checks),'submit':args.submit}),flush=True)
        if not args.submit:return
        for request in requests_list:
            if any(j['key']==request['key'] for j in ledger['jobs']):continue
            dependency=None if request['smoke'] else next(j['job_id'] for j in ledger['jobs'] if j['key']==request['group']+'_smoke')
            cmd=command(request,dependency)
            # Record intent before submission; an interrupted pending intent needs manual accounting reconciliation.
            if ledger.get('pending_intent'):raise ValueError('Unresolved submission intent')
            ledger['pending_intent']={'key':request['key'],'command':cmd};atomic_json(ledger_path,ledger)
            job=run(cmd).split(';')[0]
            if not job.isdigit():raise ValueError('Invalid sbatch result')
            ledger['jobs'].append({**request,'job_id':job,'dependency':dependency,'command':cmd})
            ledger.pop('pending_intent');atomic_json(ledger_path,ledger)
            print(json.dumps({'submitted':request['key'],'job_id':job}),flush=True)
        final=queue();assert holds(final)==top
        details={}
        for job in ledger['jobs']:
            detail=run(['scontrol','show','job','-o',job['job_id']]);details[job['job_id']]=detail
            if job['dependency'] and ('afterok:'+job['dependency']) not in detail:
                # A successful smoke can already have satisfied and removed the dependency.
                state=run(['sacct','-X','-n','-P','-j',job['dependency'],'--format=State']).splitlines()
                if not state or state[0].strip('|')!='COMPLETED':raise ValueError('Wrong smoke dependency')
            if 'JobState=PENDING' not in detail and 'JobState=RUNNING' not in detail and 'JobState=COMPLETED' not in detail:
                raise ValueError('Unexpected launch state')
        ledger['status']='submitted_and_verified';ledger['queue_after']=final
        atomic_json(ledger_path,ledger);atomic_json(OUTPUT/'verification.json',{'campaign_identity':campaign['identity'],
            'jobs':details,'top_held_ids':top,'count':len(ledger['jobs']),'status':'passed'})
        print(json.dumps({'submitted_and_verified':len(ledger['jobs']),'smokes':42,'folds':350}),flush=True)

if __name__=='__main__':main()
