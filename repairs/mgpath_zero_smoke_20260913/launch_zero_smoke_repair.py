"""Submit one replacement smoke and relink the ten existing MGPATH/PLIP folds."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone

from run_zero_smoke_repair import load_amendment
from pathotme.locked_tcga import atomic_json, sha


def query_queue():
    result=subprocess.run(['squeue','-h','-u','anonymous','-o','%i|%j|%T|%R|%E|%Q'],
                          capture_output=True,text=True,check=True,timeout=30)
    fields=['id','name','state','reason','dependency','priority']
    return {r['id']:r for r in (dict(zip(fields,line.split('|'))) for line in result.stdout.splitlines())}


def commands(amendment,path):
    output=Path(amendment['output'])
    worker=Path(__file__).parent/'run_zero_smoke_repair.py'
    original=amendment['original_smoke']
    name='ptme-s48-nsclc-mgp-plip-zero-retry'
    command=[f'--job-name={name}' if x.startswith('--job-name=') else
             f'--output={output}/logs/{name}-%j.out' if x.startswith('--output=') else
             '--time=00:45:00' if x.startswith('--time=') else x
             for x in original['command']]
    runtime=shlex.split(command[-1])
    index=runtime.index('/path/to/PathoTME/campaigns/tcga_4_8shot_20260912/run_additional_shots.py')
    runtime=runtime[:index]+[str(worker),'--amendment',str(path.resolve()),'--execute']
    command[-1]=shlex.join(runtime)
    folds=amendment['folds']
    if original['job_id']!='22017480' or {p['job_id'] for p in folds} != {str(i) for i in range(22017525,22017535)}:
        raise ValueError('repair scope changed')
    return [{'pair':'nsclc_mgpath_plip','old_smoke_job_id':original['job_id'],
             'name':name,'command':command,'folds':folds}]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment',type=Path,required=True)
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    if args.submit and args.test_only:raise ValueError('choose one execution mode')
    amendment,launch=load_amendment(args.amendment)
    plans=commands(amendment,args.amendment)
    if not args.submit and not args.test_only:
        print(json.dumps({'identity':amendment['identity'],'plans':plans},indent=2));return
    output=Path(amendment['output'])
    if args.test_only:
        reports=[]
        for plan in plans:
            result=subprocess.run([plan['command'][0],'--test-only',*plan['command'][1:]],
                                   capture_output=True,text=True,timeout=60)
            reports.append({'name':plan['name'],'returncode':result.returncode,
                            'stdout':result.stdout,'stderr':result.stderr})
            if result.returncode:raise RuntimeError(reports[-1])
        atomic_json(output/'slurm_test_only.json',{'identity':amendment['identity'],'reports':reports})
        print(json.dumps({'test_only_passed':len(reports)}));return
    checks=json.loads((output/'slurm_test_only.json').read_text())
    if checks['identity']!=amendment['identity'] or len(checks['reports'])!=1 or any(r['returncode'] for r in checks['reports']):
        raise ValueError('one exact resource check required')
    validation=json.loads(Path(amendment['validation']).read_text())
    if validation['status']!='passed':raise ValueError('passing repair validation required')
    ledger_path=output/'submission.json'
    with (output/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ledger_path.exists():raise FileExistsError('inspect existing ledger before any continuation')
        queue=query_queue()
        if any(r['name'].startswith('ptme-s48-nsclc-mgp-plip-zero-retry') for r in queue.values()):
            raise ValueError('existing repair smoke found')
        for plan in plans:
            for fold in plan['folds']:
                row=queue[fold['job_id']]
                if (row['name']!=fold['name'] or row['state']!='PENDING'
                    or f"afterok:{plan['old_smoke_job_id']}(failed)" not in row['dependency']):
                    raise ValueError(f'blocked fold identity changed: {row}')
        for job in amendment['top_held_ids']:
            row=queue[job]
            if row['state']!='PENDING' or row['reason'].strip('()')!='JobHeldUser' or row['priority']!='0':
                raise ValueError('TOP hold changed before repair')
        ids=','.join(p['old_smoke_job_id'] for p in plans)
        result=subprocess.run(['sacct','-n','-X','-P','-j',ids,'--format=JobID,State,ExitCode'],
                              capture_output=True,text=True,check=True,timeout=30)
        states={line.split('|')[0]:line.split('|')[1] for line in result.stdout.splitlines() if line.strip()}
        if any(states.get(p['old_smoke_job_id'])!='FAILED' for p in plans):
            raise ValueError('original smokes are not all terminal failed jobs')
        (output/'logs').mkdir(exist_ok=True)
        ledger={'created_at':datetime.now(timezone.utc).isoformat(),'amendment_identity':amendment['identity'],
                'amendment_path':str(args.amendment),'amendment_sha256':sha(args.amendment),
                'queue_before':queue,'plans':plans,'dependency_updates':[]}
        atomic_json(ledger_path,ledger)
        for plan in plans:
            result=subprocess.run(plan['command'],capture_output=True,text=True,timeout=60)
            plan.update(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
            if result.returncode:
                atomic_json(ledger_path,ledger);raise RuntimeError('partial submission saved; inspect ledger')
            job=result.stdout.strip().split(';')[0]
            if not job.isdigit():
                atomic_json(ledger_path,ledger);raise ValueError('unexpected Slurm job ID')
            plan['job_id']=job
            atomic_json(ledger_path,ledger)
            print(json.dumps({'smoke':plan['pair'],'old_id':plan['old_smoke_job_id'],'new_id':job}),flush=True)
            for fold in plan['folds']:
                row=query_queue()[fold['job_id']]
                if row['state']!='PENDING' or row['name']!=fold['name'] or f"afterok:{plan['old_smoke_job_id']}(failed)" not in row['dependency']:
                    raise ValueError('fold changed before dependency update; partial ledger retained')
                command=['scontrol','update',f"JobId={fold['job_id']}",f'Dependency=afterok:{job}']
                result=subprocess.run(command,capture_output=True,text=True,timeout=30)
                ledger['dependency_updates'].append({'job_id':fold['job_id'],'old_smoke':plan['old_smoke_job_id'],
                    'new_smoke':job,'command':command,'returncode':result.returncode,
                    'stdout':result.stdout,'stderr':result.stderr})
                atomic_json(ledger_path,ledger)
                if result.returncode:raise RuntimeError('partial dependency update saved; inspect ledger')
        queue=query_queue()
        for plan in plans:
            for fold in plan['folds']:
                if f"afterok:{plan['job_id']}" not in queue[fold['job_id']]['dependency']:
                    raise ValueError('new smoke dependency not verified')
        for job in amendment['top_held_ids']:
            row=queue[job]
            if row['state']!='PENDING' or row['reason'].strip('()')!='JobHeldUser' or row['priority']!='0':
                raise ValueError('TOP hold verification failed')
        ledger['queue_after']=queue
        ledger['status']='submitted_and_dependencies_verified'
        atomic_json(ledger_path,ledger)
        print(json.dumps({'new_smokes':1,'existing_folds_relinked':10,'top_holds_verified':len(amendment['top_held_ids'])}),flush=True)


if __name__=='__main__':main()
