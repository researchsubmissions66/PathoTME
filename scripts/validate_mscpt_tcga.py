#!/usr/bin/env python3
"""Lean offline campaign validation; consumes completed cached-encoder check evidence."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym']
from common.configuration import load_yaml_config
from pathotme.locked_tcga import load_launch,atomic_json,sha,rows,check_phases
from pathotme.mscpt_contract import validate_config
from pathotme.focus_contract import validate_donors


def validate(launch_path,test_report,out):
    launch=load_launch(launch_path);tests=json.loads(test_report.read_text())
    if tests['status']!='passed' or tests['passed']!=6:raise ValueError('six completed cached-encoder checks required')
    for path,value in tests['source_sha256'].items():
        if sha(path)!=value:raise ValueError(f'CPU-validated source changed: {path}')
    summary=[]
    for plan in launch['plans']:
        cfg=load_yaml_config(plan['config']);validate_config(cfg)
        phases={phase:rows(Path(cfg['split_dir'])/f"fold{plan['fold']}/{phase}.csv") for phase in ('train','val','test')}
        validate_donors(phases,json.loads(Path(plan['donor_maps']).read_text()))
        check_phases(phases,[r['slide_id'] for r in rows(cfg['dataset_csv'])])
        if cfg['k_start']!=plan['fold'] or cfg['k_end']!=plan['fold']+1:raise ValueError('fold range changed')
        if cfg['results_dir']!=plan['native_dir']:raise ValueError('native output changed')
        summary.append({k:plan[k] for k in ('cohort','encoder','fold','split_counts')})
    if len(summary)!=20 or launch['counts']['smoke_jobs']!=4:raise ValueError('bounded campaign count mismatch')
    sources={str(test_report):sha(test_report),**tests['source_sha256'],str(Path(__file__).resolve()):sha(__file__)}
    result=dict(status='passed',launch_identity=launch['identity'],source_sha256=sources,
        cached_encoder_checks=tests,plans=summary,real_gpu_smoke=False,
        note='Cached native weights and synthetic forward/backward checks; no completed GPU training or MSCPT attribution result.')
    atomic_json(out,result);print(json.dumps({'status':'passed','plans':len(summary),'validation':str(out)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',type=Path,required=True);p.add_argument('--cpu-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    validate(args.launch,args.cpu_report,args.output)
