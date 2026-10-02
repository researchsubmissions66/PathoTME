#!/usr/bin/env python3
"""Export structural zero TME sensitivity for completed zero-input adapters."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from pathotme.attribution_io import load_bound_run
from score_core import VERSION,sha,dump,write_csv,make_units,rank_fold,aggregate_ranks
from run_adapter import layout

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if any(out==r or r in out.parents for r in [ROOT,ROOT.parent/'PGVL-Gym']):raise ValueError('Use private results storage')
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text());all_ranks=[];records=[]
    for task in manifest['fusion_tasks']:
        completion=Path(task['plan']['output'])/'fold_complete.json'
        if not completion.exists():continue
        b=load_bound_run(completion,'zero',Path(task['launch']));cfg=b['config']
        if cfg['tme_mode']!='zero':raise ValueError('Not a zero-input adapter')
        names,groups=layout(cfg);units=make_units(names,groups)
        rows=[{'slide_id':r['slide_id'],'case_id':r['case_id'],'label':int(cfg['label_dict'].get(r['label'],r['label']))} for r in b['test_rows']]
        meta={'variant':'PathoTME-Adapter-Zero','cohort':cfg['task'],'panel':'brca_morph64_v1' if len(names)==64 else 'shared_core62_v1',
              'method':cfg['method'],'encoder':cfg['backbone'],'shots':cfg['shots'],'fold':cfg['_fold_index'],
              'classes':[k for k,v in sorted(cfg['label_dict'].items(),key=lambda x:x[1])],'full_test_patients':True}
        all_ranks.extend(rank_fold(meta,rows,np.zeros((len(rows),len(units),2)),units))
        record={**meta,'units':units,'all_feature_and_group_scores_pp':0.,'status':'completed',
                'score_version':VERSION,'derivation':'structural: saved tme_mode=zero discards the standardized numeric TME vector before the conditioner',
                'prediction_replay_performed':False,'not_a_measured_forward_difference':True,
                'expected_slides':len(rows),'expected_patients':len({r['case_id'] for r in rows}),
                'provenance':b['provenance']}
        dump(out/task['name']/'attribution.json',record);records.append({k:record[k] for k in ['cohort','method','encoder','fold','expected_slides','expected_patients']})
    write_csv(out/'fold_feature_scores.csv',all_ranks);write_csv(out/'feature_rankings.csv',aggregate_ranks(all_ranks))
    dump(out/'summary.json',{'status':'completed','conditions':len(records),'coverage':records,'derivation':'structural zero, not empirical forward replay',
         'source_sha256':{str(f):sha(f) for f in Path(__file__).parent.glob('*.py')},'recorded_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'zero_adapter_folds':len(records),'feature_group_rank_rows':len(all_ranks),'training':False}),flush=True)

if __name__=='__main__':main()
