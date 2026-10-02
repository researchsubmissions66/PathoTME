#!/usr/bin/env python3
"""Four-cohort post-hoc adapter feature attribution using exact frozen checkpoints."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PGVL=ROOT.parent/'PGVL-Gym'
sys.path[:0]=[str(ROOT),str(PGVL)]
from score_core import VERSION,sha,dump,write_csv,make_units,rank_fold,aggregate_ranks,patient_scores
from pathotme.attribution_io import load_bound_run
from pathotme.shared_panel import common_spec
from pathotme.brca_features import panel_spec


def layout(cfg):
    spec=panel_spec('brca_morph64_v1') if cfg['task']=='brca' and cfg.get('tme_panel')!='shared_core62_v1' else common_spec()
    names=list(spec['feature_names'])
    groups=[{'name':scale+'/'+name,'indices':[names.index(c) for c in columns]}
            for scale,key in [('low','low_tokens'),('high','high_tokens')]
            for name,columns in spec[key].items()]
    return names,groups


def choose_rows(rows,patients_file=None,max_patients=None):
    cases={r['case_id'] for r in rows}
    if patients_file:
        chosen=json.loads(Path(patients_file).read_text())
        if not isinstance(chosen,list) or len(set(chosen))!=len(chosen) or not set(chosen)<=cases:raise ValueError('Invalid patient selection')
        chosen=set(chosen)
    elif max_patients:
        chosen=set(sorted(cases,key=lambda c:hashlib.sha256(('pathotme-feature-attribution-v1:'+c).encode()).hexdigest())[:max_patients])
    else:chosen=cases
    selected=[r for r in rows if r['case_id'] in chosen]
    if not selected:raise ValueError('No selected held-out patients')
    return selected


def build_saved(bound,arm,device):
    import torch
    from train import build_loaders,set_seed
    from pathotme.attribution_runtime import make_bridge
    cfg=dict(bound['config']);set_seed(cfg['seed']+cfg['_fold_index'])
    maps=json.loads(Path(bound['completion']['plan']['donor_maps']).read_text())
    if cfg['task'] in ['crc','blca']:
        # Load the exact source-bound cohort constructor without changing globals.
        source=ROOT/'campaigns/crc_blca_core62_20260914/runtime_models.py'
        spec=importlib.util.spec_from_file_location('pathotme_attribution_cohort_runtime',source)
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        bridge=runtime.adapter_type(cfg['method'],cfg['backbone'])(cfg,maps,arm,device)
    else:bridge=make_bridge(cfg,maps,arm,device)
    model=bridge.build_model()
    saved=torch.load(bound['directory']/'best.pt',map_location=device,weights_only=True)
    if saved['identity']!=bound['record']['identity']:raise ValueError('Adapter checkpoint identity mismatch')
    model.load_adapter_state_dict(saved['state']);model.eval()
    if not bool(model.standardizer.fitted.item()):raise ValueError('Missing fitted standardizer')
    # Preserve exact constructor config; only data-loader concurrency changes.
    loader=build_loaders(cfg['method'],{**cfg,'num_workers':0,'pin_memory':False},cfg['_fold_index'])[2]
    return bridge,model,loader


def selected_features(bound,row):
    if bound['config']['task'] not in ['crc','blca']:
        from pathotme.attribution_runtime import check_selected_features
        return check_selected_features(bound,row)
    inventory=json.loads(Path(bound['completion']['plan']['feature_inventory']).read_text())
    files=inventory['slides'][row['slide_id']]['files'];result=[]
    for key in ['feature_path_column','feature_path_column_s','feature_path_column_l']:
        column=bound['config'].get(key)
        if not column:continue
        path=Path(os.path.expandvars(row[column])).resolve();record=files[str(path)];stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(record['size'],record['mtime_ns']):raise ValueError('Changed selected feature file')
        result.append({'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    if not result:raise ValueError('No audited consumed features')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion',type=Path,required=True);p.add_argument('--launch',type=Path)
    p.add_argument('--arm',choices=['actual','zero','shuffled'],default='actual')
    choose=p.add_mutually_exclusive_group(required=True)
    choose.add_argument('--all-test-patients',action='store_true');choose.add_argument('--max-patients',type=int);choose.add_argument('--patients-file',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.max_patients is not None and a.max_patients<1:raise ValueError('Positive max-patients required')
    out=a.output.expanduser().resolve()
    if any(out==r or r in out.parents for r in [ROOT,PGVL]):raise ValueError('Use private attribution storage')
    bound=load_bound_run(a.completion,a.arm,a.launch)
    cfg=bound['config'];names,groups=layout(cfg);units=make_units(names,groups)
    rows=choose_rows(bound['test_rows'],a.patients_file,a.max_patients)
    naming=json.loads((ROOT/'docs/data/variant-names.json').read_text())
    meta={'variant':naming['variants'][a.arm]['display_name'],'cohort':cfg['task'],
          'panel':'brca_morph64_v1' if cfg['task']=='brca' and len(names)==64 else 'shared_core62_v1',
          'method':cfg['method'],'encoder':cfg['backbone'],'fold':cfg['_fold_index'],'shots':cfg['shots'],
          'classes':[k for k,v in sorted(cfg['label_dict'].items(),key=lambda x:x[1])]}
    plan={**meta,'arm':a.arm,'patients':len({r['case_id'] for r in rows}),'slides':len(rows),
          'expected_fold_patients':len({r['case_id'] for r in bound['test_rows']}),'expected_fold_slides':len(bound['test_rows']),
          'features':len(names),'groups':len(groups),'estimated_forward_calls':len(rows)*(len(units)+4),
          'mode':'execute' if a.execute else 'plan','output':str(out),'training':False,
          'selection':'all held-out patients' if a.all_test_patients else 'explicit patient list' if a.patients_file else 'fixed hash order independent of label, prediction and feature values',
          'shuffled_interpretation':'donor TME row consumed by the saved checkpoint' if a.arm=='shuffled' else None}
    print(json.dumps(plan,indent=2),flush=True)
    if not a.execute:return
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'plan.json',plan)
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:os.environ[key]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from common.configuration import load_dotenv
    load_dotenv(PGVL/'.env')
    import numpy as np
    import torch
    torch.set_num_threads(a.threads);start=time.monotonic()
    # Verify every source binding before reconstruction; weight checks are explicit.
    launch= json.loads(Path(a.launch).read_text()) if a.launch else None
    if launch:
        from pathotme.locked_tcga import verify_files
        verify_files(launch['file_sha256'])
    bound=load_bound_run(a.completion,a.arm,a.launch,weights=True)
    bridge,model,loader=build_saved(bound,a.arm,a.device)
    from pathotme.attribution_runtime import one_batch
    from pathotme.attribution import ablate_tme
    all_scores=[];probabilities=[];feature_records=[];replay_errors=[]
    for i,row in enumerate(rows):
        feature_records.append(selected_features(bound,row));batch=one_batch(loader,row['slide_id'])
        def predict():
            detail=bridge.eval_step_with_details(batch,model)
            if detail['metadata']['slide_id']!=row['slide_id']:raise ValueError('Wrong slide in prediction')
            return detail['probabilities']
        with torch.no_grad():
            observed=predict();expected=observed.new_tensor([[float(bound['predictions'][row['slide_id']][f'probability_{j}']) for j in range(2)]])
            torch.testing.assert_close(observed,expected,rtol=1e-4,atol=1e-4)
            replay_errors.append(float((observed-expected).abs().max()))
        score=ablate_tme(model,predict,names,groups,granularity='both')
        indexed={(r['level'],r['name']):r for r in score['scores']}
        all_scores.append([indexed[(u['level'],u['name'])]['score_pp'] for u in units]);probabilities.append(score['probabilities'])
        print(json.dumps({'slide':i+1,'of':len(rows),'elapsed_seconds':time.monotonic()-start}),flush=True)
    numeric_rows=[{'slide_id':r['slide_id'],'case_id':r['case_id'],'label':int(cfg['label_dict'].get(r['label'],r['label']))} for r in rows]
    values=np.asarray(all_scores);cases,labels,patient_values=patient_scores(numeric_rows,values)
    np.savez_compressed(out/'scores.npz',score_pp=values,patient_score_pp=patient_values,
                        slide_id=np.asarray([r['slide_id'] for r in rows]),case_id=np.asarray([r['case_id'] for r in rows]),
                        label=np.asarray([r['label'] for r in numeric_rows]),patient_id=np.asarray(cases),patient_label=labels,probabilities=np.asarray(probabilities))
    ranks=rank_fold(meta,numeric_rows,values,units);write_csv(out/'fold_feature_scores.csv',ranks)
    aggregate=aggregate_ranks(ranks)
    for r in aggregate:r['complete_five_fold']=False
    write_csv(out/'feature_rankings.csv',aggregate)
    dump(out/'attribution.json',{**plan,'status':'completed','score_version':VERSION,'units':units,
         'rows':numeric_rows,'full_test_patients':len(rows)==len(bound['test_rows']),
         'provenance':bound['provenance'],'feature_inventory':feature_records,
         'source_sha256':{str(f):sha(f) for f in Path(__file__).parent.glob('*.py')},
         'scores_sha256':sha(out/'scores.npz'),'max_saved_prediction_error':max(replay_errors),
         'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start})
    print(json.dumps({'status':'completed','output':str(out),'patients':len(cases),'elapsed_seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
