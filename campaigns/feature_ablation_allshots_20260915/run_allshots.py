#!/usr/bin/env python3
"""Serial two-thread CPU feature ablations at every feasible canonical shot."""
import argparse
import fcntl
import os
from pathlib import Path
import time
import json
from allshot_design import base,np,load,sha,identity,verify_files,atomic_json,POLICY,check_identity,prepare,result_root


def run_task(manifest,task):
    folder=result_root(manifest,task)
    if task['shots']==16:
        return {'status':'reused_completed_16shot','conditions':42,'source':str(folder)}
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'results.json').exists():
        old=load(folder/'results.json')
        if old['manifest_identity']!=manifest['identity'] or old['task_id']!=task['id'] or old['status']!='completed':
            raise ValueError('Existing fold identity/status mismatch')
        verify_files(old['artifact_sha256'])
        return {'status':'completed','conditions':len(old['conditions']),'resumed':True}
    started=time.monotonic();verify_files(task['input_sha256'])
    frames=base.read_splits(task)
    if any(frames[p].to_dict('records')!=task['memberships'][p] for p in frames):raise ValueError('Split membership mismatch')
    names=task['feature_names'];scaler=base.fit_scaler(base.feature_matrix(task,frames['train']))
    train=base.scale_values(base.feature_matrix(task,frames['train']),scaler)
    learned,elimination=base.training_subsets(train,frames['train'].label.to_numpy(),names,task['seed'])
    conditions=base.fixed_conditions(task)+learned
    if len(conditions)!=42:raise ValueError('Unexpected feature condition count')
    design={'manifest_identity':manifest['identity'],'task_id':task['id'],'cohort':task['cohort'],
            'shots':task['shots'],'fold':task['fold'],'panel':task['panel'],'scaler':scaler,
            'conditions':conditions,'elimination':elimination,
            'selector_policy':POLICY['feature_ablation_policy']}
    design['identity']=identity(design)
    atomic_json(folder/'design.json',design)
    # The complete feature design is frozen before validation features are used.
    val=base.scale_values(base.feature_matrix(task,frames['val']),scaler)
    fitted={}; selections=[]
    for cond in conditions:
        idx=[names.index(n) for n in cond['feature_names']]
        candidates=[];models={}
        for c in base.POLICY['c_grid']:
            model=base.fit(train[:,idx],frames['train'].label.to_numpy(),c,task['seed'])
            score=base.metrics(frames['val'],model.predict_proba(val[:,idx])[:,1])
            candidates.append({'C':c,'metrics':score});models[c]=model
        chosen=min(candidates,key=lambda r:base.selection_key(r['metrics'],r['C']))
        model=models[chosen['C']];fitted[cond['name']]=(idx,model)
        selections.append({'condition':cond,'selected_C':chosen['C'],'candidates':candidates,
                           'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),
                           'indices':idx,'classes':[0,1]})
    # All 42 C choices and fitted coefficients are frozen before any test feature
    # matrix, prediction or metric is computed. Compact storage avoids thousands
    # of tiny files without altering the scientific fitting/selection functions.
    selection={'manifest_identity':manifest['identity'],'design_identity':design['identity'],
               'task_id':task['id'],'models':selections}
    atomic_json(folder/'selection.json',selection)
    test=base.scale_values(base.feature_matrix(task,frames['test']),scaler)
    matrix={'condition_name':np.asarray([c['name'] for c in conditions])};results=[]
    phase_probabilities={}
    for phase,values in [('val',val),('test',test)]:
        probabilities=[]
        for cond in conditions:
            idx,model=fitted[cond['name']]
            probabilities.append(model.predict_proba(values[:,idx])[:,1])
        phase_probabilities[phase]=np.stack(probabilities)
        matrix[phase+'_probability_1']=phase_probabilities[phase]
        for column in ['slide_id','case_id','label']:
            matrix[phase+'_'+column]=frames[phase][column].to_numpy(dtype=int if column=='label' else str)
    for j,(cond,sel) in enumerate(zip(conditions,selections)):
        results.append({'condition':cond,'selected_C':sel['selected_C'],
                       'metrics':{phase:base.metrics(frames[phase],phase_probabilities[phase][j]) for phase in ['val','test']}})
    temporary=folder/'predictions.tmp.npz';np.savez_compressed(temporary,**matrix);temporary.replace(folder/'predictions.npz')
    record={'status':'completed','manifest_identity':manifest['identity'],'task_id':task['id'],
            'variant':'PathoTME-LR','cohort':task['cohort'],'panel':task['panel'],'shots':task['shots'],'fold':task['fold'],
            'conditions':results,'wall_seconds':time.monotonic()-started,
            'prediction_layout':'probability_1[condition,row]; condition_name and phase-specific IDs/labels define exact axes',
            'artifact_sha256':{str(folder/n):sha(folder/n) for n in ['design.json','selection.json','predictions.npz']}}
    atomic_json(folder/'results.json',record)
    return {'status':'completed','conditions':len(results),'wall_seconds':record['wall_seconds']}


def execute(m):
    root=Path(m['output'])
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_files(m['source_sha256']);verify_files(m['reused_result_sha256']);verify_files(m['historical_split_sha256'])
        if sha(m['parent_manifest'])!=m['parent_sha256']:raise ValueError('Parent manifest changed')
        p=root/'status.json'
        status=load(p) if p.exists() else {'identity':m['identity'],'tasks':{},'started_at':time.time()}
        if status['identity']!=m['identity']:raise ValueError('Wrong controller identity')
        status.update(pid=os.getpid(),node=os.uname().nodename)
        for task in sorted(m['tasks'],key=lambda t:(t['shots'],t['cohort'],t['fold'])):
            key=f"{task['cohort']}_{task['shots']}shot_f{task['fold']}"
            status['tasks'][key]={'status':'running','started_at':time.time()};atomic_json(p,status)
            try:outcome=run_task(m,task)
            except Exception as error:
                status['tasks'][key]={'status':'failed','error':str(error)};atomic_json(p,status);raise
            status['tasks'][key]=outcome;atomic_json(p,status)
            print(json.dumps({'task':key,**outcome}),flush=True)
        verify_files(m['source_sha256']);verify_files(m['reused_result_sha256'])
        status['finished_at']=time.time();status['wall_seconds']=status['finished_at']-status['started_at'];atomic_json(p,status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--parent-manifest',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.prepare:
        if not a.parent_manifest or not a.output:p.error('--prepare requires --parent-manifest and --output')
        prepare(a.parent_manifest,a.output)
    else:
        if not a.manifest:p.error('--manifest is required')
        m=load(a.manifest);check_identity(m)
        if m['policy']!=POLICY:raise ValueError('Experiment policy changed')
        if a.execute:execute(m)
        else:print(json.dumps({'counts':m['counts'],'support':m['support']},indent=2))
