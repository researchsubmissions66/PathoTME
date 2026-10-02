"""Extend only the sample-count axis of the frozen within-panel LR study."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(os.environ.get('PATHOTME_ROOT',str(Path(__file__).resolve().parents[2])))
PGVL=Path(os.environ.get('PGVL_ROOT',str(ROOT.parent/'PGVL-Gym')))
OLD=ROOT/'campaigns/feature_ablation_20260915'
spec=importlib.util.spec_from_file_location('pathotme_ablation16',OLD/'run_ablation.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
np=base.np
load,sha,identity,verify_files,atomic_json=(base.load,base.sha,base.identity,base.verify_files,base.atomic_json)
REQUESTED=[4,8,16,32,64]
COHORTS=['nsclc','brca','crc','blca']
POLICY={'version':'within_panel_all_supported_shots_20260915_v1',
        'requested_shots':REQUESTED,'variant':'PathoTME-LR',
        'feature_ablation_policy':{k:v for k,v in base.POLICY.items() if k!='shots'},
        'shot_scope':'all five canonical sample counts where every frozen fold supports disjoint equal train/validation counts',
        'eligibility':'at least 2*shots development patients per class in every fold',
        'nsclc_brca_sampling':'replay original max64 permutation, train offset0 and validation offset64',
        'crc_blca_sampling':'replay original max16 permutation; preserve original phase anchors; new BLCA32 adds unused indices32:48 to train and48:64 to validation',
        'test_membership':'unchanged original outer folds; one representative slide per development patient',
        'reuse_16shot':'all 840 completed condition/fold results; no refitting',
        'neural_adapter_refits':0,'gpu_jobs':0,
        'inference':'within-panel exploratory LR results; no original-panel optimality or neural-adapter claim'}


def check_identity(record):
    if identity({k:v for k,v in record.items() if k!='identity'})!=record['identity']:
        raise ValueError('Manifest identity mismatch')


def universe(tasks):
    slides={}; owners={}; reps={}
    for task in tasks:
        for row in task['memberships']['test']:
            if row['slide_id'] in slides:raise ValueError('Overlapping outer test slides')
            slides[row['slide_id']]=row
            if owners.setdefault(row['case_id'],task['fold'])!=task['fold']:
                raise ValueError('Overlapping outer test patients')
    for sid,row in sorted(slides.items()):
        prev=reps.setdefault(row['case_id'],row)
        if prev['label']!=row['label']:raise ValueError('Conflicting patient labels')
    return slides,reps


def orders_for(task,reps):
    held={r['case_id'] for r in task['memberships']['test']}
    orders={}
    offset=64 if task['cohort'] in ['nsclc','brca'] else 16
    for label in [0,1]:
        eligible=sorted(c for c,r in reps.items() if r['label']==label and c not in held)
        rng=np.random.default_rng(1+task['fold']*1009+label*100003)
        order=np.asarray(eligible,dtype=object)[rng.permutation(len(eligible))].tolist()
        orders[label]=order
        for phase,start in [('train',0),('val',offset)]:
            actual=sorted(r['slide_id'] for r in task['memberships'][phase] if r['label']==label)
            expected=sorted(reps[c]['slide_id'] for c in order[start:start+16])
            if actual!=expected:raise ValueError('Original 16-shot sampling failed to replay')
    return orders


def phases_for(parent,reps,orders,shot):
    phases={'train':[],'val':[],'test':deepcopy(parent['memberships']['test'])}
    for label,order in orders.items():
        if len(order)<2*shot:raise ValueError('Insufficient disjoint development patients')
        if parent['cohort'] in ['nsclc','brca']:
            chosen={'train':order[:shot],'val':order[64:64+shot]}
        elif shot<=16:
            chosen={'train':order[:shot],'val':order[16:16+shot]}
        elif parent['cohort']=='blca' and shot==32:
            chosen={'train':order[:16]+order[32:48],'val':order[16:32]+order[48:64]}
        else:raise ValueError('Unsupported phase-preserving extension')
        for phase,cases in chosen.items():phases[phase].extend(deepcopy(reps[c]) for c in cases)
    phases={p:sorted(rows,key=lambda r:r['slide_id']) for p,rows in phases.items()}
    for phase,rows in phases.items():
        if len({r['slide_id'] for r in rows})!=len(rows):raise ValueError('Duplicate slide')
        if phase!='test' and (Counter(r['label'] for r in rows)!={0:shot,1:shot} or len({r['case_id'] for r in rows})!=2*shot):
            raise ValueError('Changed shot budget or repeated development patient')
    for p,q in [('train','val'),('train','test'),('val','test')]:
        if {r['case_id'] for r in phases[p]}&{r['case_id'] for r in phases[q]}:
            raise ValueError('Patient leakage')
    if phases['test']!=parent['memberships']['test']:raise ValueError('Test membership changed')
    if shot==16 and phases!=parent['memberships']:raise ValueError('16-shot membership changed')
    return phases


def result_root(manifest,task):
    if task['shots']==16:return Path(task['reuse_result_dir'])
    return Path(manifest['output'])/'folds'/task['cohort']/f"{task['shots']}shot"/f"fold{task['fold']}"


def prepare(parent_path,output):
    parent_path=Path(parent_path).resolve();old=load(parent_path);check_identity(old)
    verify_files(old['source_sha256'])
    baseline=load(old['parent_manifest']);check_identity(baseline)
    if sha(old['parent_manifest'])!=old['parent_sha256']:raise ValueError('Baseline manifest changed')
    output=base.private_output(output)
    if output.exists():raise FileExistsError('Refusing to overwrite an existing all-shot campaign')
    output.mkdir(parents=True)
    tasks=[];support={};reused={};audit=[];extra_bindings={}
    for cohort in COHORTS:
        parents=sorted([t for t in old['tasks'] if t['cohort']==cohort],key=lambda t:t['fold'])
        if [t['fold'] for t in parents]!=list(range(5)):raise ValueError('Missing parent fold')
        _,reps=universe(parents)
        orders={t['fold']:orders_for(t,reps) for t in parents}
        counts={str(f):{str(label):len(v) for label,v in o.items()} for f,o in orders.items()}
        eligible=[s for s in REQUESTED if all(len(v)>=2*s for o in orders.values() for v in o.values())]
        support[cohort]={'supported_shots':eligible,'development_patients_per_class_by_fold':counts,
            'excluded':{str(s):f'Need {2*s} disjoint development patients per class; smallest available class/fold has {min(len(v) for o in orders.values() for v in o.values())}.' for s in REQUESTED if s not in eligible}}
        for parent in parents:
            verify_files(parent['input_sha256'])
            nested={s:phases_for(parent,reps,orders[parent['fold']],s) for s in eligible}
            for small,large in zip(eligible,eligible[1:]):
                for phase in ['train','val']:
                    if not {r['case_id'] for r in nested[small][phase]}<{r['case_id'] for r in nested[large][phase]}:
                        raise ValueError('Shot memberships do not nest')
            for shot,phases in nested.items():
                task=deepcopy(parent);task['shots']=shot;task['parent_16shot_id']=parent['id']
                task['memberships']=phases
                task['id']=identity({'parent_task_id':parent['id'],'shots':shot,'memberships':phases,'policy':POLICY})
                if shot==16:
                    task['splits']=parent['splits'];task['input_sha256']=parent['input_sha256']
                    folder=Path(old['output'])/'folds'/f"{cohort}_f{parent['fold']}"
                    task['reuse_result_dir']=str(folder)
                    design=load(folder/'design.json');check_identity(design)
                    if design['task_id']!=parent['id'] or design['manifest_identity']!=old['identity']:
                        raise ValueError('Reused 16-shot design identity mismatch')
                    reused[str(folder/'design.json')]=sha(folder/'design.json')
                    for cond in design['conditions']:
                        p=folder/cond['name']/'metrics.json';r=load(p)
                        if r['status']!='completed' or r['manifest_identity']!=old['identity'] or r['design_identity']!=design['identity']:
                            raise ValueError('Incomplete or mismatched 16-shot result')
                        verify_files(r['artifact_sha256']);reused[str(p)]=sha(p)
                else:
                    where=output/'splits'/cohort/f'{shot}shot'/f"fold{parent['fold']}"
                    task['splits']={p:str(where/f'{p}.csv') for p in phases}
                    for phase,rows in phases.items():base.csv_write(task['splits'][phase],base.pd.DataFrame(rows))
                    task['input_sha256']={p:sha(p) for p in [task['tme_csv'],*task['splits'].values()]}
                    if cohort in ['nsclc','brca'] and shot in [4,8]:
                        original=Path(old['output']).parent/'tcga_4_8shot_20260912_v1'/'splits'/cohort/f'{shot}shot'/f"fold{parent['fold']}"
                        original_task={**task,'splits':{p:str(original/f'{p}.csv') for p in phases}}
                        historical=base.read_splits(original_task)
                        for phase,frame in historical.items():
                            if frame.to_dict('records')!=phases[phase]:raise ValueError('Existing 4/8-shot membership mismatch')
                            path=original_task['splits'][phase];extra_bindings[path]=sha(path)
                frames=base.read_splits(task)
                if any(frames[p].to_dict('records')!=phases[p] for p in phases):raise ValueError('Saved split membership mismatch')
                # Validate training-only imputation at every requested sample count.
                transformed=base.feature_matrix(task,frames['train']);base.fit_scaler(transformed)
                audit.append({'cohort':cohort,'shots':shot,'fold':parent['fold'],'panel':task['panel'],
                              'no_all_missing_training_column':True,'original_outer_test_unchanged':True,
                              'nested_development_patients':True,'original_16shot_replayed':True,
                              'existing_4_8shot_reproduced':cohort in ['nsclc','brca'] and shot in [4,8]})
                tasks.append(task)
    nnew=sum(t['shots']!=16 for t in tasks);nreuse=len(tasks)-nnew
    source={**old['source_sha256'],**{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}}
    manifest={'policy':POLICY,'parent_manifest':str(parent_path),'parent_sha256':sha(parent_path),
              'parent_identity':old['identity'],'output':str(output),'support':support,'tasks':tasks,
              'source_sha256':source,'reused_result_sha256':reused,'historical_split_sha256':extra_bindings,
              'counts':{'cohort_shot_settings':sum(len(x['supported_shots']) for x in support.values()),
                        'cohort_shot_folds':len(tasks),'new_cohort_shot_folds':nnew,'reused_16shot_folds':nreuse,
                        'new_condition_folds':42*nnew,'reused_condition_folds':42*nreuse,
                        'total_condition_folds':42*len(tasks),'new_C_candidate_fits':42*nnew*5,'gpu_jobs':0},
              'created_at_utc':datetime.now(timezone.utc).isoformat()}
    manifest['identity']=identity(manifest)
    atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'preflight.json',{'status':'passed','manifest_identity':manifest['identity'],
                'folds':audit,'source_binding_count':len(source),'historical_lowshot_split_files':len(extra_bindings)})
    print(json.dumps({'manifest':str(output/'manifest.json'),'identity':manifest['identity'],
                      'counts':manifest['counts'],'supported_shots':{c:s['supported_shots'] for c,s in support.items()}},indent=2),flush=True)
