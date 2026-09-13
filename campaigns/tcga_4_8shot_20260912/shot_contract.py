"""Frozen 4/8-shot follow-up to the completed two-model PathoTME matrix."""
from collections import Counter
from pathlib import Path
import csv
import json
import sys
import numpy as np

sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym']
from pathotme.locked_tcga import rows,membership,identity,verify_files

SHOTS=[4,8]
COHORTS=['nsclc','brca']
METHODS=['vila_mil','mgpath']
ENCODERS=['clip-rn50','plip']


def tag_for(method,encoder):
    return 'native' if (method,encoder) in [('vila_mil','clip-rn50'),('mgpath','plip')] else 'cross'


def phase_subset(manifest,parent_phases,cohort,fold,shot,seed=1):
    """Replay the original max-shot=64 permutation against frozen outer folds."""
    if shot not in SHOTS:raise ValueError('only 4 and 8 additional shots authorized')
    labels=['LUAD','LUSC'] if cohort=='nsclc' else ['IDC','ILC']
    test_patients={r['case_id'] for r in parent_phases['test']}
    patients={}
    representatives={}
    for row in manifest:
        patient,label=row['case_id'],row['label']
        if patient in patients and patients[patient]!=label:raise ValueError('inconsistent patient labels')
        patients[patient]=label
        key=(patient,label)
        if key not in representatives or row['slide_id']<representatives[key]['slide_id']:
            representatives[key]=row
    wanted={'train':set(),'val':set()}
    for index,label in enumerate(labels):
        cases=sorted(p for p,l in patients.items() if l==label and p not in test_patients)
        if len(cases)<128:raise ValueError('original max-shot sampling population changed')
        rng=np.random.default_rng(seed+fold*1009+index*100003)
        ordered=np.asarray(cases,dtype=object)[rng.permutation(len(cases))].tolist()
        for phase,offset in [('train',0),('val',64)]:
            parent=[r for r in parent_phases[phase] if r['label']==label]
            expected=[representatives[p,label]['slide_id'] for p in sorted(ordered[offset:offset+16])]
            if [r['slide_id'] for r in parent]!=expected:
                raise ValueError('original ordered 16-shot split did not reproduce')
            wanted[phase].update(ordered[offset:offset+shot])
    result={phase:[{**r,'shots':str(shot)} for r in values
                   if phase=='test' or r['case_id'] in wanted[phase]]
            for phase,values in parent_phases.items()}
    check_phases(result,shot)
    return result


def check_phases(phases,shot):
    patients={}
    for phase,values in phases.items():
        if len({r['slide_id'] for r in values})!=len(values):raise ValueError('duplicate slide')
        patients[phase]={r['case_id'] for r in values}
        if phase!='test':
            if Counter(int(r['label_id']) for r in values)!={0:shot,1:shot}:
                raise ValueError('wrong number of shots per class')
            if len(patients[phase])!=2*shot:raise ValueError('training/validation need one slide per patient')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if patients[a]&patients[b]:raise ValueError('patient leakage')


def write_rows(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(values[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(values)


def load_campaign(path):
    campaign=json.loads(Path(path).read_text());expected=campaign.pop('identity')
    if identity(campaign)!=expected:raise ValueError('campaign identity changed')
    campaign['identity']=expected
    if campaign['shots']!=SHOTS or campaign['counts']['fold_jobs']!=80 or campaign['counts']['smoke_jobs']!=8:
        raise ValueError('campaign exceeds authorized 4/8-shot scope')
    verify_files(campaign['file_sha256'])
    return campaign


def validate_native(plan,cfg):
    from common.configuration import load_yaml_config
    from common.run_state import validate_resume_state
    from pathotme.locked_tcga import sha
    native=Path(plan['native_dir']);fold=plan['fold']
    expected=load_yaml_config(plan['source_base_config']) if plan['native_reused'] else cfg
    valid=validate_resume_state(json.loads((native/'metrics.json').read_text()),native/'config.json',plan['method'],expected)
    record=next((r for r in valid if r['fold']==fold),None)
    if record is None or any(record.get('sample_failures',{}).values()):
        raise ValueError('native fold is incomplete or has sample failures')
    prediction=native/f'fold{fold}_predictions.csv'
    if sorted(membership(rows(prediction)))!=sorted(membership(rows(Path(cfg['split_dir'])/f'fold{fold}/test.csv'))):
        raise ValueError('native test membership changed')
    return {str(p):sha(p) for p in [native/'config.json',native/'metrics.json',native/f'fold{fold}_best.pt',prediction]}
