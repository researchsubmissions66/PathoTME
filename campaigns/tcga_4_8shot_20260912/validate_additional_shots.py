"""One focused check of exact shot configs, donor splits and training statistics."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym']
from shot_contract import load_campaign,check_phases,SHOTS
from pathotme.locked_tcga import atomic_json,identity,sha,rows,membership,verify_files
from common.configuration import load_dotenv,load_yaml_config
from launch_additional_shots import commands

OUTPUT=Path('/path/to/shared/PathoTME-results/tcga_4_8shot_20260912_v1')
load_dotenv('/path/to/PGVL-Gym/.env')
campaign=load_campaign(OUTPUT/'prepared_campaign.json')
launches=[json.loads(Path(g['launch']).read_text()) for g in campaign['groups']]
bound={}
for launch in launches:bound.update(launch['file_sha256'])
verify_files(bound)
print('Loading focused numeric/statistics validation dependencies',flush=True)
import numpy as np
import torch
from methods import get_method
from pathotme.cross_encoder_models import CLIPMGPathMethod
from pathotme.guided_vila import FoldStandardizer
from pathotme.muse_data import load_tme
from pathotme.focus_contract import validate_donors

torch.set_num_threads(2)
tables={};stats={};config_checks=[];optimizer_checks=[]
seen_optimizers=set()
for launch in launches:
    for plan in launch['plans']:
        cfg=load_yaml_config(plan['config']);shot=plan['shots'];fold=plan['fold'];cohort=plan['cohort']
        assert cfg['shots']==shot in SHOTS and cfg['k_start']==fold and cfg['k_end']==fold+1
        assert type(cfg['lr']) is float and type(cfg['weight_decay']) is float
        phases={phase:rows(Path(cfg['split_dir'])/f'fold{fold}/{phase}.csv') for phase in ['train','val','test']}
        check_phases(phases,shot)
        maps=json.loads(Path(plan['donor_maps']).read_text());validate_donors(phases,maps)
        parent=json.loads(Path(plan['parent_completed_fold']).read_text())
        old_path=Path(parent['plan']['config'])
        old_cfg=json.loads(old_path.read_text()) if old_path.suffix=='.json' else load_yaml_config(old_path)
        for phase,values in phases.items():
            original=rows(Path(old_cfg['split_dir'])/f'fold{fold}/{phase}.csv')
            if phase=='test':assert membership(values)==membership(original)
            else:assert set(membership(values))<=set(membership(original))
        ignored={'shots','split_dir','results_dir','benchmark','experiment','k_start','k_end'}
        assert {k:v for k,v in cfg.items() if k not in ignored}=={k:v for k,v in old_cfg.items() if k not in ignored}
        key=(cohort,shot,fold)
        if key not in stats:
            table=tables.setdefault(cohort,load_tme(plan['tme_csv'],cohort)) if cohort not in tables else tables[cohort]
            values=table.loc[[r['slide_id'] for r in phases['train']]].to_numpy(dtype=np.float32)
            model=FoldStandardizer(values.shape[1]);model.fit(values)
            standardized=model(torch.tensor(values))
            assert torch.isfinite(standardized).all()
            shuffled=table.loc[[maps['train'][r['slide_id']] for r in phases['train']]].to_numpy(dtype=np.float32)
            other=FoldStandardizer(values.shape[1]);other.fit(shuffled)
            for name in ['median','mean','scale']:
                torch.testing.assert_close(getattr(model,name),getattr(other,name),rtol=0,atol=0)
            stats[key]={'cohort':cohort,'shots':shot,'fold':fold,'training_rows':len(values),'features':values.shape[1],
                'actual_and_shuffled_stats_match':True,'finite_standardized_training_values':True,
                'standardizer_state_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()}
        opt_key=(plan['method'],plan['encoder'])
        if opt_key not in seen_optimizers:
            cls=CLIPMGPathMethod if opt_key==('mgpath','clip-rn50') else get_method(plan['method'])
            bridge=cls(cfg,device='cpu');probe=torch.nn.Linear(2,2)
            optimizer=bridge.build_optimizer(probe)
            probe(torch.ones(1,2)).square().sum().backward();optimizer.step()
            assert torch.isfinite(probe.weight).all()
            assert optimizer.param_groups[0]['lr']==cfg['lr'] and optimizer.param_groups[0]['weight_decay']==cfg['weight_decay']
            seen_optimizers.add(opt_key)
            optimizer_checks.append({'method':plan['method'],'encoder':plan['encoder'],'class':cls.__name__,
                                      'numeric_optimizer_construction_and_step':True})
        config_checks.append({'path':plan['config'],'sha256':sha(plan['config']),'native_reused':plan['native_reused']})
assert len(config_checks)==80 and len(stats)==20 and len(optimizer_checks)==4
for cohort in ['nsclc','brca']:
    for fold in range(5):
        small=OUTPUT/f'splits/{cohort}/4shot/fold{fold}'
        large=OUTPUT/f'splits/{cohort}/8shot/fold{fold}'
        for phase in ['train','val']:
            assert set(membership(rows(small/f'{phase}.csv')))<set(membership(rows(large/f'{phase}.csv')))
jobs=commands(campaign,OUTPUT/'campaign.json')
assert len(jobs)==88 and sum(p['smoke'] for p in jobs)==8
assert all(p['shots']==[4,8] for p in jobs if p['smoke'])
assert all(p['shots'] in SHOTS for p in jobs if not p['smoke'])
validation={'status':'passed','prepared_campaign_identity':campaign['identity'],
    'child_launch_identities':[l['identity'] for l in launches],
    'runtime_configs_checked':config_checks,'training_normalizers':list(stats.values()),'optimizer_checks':optimizer_checks,
    'bound_files_verified':len(bound),'jobs_planned':88,
    'source_sha256':{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
    'scope':'Exact 80 runtime configs and 20 training-only normalizers, four real method optimizer constructions with tiny tensors. No new model/encoder tests or broad test suite; eight live smokes gate both shots.',
    'live_gpu_smokes':False}
atomic_json(OUTPUT/'validation.json',validation)
campaign.pop('identity')
campaign['prepared_campaign_identity']=validation['prepared_campaign_identity']
campaign['validation']=str(OUTPUT/'validation.json')
campaign['file_sha256'].update({str(OUTPUT/'validation.json'):sha(OUTPUT/'validation.json'),
                               str(OUTPUT/'prepared_campaign.json'):sha(OUTPUT/'prepared_campaign.json')})
campaign['identity']=identity(campaign)
atomic_json(OUTPUT/'campaign.json',campaign)
record={'status':'validated_not_submitted','campaign':campaign,'validation':validation,
        'preparation':json.loads((OUTPUT/'preparation.json').read_text())}
atomic_json('/path/to/PGVL-Gym/benchmarks/pathotme_4_8shot_prepare_20260912.json',record)
print(json.dumps({'status':'passed','campaign_identity':campaign['identity'],'counts':campaign['counts'],
                  'runtime_configs':80,'training_normalizers':20,'optimizer_checks':4}),flush=True)
