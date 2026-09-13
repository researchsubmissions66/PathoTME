#!/usr/bin/env python3
"""Focused saved-config reload and real optimizer construction regression."""
import json
from pathlib import Path
import sys
sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym','/path/to/PathoTME/scripts']
from pathotme.locked_tcga import atomic_json,load_launch,sha
from retry_tcga_numeric_configs import OUTPUT

if __name__=='__main__':
    campaign=json.loads((OUTPUT/'campaign.json').read_text())
    launches=[load_launch(g['launch']) for g in campaign['groups']]
    print('Checking exact saved YAML through train.load_yaml_config and real method optimizers',flush=True)
    import torch
    import train
    from methods import get_method
    from common.preflight import preflight
    from pathotme.cross_encoder_models import CLIPMGPathMethod
    checked=[];optimizers=[]
    canonical=lambda x:json.dumps(x,sort_keys=True,allow_nan=False)
    for g,launch in zip(campaign['groups'],launches):
        for p in launch['plans']:
            old=json.loads(Path(p['retry_of_config']).read_text())
            cfg=train.load_yaml_config(p['config'])
            expected={**old,'results_dir':p['native_dir']}
            assert canonical(cfg)==canonical(expected),p['config']
            assert type(cfg['lr']) is float and type(cfg['weight_decay']) is float
            # The old exported JSON must reproduce the diagnosed parsing error.
            broken=train.load_yaml_config(p['retry_of_config'])
            assert isinstance(broken['weight_decay'],str)
            if p['method']=='mgpath':assert isinstance(broken['lr'],str)
            checked.append({'path':p['config'],'sha256':sha(p['config']),
                            'lr':cfg['lr'],'weight_decay':cfg['weight_decay'],
                            'exact_reload_and_old_failure_reproduction':True})
            if p['fold']!=0:continue
            is_clip=g['tag']=='cross' and p['method']=='mgpath'
            if not is_clip:
                report=preflight(cfg)
                assert report.ok,report.as_dict()
            cls=CLIPMGPathMethod if is_clip else get_method(p['method'])
            method=cls(cfg,device='cpu')
            probe=torch.nn.Linear(2,2)
            optimizer=method.build_optimizer(probe)
            assert optimizer.param_groups[0]['lr']==old['lr']
            assert optimizer.param_groups[0]['weight_decay']==old['weight_decay']
            loss=probe(torch.ones(1,2)).square().sum();loss.backward();optimizer.step()
            assert torch.isfinite(probe.weight).all()
            optimizers.append({'group':[g['tag'],p['cohort'],p['method']],
                'method_class':cls.__name__,'real_optimizer_construction_and_step':'passed',
                'model_for_optimizer_check':'tiny Linear probe; actual models checked by dependent GPU smokes'})
            print(json.dumps(optimizers[-1]),flush=True)
    assert len(checked)==25 and len(optimizers)==5
    record={'status':'passed','launch_identities':[x['identity'] for x in launches],
            'saved_runtime_configs':checked,'optimizer_checks':optimizers,
            'scope':'One focused regression: all 25 exact runtime files through the training CLI loader; real method optimizer construction for five pairings. No broad suite. Model integration remains GPU-smoke gated.',
            'checker_sha256':sha(__file__)}
    atomic_json(OUTPUT/'validation.json',record)
    print(json.dumps({'status':'passed','configs':25,'optimizers':5}),flush=True)
