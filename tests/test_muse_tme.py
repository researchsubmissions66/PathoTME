"""Focused numerical checks for MUSE's TME boundary, controls and fixed scope."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym', '/path/to/PathoTME/scripts']
from methods.muse.model import MUSEModel
from pathotme.muse_tme import TMEGuidedMUSE, MUSETMEMethod
from pathotme.muse_contract import build_config, validate_config
from pathotme.focus_contract import validate_donors
from pathotme.locked_tcga import ROOT, PGVL
from pathotme.yaml_runtime_config import write_runtime_config
from common.configuration import load_dotenv, load_yaml_config
from launch_muse_tcga import commands

torch.set_num_threads(1)


def wrapper(cohort, width):
    torch.manual_seed(41)
    input_dim = 768 if width == 512 else width
    base = MUSEModel(input_dim, 2, torch.randn(2, 20, width), torch.randn(2, width),
                     embed_dim=width, num_heads=8, dropout=0.25)
    model = TMEGuidedMUSE(base, cohort, hidden_dim=16, num_heads=4, dropout=0.0)
    values = torch.randn(32, model.standardizer.feature_count)
    model.standardizer.fit(values)
    return model, torch.randn(31, input_dim), torch.tensor([1]), values[:1]


@pytest.mark.parametrize('cohort,width', [('nsclc',512),('brca',512),('nsclc',1024),('brca',1024)])
def test_native_parity_real_gradient_and_frozen_parameters(cohort, width):
    model, features, label, values = wrapper(cohort, width)
    model.eval()
    initial = {k: v.clone() for k, v in model.base.state_dict().items()}
    with torch.no_grad():
        projection = model.conditioner.output_projection.weight.clone()
        model.conditioner.output_projection.weight.zero_()
        native = model.base(features, return_details=True)
        observed = model(features, label, values, return_details=True)
        for key in ('logits','patch_attention'):
            torch.testing.assert_close(observed[key], native[key])
        model.conditioner.output_projection.weight.copy_(projection)
        assert not torch.allclose(model(features,label,values)[0],model(features,label,values+10)[0],atol=1e-7,rtol=1e-7)
    model.train()
    assert not model.base.training and not model.base.moe.training
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    tme = values.clone().requires_grad_()
    model(features,label,tme)[2].backward()
    assert tme.grad is not None and tme.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    assert all(n.startswith('conditioner.') for n,p in model.named_parameters() if p.requires_grad)
    assert model.conditioner.query_projection.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    for key,value in model.base.state_dict().items():
        torch.testing.assert_close(value,initial[key],rtol=0,atol=0)
    model.eval()
    details = model(features,label,values,return_details=True)
    torch.testing.assert_close(details['patch_attention'].sum(-1),torch.ones(2))
    torch.testing.assert_close(details['tme_group_attention'].sum(-1),torch.ones(1,2))
    torch.testing.assert_close(details['expert_weights'].sum(-1),torch.ones(2))
    assert details['expert_weights'].gt(0).sum(-1).eq(2).all()
    assert details['tme_group_attention'].shape == (1,2,16)


def test_zero_control_checkpoint_and_label_free_prediction():
    model,features,label,values = wrapper('brca',32)
    model.tme_mode='zero'; model.eval()
    expected=model(features,label,values)[0]
    torch.testing.assert_close(expected,model(features,label,values+1000)[0])
    torch.testing.assert_close(expected,model(features,1-label,values)[0],rtol=0,atol=0)
    # Neither the retrieval function nor the native training-only bank may
    # participate in the adapter's training or evaluation predictions.
    def forbidden(*args,**kwargs):
        raise AssertionError('training-only semantic retrieval was called')
    model.base.retrieve_semantic_views=forbidden
    with torch.no_grad(): model.base.prompt_bank.fill_(1e5)
    model.train()
    torch.testing.assert_close(expected,model(features,label,values)[0],rtol=0,atol=0)
    saved={k:v.clone() for k,v in model.adapter_state_dict().items()}
    with torch.no_grad(): model.conditioner.output_projection.weight.add_(1)
    model.load_adapter_state_dict(saved)
    torch.testing.assert_close(expected,model(features,label,values)[0])
    with pytest.raises(ValueError):
        model.load_adapter_state_dict({**saved,'base.classifier.weight':model.base.classifier.weight})


def test_training_only_statistics_and_fixed_donors():
    model,_,_,_ = wrapper('nsclc',32)
    method=object.__new__(MUSETMEMethod); method.cfg={'_fold_index':0}
    method.tme=pd.DataFrame([[1.]*62,[3.]*62,[1e6]*62],index=['a','b','heldout'])
    loader=SimpleNamespace(dataset=SimpleNamespace(df=pd.DataFrame({'slide_id':['a','b']})))
    method.prepare_fold(0,model,loader)
    torch.testing.assert_close(model.standardizer.mean,torch.full((62,),2.))
    assert model.standardizer.scale.eq(1).all()
    phases={'train':[{'slide_id':'a','case_id':'p1'},{'slide_id':'b','case_id':'p2'}]}
    validate_donors(phases,{'train':{'a':'b','b':'a'}})
    with pytest.raises(ValueError): validate_donors(phases,{'train':{'a':'a','b':'b'}})
    with pytest.raises(ValueError): validate_donors(phases,{'train':{'a':'heldout','b':'a'}})


def test_config_and_numeric_roundtrip(tmp_path):
    load_dotenv(PGVL/'.env')
    for cohort in ('nsclc','brca'):
        for encoder in ('plip','clip-rn50'):
            base=load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/configs/muse_paired_{encoder.replace("-","_")}/{cohort}_16shot.yaml')
            cfg=build_config(base,'test',cohort,encoder)
            assert cfg['prompt_csvs'] == base['prompt_csvs']
            path=tmp_path/f'{cohort}_{encoder}.yaml'
            write_runtime_config(path,cfg)
            validate_config(load_yaml_config(path))
            for update in ({'feature_resolutions':{'bag':'20x'}},{'muse_semantic_updates_per_slide':1},
                           {'muse_prompt_learning':'token_16_conch'},{'feature_space_id':'wrong'},
                           {'weight_decay':'1e-5'},{'max_patches':1000}):
                with pytest.raises((ValueError,TypeError)): validate_config({**deepcopy(cfg),**update})


def test_bounded_launch_and_encoder_dependencies():
    spec=json.loads((ROOT/'configs/muse_tcga_16shot_20260912.json').read_text())
    plans=[dict(cohort=c,encoder=e,method='muse',fold=f)
           for c in spec['cohorts'] for e in spec['encoders'] for f in spec['folds']]
    jobs=commands(dict(protocol=spec,output='/tmp/muse_test_plan',plans=plans),Path('/tmp/launch.json'))
    assert len(jobs)==24 and sum(p['smoke'] for p in jobs)==4
    smoke_pairs={p['pair'] for p in jobs if p['smoke']}
    assert len(smoke_pairs)==4
    assert all(p['depends_on_smoke_pair'] in smoke_pairs for p in jobs if not p['smoke'])
    assert all('run_muse_tcga.py' in p['command'][-1] for p in jobs)
