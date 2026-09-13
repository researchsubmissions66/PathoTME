"""Lean research checks at the actual DyKo forward boundary."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
import pandas as pd
import pytest
import torch
sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym','/path/to/PathoTME/scripts']
from methods.dyko.model import DyKoModel
from common.models.paired_encoder_extension import FeatureSpacePromptLearner
from pathotme.dyko_tme import TMEGuidedDyKo,DyKoTMEMethod
from pathotme.dyko_smoke_checks import check_model,regression_checks
from pathotme.dyko_contract import build_config,validate_config
from pathotme.focus_contract import validate_donors
from pathotme.locked_tcga import ROOT,PGVL
from common.configuration import load_dotenv,load_yaml_config
from pathotme.yaml_runtime_config import write_runtime_config
from launch_dyko_tcga import commands

torch.set_num_threads(1)


def wrapper(cohort,width):
    torch.manual_seed(41)
    base=DyKoModel(FeatureSpacePromptLearner(torch.randn(2,width)),
        torch.randn(1000 if cohort=='nsclc' else 64,768),width=width,concept_input_dim=768,
        n_classes=2,visual_prototypes=10,concepts_per_prototype=10,num_heads=8)
    model=TMEGuidedDyKo(base,cohort,hidden_dim=16,num_heads=4,dropout=0.0)
    values=torch.randn(32,model.standardizer.feature_count)
    model.standardizer.fit(values)
    return model,torch.randn(15,width),torch.tensor([1]),values[:1]


@pytest.mark.parametrize('cohort,width',[('nsclc',512),('nsclc',1024),('brca',512),('brca',1024)])
def test_native_retrieval_parity_gradient_and_update(cohort,width):
    model,x,y,tme=wrapper(cohort,width);model.eval()
    frozen={k:v.clone() for k,v in model.base.state_dict().items()}
    check=check_model(model,(x,y,tme),'actual')
    assert check['tme_margin_gradient_l1']>0
    with torch.no_grad():
        native=model.base(x);guided=model(x,y,tme,return_details=True)
        for key in ['visual_cluster_prob','semantic_cluster_prob']:
            torch.testing.assert_close(native[key],guided[key],atol=0,rtol=0)
        projection=model.conditioner.output_projection.weight.clone()
        model.conditioner.output_projection.weight.zero_()
        parity=model(x,y,tme,return_details=True)
        torch.testing.assert_close(parity['visual_patch_attention'],native['visual_patch_attention'])
        model.conditioner.output_projection.weight.copy_(projection)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    model.train();assert not model.base.training
    model(x,y,tme)[2].backward()
    assert model.conditioner.query_projection.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.base.parameters())
    optimizer.step()
    assert all(n.startswith('conditioner.') for n,p in model.named_parameters() if p.requires_grad)
    for key,v in model.base.state_dict().items():torch.testing.assert_close(v,frozen[key],rtol=0,atol=0)
    model.eval();detail=model(x,y,tme,return_details=True)
    assert detail['tme_group_attention'].shape==(1,2,16)
    torch.testing.assert_close(detail['tme_group_attention'].sum(-1),torch.ones(1,2))
    assert len(model.base.prompt_learner._forward_hooks)==0


def test_zero_checkpoint_and_intervention_cleanup():
    model,x,y,tme=wrapper('brca',32);model.eval();model.tme_mode='zero'
    assert check_model(model,(x,y,tme),'zero')['zero_input_independent']
    saved={k:v.clone() for k,v in model.adapter_state_dict().items()}
    expected=model(x,y,tme)[0]
    with torch.no_grad():model.conditioner.output_projection.weight.add_(1)
    model.load_adapter_state_dict(saved)
    torch.testing.assert_close(model(x,y,tme)[0],expected)
    with pytest.raises(ValueError):model.load_adapter_state_dict({})
    def fail(*args,**kwargs):raise RuntimeError('synthetic forward failure')
    original=model.base.visual_attention.forward;model.base.visual_attention.forward=fail
    try:
        with pytest.raises(RuntimeError,match='synthetic'):model(x,y,tme)
        assert not model.base.prompt_learner._forward_hooks
    finally:model.base.visual_attention.forward=original
    regression_checks()


def test_train_only_standardization_and_donors():
    model,_,_,_=wrapper('nsclc',32)
    bridge=object.__new__(DyKoTMEMethod);bridge.cfg={'_fold_index':0}
    bridge.tme=pd.DataFrame([[1.]*62,[3.]*62,[1e6]*62],index=['a','b','heldout'])
    loader=SimpleNamespace(dataset=SimpleNamespace(df=pd.DataFrame({'slide_id':['a','b']})))
    bridge.prepare_fold(0,model,loader)
    torch.testing.assert_close(model.standardizer.mean,torch.full((62,),2.))
    phases={'train':[{'slide_id':'a','case_id':'p1'},{'slide_id':'b','case_id':'p2'}]}
    validate_donors(phases,{'train':{'a':'b','b':'a'}})
    with pytest.raises(ValueError):validate_donors(phases,{'train':{'a':'a','b':'b'}})


def test_four_configs_numeric_types_and_bank_scope(tmp_path):
    load_dotenv(PGVL/'.env')
    spec=json.loads((ROOT/'configs/dyko_tcga_16shot_20260912.json').read_text())
    base=load_yaml_config(PGVL/'benchmarks/tcga_nsclc/configs/dyko_plip/nsclc_16shot.yaml')
    for cohort in spec['cohorts']:
        protocol=load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/protocol.yaml')
        clip=load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/configs/muse_paired_clip_rn50/{cohort}_16shot.yaml')
        for encoder in spec['encoders']:
            key=encoder.replace('-','_')+'_20x'
            cfg=build_config(base,clip,protocol['feature_sources'][key],'test',cohort,encoder,spec['brca_assets'])
            write_runtime_config(tmp_path/f'{cohort}_{encoder}.yaml',cfg)
            for update in [{'weight_decay':'1e-5'},{'concept_count':999},{'feature_dim':17},{'max_patches':100}]:
                with pytest.raises((ValueError,TypeError)):validate_config({**deepcopy(cfg),**update})
    bank=json.loads(Path(spec['brca_assets']['source_bank_path']).read_text())
    assert len(bank['concepts'])==len({x['text'] for x in bank['concepts']})==64
    assert not any(x in ' '.join(c['text'].lower() for c in bank['concepts']) for x in ['e-cadherin','her2','estrogen receptor'])


def test_twenty_four_job_boundary():
    spec=json.loads((ROOT/'configs/dyko_tcga_16shot_20260912.json').read_text())
    plans=[dict(cohort=c,encoder=e,method='dyko',fold=f) for c in spec['cohorts'] for e in spec['encoders'] for f in spec['folds']]
    jobs=commands(dict(protocol=spec,output='/tmp/dyko_plan',plans=plans),Path('/tmp/launch.json'))
    assert len(jobs)==24 and sum(x['smoke'] for x in jobs)==4
    pairs={x['pair'] for x in jobs if x['smoke']}
    assert all(x['depends_on_smoke_pair'] in pairs for x in jobs if not x['smoke'])
