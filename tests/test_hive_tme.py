"""Eight focused checks for native graph parity, controls and fixed study scope."""
from collections import OrderedDict
from copy import deepcopy
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym','/path/to/PathoTME/scripts']
from methods.hive_mil.model import CustomCLIP
from methods.hive_mil.dataset import compile_hierarchy,resolve_hierarchy_geometry as native_geometry
from pathotme.hive_geometry import hierarchy_indices,resolve_hierarchy_geometry
from pathotme.hive_tme import PairedHiVECore,PairedHiVE,TMEGuidedHiVE,HiVETMEMethod
from pathotme.hive_contract import build_config,validate_config
from pathotme.hive_text import HiVETokenPromptLearner
from pathotme.focus_contract import validate_donors
from pathotme.locked_tcga import ROOT,PGVL
from pathotme.yaml_runtime_config import write_runtime_config
from common.configuration import load_dotenv,load_yaml_config
from launch_hive_tcga import commands

torch.set_num_threads(1)


class FixedPromptLearner(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.register_buffer('features',torch.randn(32,width))
        self.register_buffer('tokenized_prompts',torch.zeros(32,1,dtype=torch.long))
    def forward(self,device=None): return self.features


class IdentityTower(nn.Module):
    def forward(self,features,ids):
        return OrderedDict(A=features[:16],B=features[16:])


def wrapper(cohort,width):
    torch.manual_seed(41)
    raw=768 if width==512 else width
    projector=nn.Linear(raw,width,bias=False) if raw!=width else nn.Identity()
    core=PairedHiVECore(FixedPromptLearner(width),IdentityTower(),width,torch.tensor(2.7))
    base=PairedHiVE(core,projector,raw)
    model=TMEGuidedHiVE(base,cohort,hidden_dim=16,num_heads=4,dropout=0.0)
    values=torch.randn(32,model.standardizer.feature_count);model.standardizer.fit(values)
    low=torch.randn(5,raw);high=torch.randn(5,16,raw);high[:,-2:]=0
    return model,low,high,torch.tensor([1]),values[:1]


@pytest.mark.parametrize('cohort,width',[('nsclc',512),('brca',512),('nsclc',1024),('brca',1024)])
def test_native_graph_parity_tme_gradients_and_frozen_parameters(cohort,width):
    model,low,high,label,values=wrapper(cohort,width);model.eval()
    assert PairedHiVECore.forward is CustomCLIP.forward
    assert PairedHiVECore.build_hier_hetero_graph is CustomCLIP.build_hier_hetero_graph
    state={k:v.clone() for k,v in model.base.state_dict().items()}
    with torch.no_grad():
        native=model.base(low,high)
        native_mask=model.base.custom_vlm.x5_mask.clone()
        native_low_text=model.base.custom_vlm.x5_text_embeddings.clone()
        projection=model.conditioner.output_projection.weight.clone()
        model.conditioner.output_projection.weight.zero_()
        detail=model(low,high,label,values,return_details=True)
        torch.testing.assert_close(detail['logits'],native,rtol=0,atol=0)
        torch.testing.assert_close(detail['retained_parent_mask'],native_mask)
        torch.testing.assert_close(detail['low_text_embeddings'],native_low_text)
        model.conditioner.output_projection.weight.copy_(projection)
        other=model(low,high,label,values+10,return_details=True)['logits']
        assert not torch.allclose(model(low,high,label,values,return_details=True)['logits'],other,rtol=1e-7,atol=1e-7)
    model.train();assert not model.base.training
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    tme=values.clone().requires_grad_()
    model(low,high,label,tme)[2].backward()
    assert tme.grad is not None and tme.grad.abs().sum()>0
    assert model.conditioner.query_projection.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.base.parameters())
    assert all(n.startswith('conditioner.') for n,p in model.named_parameters() if p.requires_grad)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    for k,v in model.base.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
    weights=model(low,high,label,values,return_details=True)['tme_group_attention']
    assert weights.shape==(1,32,16)
    torch.testing.assert_close(weights.sum(-1),torch.ones(1,32))
    assert not model.base.custom_vlm.text_encoder._forward_hooks


def test_zero_label_checkpoint_and_hook_restoration():
    model,low,high,label,values=wrapper('brca',32);model.tme_mode='zero';model.eval()
    expected=model(low,high,label,values)[0]
    torch.testing.assert_close(expected,model(low,high,1-label,values+10000)[0],rtol=0,atol=0)
    saved={k:v.clone() for k,v in model.adapter_state_dict().items()}
    with torch.no_grad():model.conditioner.output_projection.weight.add_(1)
    model.load_adapter_state_dict(saved)
    torch.testing.assert_close(expected,model(low,high,label,values)[0])
    with pytest.raises(ValueError):model.load_adapter_state_dict({**saved,'extra':torch.tensor(0)})
    original=model.base.custom_vlm.process_similarity
    def fail(*args):raise RuntimeError('deliberate graph failure')
    model.base.custom_vlm.process_similarity=fail
    with pytest.raises(RuntimeError,match='deliberate'):model(low,high,label,values)
    assert not model.base.custom_vlm.text_encoder._forward_hooks
    model.base.custom_vlm.process_similarity=original
    torch.testing.assert_close(expected,model(low,high,label,values)[0])


def test_hierarchy_geometry_parent_order_and_padding_match_native():
    low=np.array([[8,0],[0,0],[0,8]]);high=np.array([[2,2],[10,1],[0,0],[7,7]])
    parents,children=hierarchy_indices(low,high,8)
    np.testing.assert_array_equal(parents,[0,1])
    np.testing.assert_array_equal(children[0,:1],[1]);np.testing.assert_array_equal(children[1,:3],[0,2,3])
    low_f=np.arange(12,dtype=np.float32).reshape(3,4);high_f=np.arange(16,dtype=np.float32).reshape(4,4)
    native_low,native_high,valid=compile_hierarchy(low_f,low,high_f,high,low_patch_span_level0=8)
    expected=np.zeros((2,16,4),np.float32)
    for i,row in enumerate(children):expected[i,row>=0]=high_f[row[row>=0]]
    np.testing.assert_array_equal(native_low.numpy(),low_f[parents])
    np.testing.assert_array_equal(native_high.numpy(),expected)
    np.testing.assert_array_equal(valid.numpy(),children>=0)
    for mag in [20,40,60,80]:
        attrs=[dict(level0_width=40000,level0_height=50000,level0_magnification=mag,
                    target_magnification=s,patch_size=224,patch_size_level0=224*mag//s) for s in [5,20]]
        assert resolve_hierarchy_geometry(*attrs)==native_geometry(*attrs)
    with pytest.raises(ValueError):hierarchy_indices(low,np.repeat(high[:1],2,axis=0),8)
    with pytest.raises(ValueError):hierarchy_indices(low[:1],np.array([[8,i] for i in range(17)]),32)


def test_training_only_statistics_and_donor_boundaries():
    model,*_=wrapper('nsclc',32)
    method=object.__new__(HiVETMEMethod);method.cfg={'_fold_index':0}
    method.tme=pd.DataFrame([[1.]*62,[3.]*62,[1e6]*62],index=['a','b','heldout'])
    loader=SimpleNamespace(dataset=SimpleNamespace(frame=pd.DataFrame({'slide_id':['a','b']})))
    method.prepare_fold(0,model,loader)
    torch.testing.assert_close(model.standardizer.mean,torch.full((62,),2.))
    assert model.standardizer.scale.eq(1).all()
    phases={'train':[{'slide_id':'a','case_id':'p1'},{'slide_id':'b','case_id':'p2'}]}
    validate_donors(phases,{'train':{'a':'b','b':'a'}})
    for donors in [{'a':'a','b':'b'},{'a':'heldout','b':'a'}]:
        with pytest.raises(ValueError):validate_donors(phases,{'train':donors})


def test_prefix_numeric_configs_and_24_job_boundary(tmp_path):
    ids=torch.zeros(32,77,dtype=torch.long);ids[:,0]=49406;ids[:,1:17]=343;ids[:,30]=49407
    embeddings=torch.randn(32,77,12)
    learner=HiVETokenPromptLearner(embeddings,ids);observed=learner()
    torch.testing.assert_close(observed[:,:1],embeddings[:,:1]);torch.testing.assert_close(observed[:,17:],embeddings[:,17:])
    assert learner.ctx.shape==(16,12)
    load_dotenv(PGVL/'.env')
    for cohort in ['nsclc','brca']:
        for encoder in ['plip','clip-rn50']:
            base=load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/configs/hive_mil/{cohort}_16shot.yaml')
            paired=load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/configs/muse_paired_{encoder.replace("-","_")}/{cohort}_16shot.yaml')
            cfg=build_config(base,paired,'test',cohort,encoder)
            assert cfg['text_prompt_path']==base['text_prompt_path']
            path=tmp_path/f'{cohort}_{encoder}.yaml';write_runtime_config(path,cfg)
            validate_config(load_yaml_config(path))
            for update in [{'feature_dim':111},{'n_ctx':4},{'max_children':4},{'weight_decay':'1e-5'}, {'max_patches':1000}]:
                with pytest.raises((ValueError,TypeError)):validate_config({**deepcopy(cfg),**update})
    spec=json.loads((ROOT/'configs/hive_tcga_16shot_20260912.json').read_text())
    plans=[dict(cohort=c,encoder=e,method='hive_mil',fold=f)
           for c in spec['cohorts'] for e in spec['encoders'] for f in spec['folds']]
    jobs=commands(dict(protocol=spec,output='/tmp/hive_test_plan',plans=plans),Path('/tmp/launch.json'))
    assert len(jobs)==24 and sum(p['smoke'] for p in jobs)==4
    pairs={p['pair'] for p in jobs if p['smoke']}
    assert len(pairs)==4
    assert all(p['depends_on_smoke_pair'] in pairs for p in jobs if not p['smoke'])
