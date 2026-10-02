"""Focused cached-encoder checks; synthetic features are not research results."""
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from common.configuration import load_dotenv,load_yaml_config
from pathotme.mscpt_contract import build_config
from pathotme.mscpt_tme import MSCPTNativeMethod,TMEGuidedMSCPT
from pathotme.mscpt_smoke_checks import check_model
from pathotme.locked_tcga import sha

@pytest.fixture(scope='module',params=['plip','clip-rn50'])
def built(request):
    root=Path('/path/to/PGVL-Gym');load_dotenv(root/'.env')
    base=load_yaml_config(root/'benchmarks/tcga_nsclc/configs/mscpt_5x20x/nsclc_16shot.yaml')
    key=request.param.replace('-','_')
    paired=load_yaml_config(root/f'benchmarks/tcga_nsclc/configs/muse_paired_{key}/nsclc_16shot.yaml')
    cfg=build_config(base,paired,'cpu_check','nsclc',request.param)
    cfg['description_prompt_sha256']=sha(cfg['description_prompt_path'])
    cfg['selection_prompt_sha256']=sha(cfg['selection_prompt_path'])
    torch.manual_seed(17)
    method=MSCPTNativeMethod(cfg,'cpu');model=method.build_model()
    return cfg,method,model


def test_native_and_semantic_boundaries(built):
    cfg,method,base=built;torch.manual_seed(18)
    high=torch.randn(1,24,cfg['feature_dim']);low=torch.randn(1,12,cfg['feature_dim'])
    base.train(); optimizer=method.build_optimizer(base)
    step=method.train_step((high,low,torch.tensor([1])),base,optimizer,torch.nn.CrossEntropyLoss())
    assert np.isfinite(step['loss'])
    assert base.Custom_model.prompt_learner.p_input.grad.abs().sum()>0
    assert base.Custom_model.gcn_prompt_learner_big.conv1.lin.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in base.parameters() if p.grad is not None)
    for p in base.parameters():p.grad=None
    guided=TMEGuidedMSCPT(base,'nsclc',dropout=0.0)
    guided.standardizer.fit(np.random.default_rng(1).normal(size=(8,62)))
    tme=torch.randn(1,62);inputs=(high,low,torch.tensor([1]),tme)
    diagnostic=check_model(guided,inputs,'actual')
    assert diagnostic['tme_margin_gradient_l1']>0
    guided.train();detail=guided(*inputs,return_details=True);detail['loss'].backward()
    assert guided.conditioner.output_projection.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in guided.base.parameters())
    assert len(detail['tme_group_attention'])==2
    assert not guided.base.Custom_model.frozen_semantics._forward_hooks
    assert not guided.base.Custom_model.text_encoder._forward_hooks
    guided.tme_mode='zero';check_model(guided,inputs,'zero')
    # Restore native parameter state for any subsequent uses of the fixture.


def test_frozen_text_encoding_parity(built):
    cfg,method,base=built
    core=base.Custom_model
    bank=core.text_prompts
    if cfg['backbone']=='plip':
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(cfg['backbone_weights'],local_files_only=True)
        ids=tok(bank['LUAD']['small_mag'],padding='max_length',truncation=True,max_length=64,return_tensors='pt')
        with torch.no_grad():ref=core.model.get_text_features(**ids)
    else:
        # Reload only for this bounded cached checkpoint parity check.
        from pathotme.mscpt_text import token_ids,frozen_rn50
        encoder=method.load_encoder(weights_path=cfg['backbone_weights']).freeze()
        raw=encoder.raw_model.float().eval()
        ids=token_ids(bank['LUAD']['small_mag'])
        padded=torch.nn.functional.pad(ids,(0,13))
        with torch.no_grad():ref=raw.encode_text(padded)
        actual,_=frozen_rn50(raw,bank['LUAD']['small_mag'])
        torch.testing.assert_close(actual,torch.nn.functional.normalize(ref,dim=-1),rtol=2e-4,atol=2e-5)
    torch.testing.assert_close(core.frozen_text_bank[0],torch.nn.functional.normalize(ref,dim=-1),rtol=2e-4,atol=2e-5)


def test_hook_cleanup_on_exception(built):
    cfg,_,base=built
    guided=TMEGuidedMSCPT(base,'nsclc',dropout=0.0)
    guided.standardizer.fit(np.random.default_rng(4).normal(size=(8,62)))
    def fail(*args):raise RuntimeError('injected')
    handle=base.Custom_model.gcn_prompt_learner_big.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError,match='injected'):
            guided(torch.randn(1,24,cfg['feature_dim']),torch.randn(1,12,cfg['feature_dim']),torch.tensor([0]),torch.randn(1,62))
    finally:handle.remove()
    assert not base.Custom_model.frozen_semantics._forward_hooks
    assert not base.Custom_model.text_encoder._forward_hooks
