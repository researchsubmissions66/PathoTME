"""Encoder-boundary, prompt-policy and real native-trainer lifecycle checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym','/path/to/PathoTME/scripts']

import numpy as np
import pytest
import torch
from torch import nn
from common.configuration import load_dotenv,load_yaml_config
from pathotme.cross_encoder_contract import validate_clip_config
from pathotme.cross_encoder_models import CLIPMGPathMethod,PLIPGuidedViLa
from pathotme.guided_mgpath import TMEGuidedMGPath
from pathotme.locked_models import BreastGuidedMGPath
from prepare_cross_encoder_tcga import build_clip_cfg,clip_tokens
from launch_cross_encoder_tcga import commands
from pathotme.locked_tcga import PGVL


def config():
    load_dotenv(PGVL/'.env')
    base=load_yaml_config(PGVL/'benchmarks/tcga_nsclc/configs/mgpath/nsclc_16shot.yaml')
    clip=load_yaml_config(PGVL/'benchmarks/tcga_nsclc/configs/vila_mil/nsclc_16shot.yaml')
    return build_clip_cfg(base,clip,'unit_test','nsclc')


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__();self.logit_scale=nn.Parameter(torch.tensor(1.0))
    def encode_text(self,tokens):
        return torch.sin(torch.arange(1024)[None,:]/32+tokens.sum(1)[:,None]*.001)


class FakeBundle:
    def __init__(self):
        self.raw_model=FakeCLIP()
        self.spec=SimpleNamespace(shared_dim=1024,feature_space_id='openai/clip-rn50@official')
    def freeze(self):self.raw_model.requires_grad_(False).eval();return self


def tiny_native(monkeypatch):
    monkeypatch.setattr(CLIPMGPathMethod,'load_encoder',lambda *a,**k:FakeBundle())
    torch.manual_seed(41)
    return CLIPMGPathMethod(config(),'cpu').build_model()


def bags():
    torch.manual_seed(19)
    return torch.randn(4,1024),torch.arange(4).repeat(2,1),torch.randn(6,1024),torch.arange(6).repeat(2,1)


def test_explicit_contract_rejects_wrong_same_width_encoder_and_projection():
    cfg=config();validate_clip_config(cfg)
    for key,value in [('backbone','musk'),('feature_space_id','hf:xiangjx/musk'),
                      ('feature_projection','learned'),('pathotme_text_policy','implicit_truncation')]:
        bad=copy.deepcopy(cfg);bad[key]=value
        with pytest.raises(ValueError):validate_clip_config(bad)


def test_native_clip_token_consumption_matches_audit_for_long_and_short_text():
    import clip
    prompts=['lung adenocarcinoma','a glandular tumor '*200]
    audit=clip_tokens(prompts)
    assert audit['original_token_counts_including_specials'][1]>77
    assert audit['consumed_token_ids']==clip.tokenize(prompts,context_length=77,truncate=True).tolist()
    assert audit['consumed_token_ids'][1][-1]==49407


def test_native_clip_graph_model_has_four_trainable_contexts_and_exact_token_audit(monkeypatch):
    native=tiny_native(monkeypatch)
    assert native.source_dim==1024 and native.centers.shape==(64,1024)
    assert native.prompt.context.shape==(4,1,1024)
    assert native.pathotme_consumed_prompt_tokens.shape==(4,77)
    loss=nn.CrossEntropyLoss()(native(*bags()),torch.tensor([1]));loss.backward()
    assert native.prompt.context.grad is not None and native.prompt.context.grad.abs().sum()>0
    assert native.graph.projection.weight.grad is not None


@pytest.mark.parametrize('breast',[False,True])
def test_rn50_guided_path_reproduces_native_and_freezes_base(monkeypatch,breast):
    native=tiny_native(monkeypatch)
    wrapper=BreastGuidedMGPath if breast else TMEGuidedMGPath
    width=64 if breast else 62
    model=wrapper(native,hidden_dim=16,dropout=0).eval()
    model.standardizer.fit(np.stack([np.zeros(width),np.ones(width)]))
    inputs=bags();tme=torch.rand(1,width)
    torch.testing.assert_close(model(*inputs,tme,bypass_conditioner=True),native(*inputs))
    model.train();model(*inputs,tme).square().sum().backward()
    assert all(p.grad is None and not p.requires_grad for p in native.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.conditioner.parameters())
    model.eval();before=model(*inputs,tme)
    state=model.adapter_state_dict();model.load_adapter_state_dict(state)
    torch.testing.assert_close(before,model(*inputs,tme))


def test_plip_vila_bridge_applies_exact_frozen_projection_before_conditioning():
    bridge=PLIPGuidedViLa.__new__(PLIPGuidedViLa)
    bridge.cfg={'backbone':'plip'};bridge.device='cpu';bridge.is_encoder_extension=True
    bridge._tme_values=lambda metadata:torch.zeros(1,62)
    base=nn.Module();base.paired_feature_projector=nn.Linear(768,512,bias=False).requires_grad_(False)
    model=SimpleNamespace(base=base)
    low,high=torch.randn(1,4,768),torch.randn(1,6,768)
    inputs,meta=bridge._inputs((low,high,{'slide_id':['s'],'case_id':['p']},torch.tensor([1])),model)
    torch.testing.assert_close(inputs[0],base.paired_feature_projector(low.squeeze(0)))
    torch.testing.assert_close(inputs[2],base.paired_feature_projector(high.squeeze(0)))
    assert inputs[0].shape==(4,512) and inputs[2].shape==(6,512)


def test_added_launcher_only_submits_the_missing_twenty_four_jobs():
    protocol=json.loads(Path('/path/to/PathoTME/configs/tcga_locked_16shot_20260909.json').read_text())
    launch={'protocol':protocol,'output':'/tmp/cross_unit','plans':[{'cohort':c,'method':m,'fold':f} for c in ['nsclc','brca'] for m in ['vila_mil','mgpath'] for f in range(5)]}
    plans=commands(launch,Path('/tmp/cross_unit/launch.json'))
    assert len(plans)==24 and sum(p['smoke'] for p in plans)==4
    assert all(p['name'].startswith('ptme-cross-') for p in plans)
    assert all('run_cross_encoder_tcga.py' in p['command'][-1] for p in plans)


def test_native_extension_uses_real_fold_trainer_and_binds_resume_state(monkeypatch,tmp_path):
    import csv
    import train
    import run_cross_encoder_tcga as runner
    monkeypatch.setattr(CLIPMGPathMethod,'load_encoder',lambda *a,**k:FakeBundle())
    cfg=config();out=tmp_path/'native';cfg.update({'results_dir':str(out),'split_dir':str(tmp_path/'splits'),
        'epochs':2,'k_start':0,'k_end':1,'num_workers':0})
    phases=[]
    for phase in ['train','val','test']:
        phase_batches=[]
        for i in range(2):
            low,le,high,he=bags()
            phase_batches.append((low[None],le[None],high[None],he[None],
                {'slide_id':[f'{phase}{i}'],'case_id':[f'{phase}{i}']},torch.tensor([i])))
        phases.append(phase_batches)
    monkeypatch.setattr(train,'build_loaders',lambda *a:phases)
    split=tmp_path/'splits/fold0';split.mkdir(parents=True)
    with (split/'test.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=['slide_id','case_id','label_id']);writer.writeheader()
        writer.writerows({'slide_id':f'test{i}','case_id':f'test{i}','label_id':i} for i in range(2))
    plan={'native_reused':False,'native_dir':str(out),'method':'mgpath','fold':0}
    first=runner.native_stage(plan,cfg,'cpu')
    assert first==runner.native_stage(plan,cfg,'cpu')
    assert (out/'producer.json').is_file()
    bad={**cfg,'lr':cfg['lr']*2}
    with pytest.raises(Exception):runner.validate_native(plan,bad)
