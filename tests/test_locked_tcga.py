"""Scientific invariants of the new common-cohort study and BRCA MGPATH path."""
import copy
import json
from pathlib import Path
import sys

sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym', '/path/to/PathoTME/scripts']
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from methods.mgpath.model import MGPathModel
from pathotme.locked_models import BreastGuidedMGPath, permute_table
from pathotme.locked_tcga import check_phases, membership, identity, atomic_json, load_launch
from pathotme.controlled_vila import donor_maps
from launch_locked_tcga import commands


class FixedPrompt(nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('texts', torch.randn(4, 4, 16))
    def forward(self):
        return self.texts


def model(mode='actual'):
    torch.manual_seed(23)
    base = MGPathModel(None, None, [], n_classes=2, source_dim=16, shared_dim=16,
                       prompt_module=FixedPrompt(), projector=nn.Identity(), logit_scale=torch.tensor(1.0))
    result = BreastGuidedMGPath(base, tme_mode=mode, hidden_dim=16, dropout=0)
    result.standardizer.fit(np.stack([np.zeros(64), np.ones(64)]))
    return result.eval()


def bags():
    torch.manual_seed(9)
    return torch.randn(5,16), torch.arange(5).repeat(2,1), torch.randn(7,16), torch.arange(7).repeat(2,1)


def test_brca_mgpath_native_equivalence_and_panel_binding():
    m = model(); inputs = bags(); tme = torch.rand(1,64)
    torch.testing.assert_close(m(*inputs,tme,bypass_conditioner=True),m.base(*inputs))
    details = m(*inputs,tme,return_details=True)
    assert details['low_tme_group_attention'].shape == (1,64,6)
    assert details['high_tme_group_attention'].shape == (1,64,10)
    with torch.no_grad(): m.conditioner.output_projection.weight.zero_()
    torch.testing.assert_close(m(*inputs,tme),m.base(*inputs))


def test_brca_mgpath_real_sensitivity_zero_invariance_and_capacity():
    m = model(); z = model('zero'); inputs = bags()
    assert not torch.allclose(m(*inputs,torch.zeros(1,64)),m(*inputs,torch.ones(1,64)))
    torch.testing.assert_close(z(*inputs,torch.zeros(1,64)),z(*inputs,torch.ones(1,64)))
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == sum(p.numel() for p in z.parameters() if p.requires_grad)


def test_brca_mgpath_frozen_base_gradients_and_roundtrip():
    m = model(); before = copy.deepcopy(m.base.state_dict())
    m.train(); m(*bags(),torch.randn(1,64)).square().sum().backward()
    assert not m.base.training
    assert all(p.grad is None and not p.requires_grad for p in m.base.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.conditioner.parameters())
    for k,v in before.items(): torch.testing.assert_close(v,m.base.state_dict()[k])
    clone = model(); clone.load_adapter_state_dict(m.adapter_state_dict()); m.eval()
    inputs=bags(); tme=torch.rand(1,64)
    torch.testing.assert_close(m(*inputs,tme),clone(*inputs,tme))
    with pytest.raises(ValueError): m(*inputs,torch.zeros(1,62))


def phases():
    return {phase:[{'slide_id':f'{phase}{i}','case_id':f'{phase}_p{i}','label_id':str(i%2)} for i in range(32)] for phase in ['train','val','test']}


def test_shuffle_preserves_missingness_rows_and_ignores_labels():
    p=phases(); maps=donor_maps(p,1,0)
    ids=[r['slide_id'] for v in p.values() for r in v]
    table=pd.DataFrame({'a':range(len(ids)),'b':[float('nan') if i%3==0 else i for i in range(len(ids))]},index=ids)
    out=permute_table(table,maps)
    for phase,mapping in maps.items():
        assert sorted(out.loc[list(mapping),'a']) == sorted(table.loc[list(mapping),'a'])
        for recipient,donor in mapping.items():
            assert recipient != donor
            np.testing.assert_equal(out.loc[recipient].to_numpy(),table.loc[donor].to_numpy())
    changed=copy.deepcopy(p)
    for v in changed.values():
        for r in v:r['label_id']=str(1-int(r['label_id']))
    assert donor_maps(changed,1,0)==maps


def test_common_split_rejects_missing_train_patient_and_leakage():
    p=phases(); ids={r['slide_id'] for v in p.values() for r in v}
    check_phases(p,ids)
    with pytest.raises(ValueError):check_phases(p,ids-{'train0'})
    p['test'][0]['case_id']=p['train'][0]['case_id']
    with pytest.raises(ValueError):check_phases(p,ids)


def test_membership_detects_labels_order_and_subject_changes():
    original=phases()['train']; changed=copy.deepcopy(original)
    changed[0]['case_id']='other'
    assert membership(original)!=membership(changed)
    assert membership(original)!=membership(original[::-1])


def test_launch_rejects_contract_and_source_drift(tmp_path):
    source=tmp_path/'input';source.write_text('a')
    from pathotme.locked_tcga import sha
    payload={'file_sha256':{str(source):sha(source)},'arbitrary':1}
    payload['identity']=identity(payload);path=tmp_path/'launch.json';atomic_json(path,payload)
    assert load_launch(path)['arbitrary']==1
    source.write_text('b')
    with pytest.raises(ValueError):load_launch(path)
    payload['arbitrary']=2;atomic_json(path,payload)
    with pytest.raises(ValueError):load_launch(path)


def test_launcher_is_bounded_and_pairs_smokes():
    protocol=json.loads(Path('/path/to/PathoTME/configs/tcga_locked_16shot_20260909.json').read_text())
    launch={'output':'/tmp/example','protocol':protocol,'plans':[{'cohort':c,'method':m,'fold':f} for c in protocol['cohorts'] for f in protocol['folds'] for m in protocol['methods']]}
    result=commands(launch,Path('/tmp/example/launch.json'))
    assert len(result)==24
    assert sum(r['smoke'] for r in result)==4
    assert all(r['depends_on_smoke_pair']==r['pair'] for r in result if not r['smoke'])
    assert all('CPTAC' not in str(r['command']) for r in result)


def test_fold_runner_selects_before_test_and_resumes_without_retraining(tmp_path, monkeypatch):
    import csv
    import train
    import run_locked_tcga as runner
    import pathotme.locked_models as bridges
    from pathotme.guided_mgpath_adapter import TMEGuidedMGPathMethod
    events=[]
    class TinyBridge:
        def build_model(self):return model()
        def prepare_fold(self,*args):pass
        def build_optimizer(self,m):return torch.optim.Adam(m.conditioner.parameters(),lr=1e-4)
        def inputs(self,batch):return (*batch[:4],torch.ones(1,64)),batch[-1],batch[-2]
        train_step=TMEGuidedMGPathMethod.train_step
        eval_step=TMEGuidedMGPathMethod.eval_step
        eval_step_with_details=TMEGuidedMGPathMethod.eval_step_with_details
    phases={}
    for phase in ['train','val','test']:
        phases[phase]=[(*bags(),{'slide_id':[f'{phase}{i}'],'case_id':[f'{phase}{i}']},torch.tensor([i])) for i in range(2)]
    monkeypatch.setattr(train,'build_loaders',lambda *a:tuple(phases.values()))
    monkeypatch.setattr(bridges,'make_method',lambda *a:TinyBridge())
    original_epoch=runner.__dict__.get('_run_epoch')
    import run_vila_guided
    native_epoch=run_vila_guided._run_epoch
    def epoch(loader,*args,**kwargs):
        events.append('train' if loader is phases['train'] else 'val')
        return native_epoch(loader,*args,**kwargs)
    monkeypatch.setattr(run_vila_guided,'_run_epoch',epoch)
    protocol=json.loads(Path('/path/to/PathoTME/configs/tcga_locked_16shot_20260909.json').read_text())
    protocol['adapter']['epochs']=2
    launch={'protocol':protocol,'identity':'test-lock','output':str(tmp_path)}
    donor=tmp_path/'donors.json';atomic_json(donor,{})
    split=tmp_path/'splits/fold0';split.mkdir(parents=True)
    with (split/'test.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=['slide_id','case_id','label_id']);writer.writeheader()
        writer.writerows({'slide_id':f'test{i}','case_id':f'test{i}','label_id':i} for i in range(2))
    plan={'cohort':'brca','method':'mgpath','fold':0,'output':str(tmp_path/'run'),
          'native_dir':str(tmp_path),'donor_maps':str(donor),'tme_csv':str(tmp_path/'unused')}
    cfg={'split_dir':str(split.parent)}
    actual_evaluate=runner.evaluate
    def interrupt_test(*args):
        assert events==['train','val','train','val']
        assert (tmp_path/'run/actual/best.pt').is_file()
        raise RuntimeError('simulated interruption before heldout evaluation')
    monkeypatch.setattr(runner,'evaluate',interrupt_test)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        runner.arm_run(launch,plan,cfg,'actual',False,'cpu',{})
    monkeypatch.setattr(runner,'evaluate',actual_evaluate)
    result=runner.arm_run(launch,plan,cfg,'actual',False,'cpu',{})
    assert events==['train','val','train','val']
    assert result['status']=='completed'
    assert len(rows_for_test(tmp_path/'run/actual/predictions.csv'))==2
    assert runner.arm_run(launch,plan,cfg,'actual',False,'cpu',{})==result


def rows_for_test(path):
    import csv
    with path.open() as h:return list(csv.DictReader(h))


def test_feature_header_gate_rejects_same_width_wrong_encoder_or_geometry(tmp_path):
    import h5py
    from prepare_locked_tcga import inspect_feature
    path=tmp_path/'feature.h5'
    with h5py.File(path,'w') as h:
        features=h.create_dataset('features',data=np.zeros((2,768),dtype='float32'))
        coords=h.create_dataset('coords',data=np.zeros((2,2),dtype='int64'))
        features.attrs['encoder']='plip';coords.attrs['patch_size']=224;coords.attrs['target_magnification']=5
    task=(str(path),'plip',768,5,{'feature_space_id':'hf:vinid/plip#vision-preprojection','feature_representation':'vision_preprojection'})
    _,result=inspect_feature(task)
    assert result['registry_representation']=='vision_preprojection'
    with h5py.File(path,'a') as h:h['features'].attrs['encoder']='wrong_encoder'
    with pytest.raises(ValueError,match='identity/geometry'):inspect_feature(task)
    with h5py.File(path,'a') as h:
        h['features'].attrs['encoder']='plip';h['coords'].attrs['target_magnification']=10
    with pytest.raises(ValueError,match='identity/geometry'):inspect_feature(task)
