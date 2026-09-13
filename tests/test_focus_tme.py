"""Focused research checks: native parity, real gradients, controls and scope."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym', '/path/to/PathoTME/scripts']
from methods.focus.model import FOCUS
from pathotme.focus_tme import TMEGuidedFOCUS, FocusTMEMethod
from pathotme.focus_contract import build_config, validate_config, validate_donors
from pathotme.locked_tcga import ROOT, PGVL
from pathotme.yaml_runtime_config import write_runtime_config
from common.configuration import load_dotenv, load_yaml_config
from launch_focus_tcga import commands

torch.set_num_threads(1)


def wrapper(cohort, width):
    torch.manual_seed(41)
    cfg = SimpleNamespace(input_size=width, window_size=8, sim_threshold=0.8,
                          max_context_length=12)
    base = FOCUS(cfg, num_classes=2, prompt_features=torch.randn(4, width))
    model = TMEGuidedFOCUS(base, cohort, hidden_dim=16, num_heads=4, dropout=0.0)
    count = model.standardizer.feature_count
    train = torch.randn(32, count)
    model.standardizer.fit(train)
    return model, torch.randn(25, width), torch.tensor([1]), train[:1]


@pytest.mark.parametrize('cohort,width', [('nsclc', 512), ('brca', 512), ('nsclc', 1024), ('brca', 1024)])
def test_native_parity_and_real_gradient(cohort, width):
    model, features, label, values = wrapper(cohort, width)
    model.eval()
    initial = {k: v.clone() for k, v in model.base.state_dict().items()}
    with torch.no_grad():
        original = model.conditioner.output_projection.weight.clone()
        model.conditioner.output_projection.weight.zero_()
        native = model.base(features, features, label, return_details=True)
        observed = model(features, label, values, return_details=True)
        for key in ('logits', 'probabilities', 'patch_attention', 'selected_patch_indices'):
            torch.testing.assert_close(observed[key], native[key])
        model.conditioner.output_projection.weight.copy_(original)
        actual = model(features, label, values)[0]
        changed = model(features, label, values + 10)[0]
        assert not torch.allclose(actual, changed, atol=1e-7, rtol=1e-7)
    model.train()
    assert not model.base.training
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    tme = values.clone().requires_grad_()
    model(features, label, tme)[2].backward()
    assert tme.grad is not None and tme.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    assert all(n.startswith('conditioner.') for n, p in model.named_parameters() if p.requires_grad)
    assert model.conditioner.query_projection.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    for key, value in model.base.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    model.eval()
    details = model(features, label, values, return_details=True)
    torch.testing.assert_close(details['patch_attention'].sum(-1), torch.ones(2))
    torch.testing.assert_close(details['tme_group_attention'].sum(-1), torch.ones(1, 2))
    assert details['tme_group_attention'].shape == (1, 2, 16)
    assert len(model.conditioner.token_names) == 16


def test_zero_control_and_checkpoint_roundtrip():
    model, features, label, values = wrapper('brca', 32)
    model.tme_mode = 'zero'
    model.eval()
    torch.testing.assert_close(model(features, label, values)[0], model(features, label, values + 1000)[0])
    saved = {k: v.clone() for k, v in model.adapter_state_dict().items()}
    expected = model(features, label, values)[0]
    with torch.no_grad():
        model.conditioner.output_projection.weight.add_(1)
    model.load_adapter_state_dict(saved)
    torch.testing.assert_close(model(features, label, values)[0], expected)
    with pytest.raises(ValueError):
        model.load_adapter_state_dict({**saved, 'base.classifier.weight': model.base.classifier.weight})


def test_training_only_statistics_and_fixed_donors():
    model, _, _, _ = wrapper('nsclc', 32)
    method = object.__new__(FocusTMEMethod)
    method.cfg = {'_fold_index': 0}
    method.tme = pd.DataFrame([[1.] * 62, [3.] * 62, [1e6] * 62], index=['a', 'b', 'heldout'])
    loader = SimpleNamespace(dataset=SimpleNamespace(df=pd.DataFrame({'slide_id': ['a', 'b']})))
    method.prepare_fold(0, model, loader)
    torch.testing.assert_close(model.standardizer.mean, torch.full((62,), 2.))
    assert model.standardizer.scale.eq(1).all()
    phases = {'train': [{'slide_id': 'a', 'case_id': 'p1'}, {'slide_id': 'b', 'case_id': 'p2'}]}
    validate_donors(phases, {'train': {'a': 'b', 'b': 'a'}})
    with pytest.raises(ValueError):
        validate_donors(phases, {'train': {'a': 'a', 'b': 'b'}})
    with pytest.raises(ValueError):
        validate_donors(phases, {'train': {'a': 'heldout', 'b': 'a'}})


def test_configs_and_numeric_roundtrip(tmp_path):
    load_dotenv(PGVL/'.env')
    for cohort in ('nsclc', 'brca'):
        base = load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/configs/focus_plip/{cohort}_16shot.yaml')
        clip = load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/configs/vila_mil/{cohort}_16shot.yaml')
        protocol = load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/protocol.yaml')
        for encoder, key in [('plip', 'plip_20x'), ('clip-rn50', 'clip_rn50_20x')]:
            cfg = build_config(base, clip, protocol['feature_sources'][key], 'test', cohort, encoder)
            assert cfg['text_prompt_path'] == base['text_prompt_path']
            assert cfg['focus_prompt_bank_sha256'] == base['focus_prompt_bank_sha256']
            write_runtime_config(tmp_path/f'{cohort}_{encoder}.yaml', cfg)
            for update in ({'feature_resolutions': {'high': '10x'}},
                           {'encoder_extension_strategy': 'paired_token_context_v1'},
                           {'feature_space_id': 'wrong'}, {'weight_decay': '1e-5'}):
                with pytest.raises((ValueError, TypeError)):
                    validate_config({**deepcopy(cfg), **update})


def test_bounded_launch_and_encoder_dependencies():
    spec = json.loads((ROOT/'configs/focus_tcga_16shot_20260911.json').read_text())
    plans = [dict(cohort=c, encoder=e, method='focus', fold=f)
             for c in spec['cohorts'] for e in spec['encoders'] for f in spec['folds']]
    jobs = commands(dict(protocol=spec, output='/tmp/focus_test_plan', plans=plans), Path('/tmp/launch.json'))
    assert len(jobs) == 24 and sum(p['smoke'] for p in jobs) == 4
    smoke_pairs = {p['pair'] for p in jobs if p['smoke']}
    assert len(smoke_pairs) == 4
    assert all(p['depends_on_smoke_pair'] in smoke_pairs for p in jobs if not p['smoke'])
    assert all('run_focus_tcga.py' in p['command'][-1] for p in jobs)
