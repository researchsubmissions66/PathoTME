"""Dependency-light contracts, launch planning and leakage guards."""
import copy
import importlib.util
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), '/path/to/PGVL-Gym']
from pathotme.standalone_teacher import CONDITIONS, NEW_CONDITIONS, subtype_logits, source_arrays, complementarity
from pathotme.controlled_vila import donor_maps
from pathotme.brca_features import panel_spec
import run_standalone_tme_teacher as runner
import launch_standalone_tme_teacher as launcher


def test_logits_preserve_order_temperature_and_finiteness():
    class Teacher:
        classes_ = [0, 1]
        def decision_function(self, values):
            return np.asarray(values)
    model = Teacher(); scores = np.array([-1000., -2., 0., 2., 1000.])
    logits = subtype_logits(model, scores)
    assert np.isfinite(logits).all()
    np.testing.assert_allclose(logits[:, 1]-logits[:, 0], scores)
    np.testing.assert_allclose((logits/2)[:, 1]-(logits/2)[:, 0], scores/2)
    model.classes_ = [1, 0]
    with pytest.raises(ValueError): subtype_logits(model, scores)
    model.classes_ = [0, 1]
    with pytest.raises(ValueError): subtype_logits(model, [float('inf')])


def test_teacher_never_transforms_test_values_and_rejects_cross_split_donors():
    phases = {p: [{'slide_id':f'{p}{i}', 'case_id':f'{p}{i}', 'label_id':i%2} for i in range(4)]
              for p in ('train','val')}
    names = panel_spec('brca_morph64_v1')['feature_names']
    rows = [{'slide_id':r['slide_id'], **{n:str(i+1) for n in names}}
            for split in phases.values() for i,r in enumerate(split)]
    rows.append({'slide_id':'test_POISON', **{n:'not a number' for n in names}})
    arrays, labels = source_arrays(phases, rows, 'brca_morph64_v1')
    assert arrays['train'].shape == (4, 64) and labels['val'].tolist() == [0,1,0,1]
    donors = donor_maps(phases, 1, 0)
    shuffled, _ = source_arrays(phases, rows, 'brca_morph64_v1', donors)
    np.testing.assert_allclose(np.sort(shuffled['train'],axis=0), np.sort(arrays['train'],axis=0))
    donors['train']['train0'] = 'val0'
    with pytest.raises(ValueError): source_arrays(phases, rows, 'brca_morph64_v1', donors)
    with pytest.raises(ValueError): source_arrays({**phases,'test':[]}, rows, 'brca_morph64_v1')
    phases['val'][0]['case_id'] = 'train0'
    with pytest.raises(ValueError): source_arrays(phases, rows, 'brca_morph64_v1')


def test_complementarity_counts_disagreement_not_a_deployable_oracle():
    v = [[.9,.1],[.9,.1],[.1,.9],[.1,.9]]
    t = [[.9,.1],[.1,.9],[.9,.1],[.1,.9]]
    result = complementarity(v,t,[0,1,0,0])
    assert result['both_correct'] == 1 and result['both_wrong'] == 1
    assert result['teacher_correct_visual_wrong'] == 2
    assert result['oracle_either_correct_accuracy'] == .75
    with pytest.raises(ValueError): complementarity([[2,0]],[[1,0]],[0])


def minimal_contract(tmp_path):
    return {'conditions':list(CONDITIONS), 'reuse_conditions':list(CONDITIONS[:2]), 'folds':[0],
        'results_root':str(tmp_path/'new'), 'matched_contract':'old',
        'teacher':{'real':'reuse_completed_TME_only_logistic_regression',
            'shuffled':'fit_same_pipeline_on_split_local_shuffled_rows',
            'logits':'symmetric_binary_decision_function', 'training_targets':'exact_few_shot_training_slides_only',
            'selection':'source_validation_error_first_C_tie',
            'complementarity':'source_validation_only_diagnostic_not_selection_gate'},
        'student':{'recipe':'inherit_exact_matched_contract',
            'auxiliary_head':'retained_unused_to_match_existing_controls', 'inference':'WSI_only_native_CONCH_ViLa'}}


def test_preflight_reuses_controls_and_identity_binds_new_conditions(tmp_path,monkeypatch):
    contract = minimal_contract(tmp_path)
    matched = {'folds':[0], 'results_root':str(tmp_path/'old')}
    old = {'source_sha256':{}, 'reference_tme_only_identity':'ref',
           'donor_maps':{'train':{},'val':{},'test':{}}}
    monkeypatch.setattr(runner,'load_yaml_config',lambda _:matched)
    monkeypatch.setattr(runner.legacy,'preflight',lambda c,f,m:({}, {}, old, 'id_'+m))
    monkeypatch.setattr(runner,'verify_completed',lambda *a:{'status':'completed'})
    monkeypatch.setattr(runner,'sha',lambda p:'hash')
    a = runner.preflight(contract,0,'kd_tme_only')
    assert a[2]['inference_requires_tme'] is False and 'test' not in a[2]['donor_maps']
    assert a[3] != runner.preflight(contract,0,'kd_tme_only_shuffled')[3]
    assert runner.preflight(contract,0,'student_ce')[3] == 'id_student_ce'
    assert str(runner.preflight(contract,0,'kd_visual')[4]).startswith(matched['results_root'])
    changed = copy.deepcopy(contract); changed['student']['recipe'] = 'extra_epochs'
    with pytest.raises(ValueError): runner.preflight(changed,0,'kd_tme_only')
    with pytest.raises(ValueError): runner.preflight(contract,5,'kd_tme_only')
    monkeypatch.setattr(runner,'verify_completed',lambda *a:None)
    with pytest.raises(ValueError): runner.preflight(contract,0,'kd_tme_only')


def test_plan_reuses_ten_controls_and_submits_only_ten_new_folds(tmp_path,monkeypatch):
    contract = {'folds':list(range(5)), 'matched_contract':'old', 'results_root':str(tmp_path/'new')}
    matched = {'external':{'contract':'target'}}
    monkeypatch.setenv('PGVL_CONDA_ENV','/env'); monkeypatch.setenv('PATHOTME_RESULTS_ROOT',str(tmp_path))
    monkeypatch.setattr(launcher,'load_yaml_config',lambda p:matched if p=='old' else contract)
    def check(c,f,m,s=False):
        return {}, {}, {}, f'{m}-{f}-{s}', tmp_path/m/f'{f}-{s}'
    monkeypatch.setattr(launcher,'preflight',check)
    monkeypatch.setattr(launcher,'verify_completed',lambda p,i,s=False: {} if i.startswith(NEW_CONDITIONS) else {'done':True})
    reused, plans = launcher.build_plan(tmp_path/'config.yaml')
    assert len(reused) == 10 and len(plans) == 11
    assert len(plans[0]['components']) == 2
    for p in plans[1:]:
        assert p['depends_on'] == ['smoke'] and p['time'] == '00:45:00'
        assert '--gres=gpu:1' in p['sbatch'] and 'HF_HUB_OFFLINE=1' in p['sbatch'][-1]
        assert p['components'][0]['condition'] in NEW_CONDITIONS


def test_normal_import_does_not_initialize_foundation_models():
    # The individual helper module is importable without any heavyweight model dependency.
    spec = importlib.util.spec_from_file_location('standalone_probe',ROOT/'pathotme/standalone_teacher.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.NEW_CONDITIONS == NEW_CONDITIONS
    probe = subprocess.run([sys.executable, '-c',
        f"import sys; sys.path[:0]=[{str(ROOT)!r},{str(ROOT/'scripts')!r},'/path/to/PGVL-Gym']; "
        "import run_standalone_tme_teacher, launch_standalone_tme_teacher; "
        "assert not {'torch','transformers','sklearn'} & set(sys.modules)"],
        capture_output=True, text=True, timeout=30)
    assert probe.returncode == 0, probe.stderr
