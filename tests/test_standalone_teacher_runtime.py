"""Synthetic CPU execution and numerical equivalence to the completed controls."""
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), '/path/to/PGVL-Gym']
import run_standalone_tme_teacher as runner
import run_controlled_vila as legacy
from common.configuration import load_yaml_config
from methods.vila_mil.adapter import ViLaMILMethod
from methods.vila_mil.model import ViLa_MIL_Model
from pathotme.controlled_vila import donor_maps
from pathotme.brca_features import panel_spec, transform_rows
from run_brca_tme_only import fit_classifier


def test_four_conditions_match_controls_export_wsi_only_and_never_cache_test(tmp_path,monkeypatch):
    import train
    runner.environment()
    cfg = load_yaml_config('/path/to/PGVL-Gym/benchmarks/tcga_brca/configs/vila_mil_conch/brca_16shot.yaml')
    matched = load_yaml_config(ROOT/'configs/vila_conch_brca_controlled_16shot.yaml')
    matched['training']['epochs'] = 1; matched['base_checkpoint_dir'] = str(tmp_path/'baseline')
    base_dir = Path(matched['base_checkpoint_dir']); base_dir.mkdir()
    torch.manual_seed(7)
    base = ViLa_MIL_Model(SimpleNamespace(input_size=16,hidden_size=8,prototype_number=3),
                         num_classes=2,prompt_features=torch.randn(4,16))
    torch.save(base.state_dict(),base_dir/'fold0_best.pt')
    monkeypatch.setattr(ViLaMILMethod,'build_model',lambda self:copy.deepcopy(base).to(self.device))
    phases = {p:[{'slide_id':f'{p}{i}','case_id':f'{p}{i}','label_id':i} for i in (0,1)]
              for p in ('train','val','test')}
    loaders = [[(torch.randn(1,4,16),torch.randn(1,6,16),
        {'slide_id':[r['slide_id']],'case_id':[r['case_id']]},torch.tensor([r['label_id']])) for r in rows]
        for rows in phases.values()]
    monkeypatch.setattr(train,'build_loaders',lambda *a:loaders)
    names = panel_spec(matched['panel'])['feature_names']
    raw = [{'slide_id':r['slide_id'], **{n:str(r['label_id']+1) for n in names}}
           for split in phases.values() for r in split]
    feature = tmp_path/'tme.csv'
    with feature.open('w') as handle:
        w = csv.DictWriter(handle,fieldnames=['slide_id',*names]); w.writeheader(); w.writerows(raw)
    reference = {'selected_features_csv':str(feature),'results_root':str(tmp_path/'reference'),
                 'panel':matched['panel'],'seed':1,'tme_only':{'c_grid':[.1,1.]}}
    monkeypatch.setattr(runner,'load_yaml_config',lambda path:reference if str(path)==str(matched['reference_contract']) else cfg)
    monkeypatch.setattr(legacy,'load_yaml_config',runner.load_yaml_config)
    directory = Path(reference['results_root'])/'tme_only/fold0'; directory.mkdir(parents=True)
    values = np.asarray(transform_rows(raw,matched['panel']))
    classifier,c,error = fit_classifier(values[:2],np.array([0,1]),values[2:4],np.array([0,1]),[.1,1.],1)
    path = directory/'fold0_classifier.joblib'; joblib.dump({'identity':'ref','pipeline':classifier},path)
    (directory/'metrics.json').write_text(json.dumps({'status':'completed','identity':'ref','selected_c':c,
        'validation_error':error,'artifact_sha256':{path.name:runner.sha(path)}}))
    fusion = tmp_path/'fusion'; fusion.mkdir()
    pd.DataFrame([{'slide_id':r['slide_id'],'case_id':r['case_id'],'label':r['label_id'],
                   'probability_0':.6,'probability_1':.4} for r in phases['val']]).to_csv(fusion/'validation_predictions.csv',index=False)
    payload = {'matched_contract':matched,'reference_tme_only_identity':'ref','donor_maps':donor_maps(phases,1,0),
               'matched_references':{'fusion':{'output':str(fusion)}},'inference_requires_tme':False}
    # Compare the two control branches numerically, not just by claimed config equality.
    for condition in ('student_ce','kd_visual'):
        old_dir = tmp_path/('old_'+condition); old_dir.mkdir()
        old_payload = {'donor_maps':payload['donor_maps'],'reference_tme_only_identity':'ref'}
        legacy.execute(matched,cfg,phases,old_payload,'old',old_dir,0,condition,'cpu',False)
        new_dir = tmp_path/condition; new_dir.mkdir()
        runner.execute(cfg,phases,payload,'new',new_dir,0,condition,'cpu',False)
        a = torch.load(old_dir/'fold0_inference.pt',weights_only=True)['state_dict']
        b = torch.load(new_dir/'fold0_inference.pt',weights_only=True)['state_dict']
        for key in a:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    for condition in runner.NEW_CONDITIONS:
        output = tmp_path/condition; output.mkdir()
        runner.execute(cfg,phases,payload,condition,output,0,condition,'cpu',False)
        record = runner.verify_completed(output,condition)
        assert record['status'] == 'completed'
        cache = torch.load(output/'training_teacher_logits.pt',weights_only=True)
        assert set(cache) == {'train0','train1'}
        export = torch.load(output/'fold0_inference.pt',weights_only=True)
        assert not any(any(s in k for s in ('auxiliary','conditioner','standardizer')) for k in export['state_dict'])
        clone = copy.deepcopy(base); clone.load_state_dict(export['state_dict'],strict=True)
        diagnostic = json.loads((output/'teacher_diagnostics.json').read_text())
        assert diagnostic['target_test_tme_used'] is False
        assert diagnostic['teacher_reused'] == (condition=='kd_tme_only')
        assert diagnostic['complementarity']['slides'] == 2
