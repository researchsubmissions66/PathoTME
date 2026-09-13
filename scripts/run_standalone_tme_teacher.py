#!/usr/bin/env python3
"""Two new WSI-only students; completed matched controls remain read-only."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import run_controlled_vila as legacy
from run_controlled_vila import ROOT, PGVL as PGVL, environment, load_yaml_config, sha, verify_completed
from pathotme.standalone_teacher import CONDITIONS, NEW_CONDITIONS


def validate_contract(contract):
    """Refuse silent changes to the four-condition attribution experiment."""
    expected = {
        'real': 'reuse_completed_TME_only_logistic_regression',
        'shuffled': 'fit_same_pipeline_on_split_local_shuffled_rows',
        'logits': 'symmetric_binary_decision_function',
        'training_targets': 'exact_few_shot_training_slides_only',
        'selection': 'source_validation_error_first_C_tie',
        'complementarity': 'source_validation_only_diagnostic_not_selection_gate',
    }
    if (tuple(contract['conditions']) != CONDITIONS
            or contract['reuse_conditions'] != list(CONDITIONS[:2])
            or contract['teacher'] != expected
            or contract['student'] != {'recipe': 'inherit_exact_matched_contract',
                'auxiliary_head': 'retained_unused_to_match_existing_controls',
                'inference': 'WSI_only_native_CONCH_ViLa'}
            or not contract['folds']
            or len(set(contract['folds'])) != len(contract['folds'])):
        raise ValueError('unsupported standalone-teacher contract')


def preflight(contract, fold, condition, smoke=False):
    """Bind new runs to validated old recipes, controls, teachers and code."""
    validate_contract(contract)
    matched = load_yaml_config(contract['matched_contract'])
    if fold not in contract['folds'] or fold not in matched['folds'] or condition not in CONDITIONS:
        raise ValueError('unregistered fold/condition')
    if Path(contract['results_root']).resolve() == Path(matched['results_root']).resolve():
        raise ValueError('new experiment must not overwrite old results')
    references = {}
    cfg = phases = old_payload = None
    for control in (*CONDITIONS[:2], 'fusion'):
        c, p, payload, identity = legacy.preflight(matched, fold, control)
        directory = legacy.output_dir(matched, fold, control)
        completed = verify_completed(directory, identity)
        if completed is None:
            raise ValueError(f'matched reference is not complete: {directory}')
        references[control] = {'identity': identity, 'output': str(directory),
                               'marker_sha256': sha(directory/'metrics.json')}
        if control == 'student_ce':
            cfg, phases, old_payload = c, p, payload
    if condition in CONDITIONS[:2]:
        if smoke:
            raise ValueError('do not repeat existing control smokes')
        return cfg, phases, old_payload, references[condition]['identity'], Path(references[condition]['output'])
    files = [Path(__file__), ROOT/'pathotme/standalone_teacher.py',
             ROOT/'scripts/run_controlled_vila.py', Path(contract['matched_contract'])]
    payload = {'contract': contract, 'matched_contract': matched, 'fold': fold,
        'condition': condition, 'smoke': smoke, 'matched_references': references,
        'matched_source_provenance': old_payload, 'inference_requires_tme': False,
        'source_sha256': {**old_payload['source_sha256'], **{str(p.resolve()): sha(p) for p in files}},
        'reference_tme_only_identity': old_payload['reference_tme_only_identity'],
        'donor_maps': {p: old_payload['donor_maps'][p] for p in ('train', 'val')},
        'teacher_training_scope': 'exact train slides; validation selects C only; no test TME targets',
        'test_history': 'Both TCGA and CPTAC benchmark outcomes were previously inspected.'}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    output = Path(contract['results_root'])
    if smoke:
        output /= 'smoke'
    return cfg, phases, payload, identity, output/condition/f'fold{fold}'


def prepare_teacher(payload, phases, condition, fold):
    """Reuse the true teacher or fit its shuffled-row control, without test data."""
    import joblib
    import numpy as np
    from pathotme.standalone_teacher import source_arrays, subtype_logits, complementarity
    from run_brca_tme_only import fit_classifier
    reference = load_yaml_config(payload['matched_contract']['reference_contract'])
    _, raw = legacy.read_csv(reference['selected_features_csv'])
    shuffled = condition == 'kd_tme_only_shuffled'
    arrays, labels = source_arrays({p: phases[p] for p in ('train', 'val')}, raw,
                                  reference['panel'], payload['donor_maps'] if shuffled else None)
    if shuffled:
        classifier, selected_c, val_error = fit_classifier(arrays['train'], labels['train'],
            arrays['val'], labels['val'], reference['tme_only']['c_grid'], reference['seed']+fold)
    else:
        directory = Path(reference['results_root'])/'tme_only'/f'fold{fold}'
        record = verify_completed(directory, payload['reference_tme_only_identity'])
        if record is None:
            raise ValueError('standalone TME teacher missing')
        saved = joblib.load(directory/f'fold{fold}_classifier.joblib')
        if saved['identity'] != payload['reference_tme_only_identity']:
            raise ValueError('standalone classifier identity mismatch')
        classifier = saved['pipeline']; selected_c = record['selected_c']; val_error = record['validation_error']
    logits = {p: subtype_logits(classifier, x) for p, x in arrays.items()}
    reference_path = Path(payload['matched_references']['fusion']['output'])/'validation_predictions.csv'
    _, rows = legacy.read_csv(reference_path)
    by_id = {r['slide_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {r['slide_id'] for r in phases['val']}:
        raise ValueError('visual validation coverage mismatch')
    visual = []; diagnostic_rows = []
    probabilities = classifier.predict_proba(arrays['val'])
    for i, row in enumerate(phases['val']):
        v = by_id[row['slide_id']]
        if v['case_id'] != row['case_id'] or int(v['label']) != int(row['label_id']):
            raise ValueError('visual validation patient/label mismatch')
        visual.append([float(v['probability_0']), float(v['probability_1'])])
        diagnostic_rows.append({'slide_id': row['slide_id'], 'case_id': row['case_id'],
            'label': int(row['label_id']), 'visual_probability_0': visual[-1][0],
            'visual_probability_1': visual[-1][1], 'teacher_probability_0': float(probabilities[i, 0]),
            'teacher_probability_1': float(probabilities[i, 1])})
    training_cache = {r['slide_id']: logits['train'][i] for i, r in enumerate(phases['train'])}
    if not np.isfinite(logits['train']).all():
        raise ValueError('nonfinite teacher cache')
    diagnostics = {'condition': condition, 'fold': fold, 'selected_c': selected_c,
        'teacher_validation_error': val_error, 'training_slides': len(training_cache),
        'training_target_ids': sorted(training_cache), 'teacher_reused': not shuffled,
        'complementarity': complementarity(visual, probabilities, labels['val']),
        'logit_representation': '[-decision_function/2, +decision_function/2]',
        'temperature_applied': False, 'target_test_tme_used': False}
    return training_cache, diagnostics, diagnostic_rows, classifier


def execute(cfg, phases, payload, identity, output, fold, condition, device, smoke=False):
    """Match old student initialization/optimizer/selection; replace only teacher."""
    import copy
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from train import build_loaders, classification_metrics, set_seed
    from methods.vila_mil.adapter import ViLaMILMethod
    from pathotme.privileged_vila import PrivilegedStudent, visual_inputs, distillation_loss
    from pathotme.guided_vila_adapter import _load_torch_state, _one_metadata_value
    from run_vila_guided import _atomic_json, _atomic_csv, _atomic_torch, _metric_bundle
    matched = payload['matched_contract']; training = matched['training']; student = matched['student']
    cfg = {**cfg, '_fold_index': fold, 'results_dir': str(output)}
    _atomic_json(output/'config.json', {'identity': identity, 'provenance': payload, 'resolved_config': cfg})
    seed = matched['seed']+fold; set_seed(seed)
    loaders = build_loaders('vila_mil', cfg, fold)
    base = ViLaMILMethod(cfg, device).build_model()
    base.load_state_dict(_load_torch_state(Path(matched['base_checkpoint_dir'])/f'fold{fold}_best.pt', device), strict=True)
    base.eval()
    model = PrivilegedStudent(base, 64, student['auxiliary_hidden_dim']).to(device)
    artifacts = [output/'config.json']; cache = {}; diagnostics = None
    if condition in NEW_CONDITIONS:
        values, diagnostics, rows, classifier = prepare_teacher(payload, phases, condition, fold)
        cache = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in values.items()}
        _atomic_json(output/'teacher_diagnostics.json', diagnostics)
        _atomic_csv(output/'teacher_validation_predictions.csv', pd.DataFrame(rows))
        checkpoint = output/'teacher_classifier.joblib'
        joblib.dump({'identity': identity, 'pipeline': classifier}, checkpoint)
        artifacts.extend([checkpoint, output/'teacher_diagnostics.json', output/'teacher_validation_predictions.csv'])
    elif condition == 'kd_visual':  # Used by synthetic equivalence tests, not production launches.
        teacher = copy.deepcopy(base).eval()
        with torch.no_grad():
            for batch in loaders[0]:
                x = visual_inputs(batch, teacher, device)
                key = _one_metadata_value(batch[2], 'slide_id')
                cache[key] = teacher(*x, return_details=True)['logits'].detach().cpu()
        del teacher
    elif condition != 'student_ce':
        raise ValueError('unknown student condition')
    if condition != 'student_ce':
        if set(cache) != {r['slide_id'] for r in phases['train']}:
            raise ValueError('distillation cache must contain exactly training slides')
        path = output/'training_teacher_logits.pt'; _atomic_torch(path, cache); artifacts.append(path)
    set_seed(seed)  # Same training-order reset as the immutable control runner.
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=training['lr'], weight_decay=training['weight_decay'])

    def step(batch, train=False):
        details = model(*visual_inputs(batch, model.base, device))
        loss = details['loss']; kd = loss.new_zeros(())
        if train and condition != 'student_ce':
            key = _one_metadata_value(batch[2], 'slide_id')
            kd = distillation_loss(details['logits'], cache[key].to(device), student['temperature'])
            loss = loss + student['distillation_weight']*kd
        if not torch.isfinite(loss) or not torch.isfinite(details['logits']).all():
            raise RuntimeError('nonfinite student loss/logits')
        if train:
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError('nonfinite gradient')
            if any(p.grad is not None for p in model.auxiliary.parameters()):
                raise RuntimeError('unused auxiliary head received gradients')
            optimizer.step()
        return details, float(loss.detach()), float(kd.detach())

    def evaluate(loader):
        model.eval(); ps = []; ys = []; metas = []
        with torch.no_grad():
            for batch in loader:
                details, _, _ = step(batch)
                ps.append(details['probabilities'][0].cpu().numpy()); ys.append(int(batch[-1].item()))
                metas.append({k: _one_metadata_value(batch[2], k) for k in ('slide_id', 'case_id')})
        probabilities = np.asarray(ps)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1):
            raise RuntimeError('invalid student probabilities')
        return probabilities, np.asarray(ys), metas

    def frame(p, y, meta):
        data = pd.DataFrame(meta); data['label'] = y; data['prediction'] = p.argmax(1)
        data[['probability_0', 'probability_1']] = p
        return data

    if smoke:
        batch = next(iter(loaders[0])); model.eval()
        with torch.no_grad():
            x = visual_inputs(batch, base, device)
            torch.testing.assert_close(model(*x)['probabilities'], base(*x)[0])
        model.train(); _, loss, kd = step(batch, True)
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base.parameters()):
            raise RuntimeError('no student gradient')
        report = {'loss': loss, 'distillation_loss': kd, 'native_forward_equivalent': True}
    else:
        best = float('inf'); best_epoch = None; stale = 0; history = []
        best_path = output/f'fold{fold}_best.pt'
        for epoch in range(training['epochs']):
            model.train(); losses = []; kd_losses = []
            for batch in loaders[0]:
                _, loss, kd = step(batch, True); losses.append(loss); kd_losses.append(kd)
            vp, vy, vm = evaluate(loaders[1]); error = float((vp.argmax(1) != vy).mean())
            if error < best:
                best = error; best_epoch = epoch; stale = 0
                _atomic_torch(best_path, {'identity': identity, 'epoch': epoch, 'state_dict': model.state_dict()})
                _atomic_csv(output/'validation_predictions.csv', frame(vp, vy, vm))
            else:
                stale += 1
            history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                            'distillation_loss': float(np.mean(kd_losses)), 'val_error': error})
            _atomic_csv(output/'training_history.csv', pd.DataFrame(history))
            print(f'{condition} fold={fold} epoch={epoch} val_error={error:.4f}', flush=True)
            if stale >= training['early_stopping_patience'] and epoch > training['early_stopping_min_epoch']:
                break
        if best_epoch is None:
            raise RuntimeError('no validation-selected checkpoint')
        model.load_state_dict(_load_torch_state(best_path, device)['state_dict'], strict=True)
        p, y, meta = evaluate(loaders[2])
        prediction = output/f'fold{fold}_predictions.csv'; _atomic_csv(prediction, frame(p, y, meta))
        export = output/f'fold{fold}_inference.pt'
        _atomic_torch(export, {'identity': identity, 'state_dict': model.base.state_dict(),
            'input_contract': 'native_CONCH_ViLa_WSI_only_no_TME', 'config': load_yaml_config(matched['base_config'])})
        artifacts.extend([best_path, prediction, export, output/'validation_predictions.csv', output/'training_history.csv'])
        report = {'best_epoch': best_epoch, 'validation_error': best,
                  'metrics': _metric_bundle(p, y, meta, classification_metrics)}
    report.update(status='smoke_passed' if smoke else 'completed', identity=identity, provenance=payload,
        fold=fold, condition=condition, seed=seed, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        trainable_parameters=sum(n for _, n in trainable), trainable_parameter_names=[n for n, _ in trainable],
        teacher_diagnostics=diagnostics, artifact_sha256={p.name: sha(p) for p in artifacts})
    _atomic_json(output/('smoke_report.json' if smoke else 'metrics.json'), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--condition', choices=NEW_CONDITIONS, required=True)
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true'); parser.add_argument('--expected-identity')
    parser.add_argument('--teacher-diagnostics', type=Path, help='CPU-only diagnostic JSON output; no student training')
    args = parser.parse_args(); environment(); contract = load_yaml_config(args.config)
    cfg, phases, payload, identity, output = preflight(contract, args.fold, args.condition, args.smoke_only)
    if args.expected_identity and identity != args.expected_identity:
        raise ValueError('queued scientific identity changed')
    if args.check_only:
        print(json.dumps({'status': 'ready', 'identity': identity, 'output': str(output)})); return
    if args.teacher_diagnostics:
        if args.smoke_only:
            parser.error('diagnostics and smoke are separate modes')
        _, diagnostic, rows, _ = prepare_teacher(payload, phases, args.condition, args.fold)
        with args.teacher_diagnostics.open('x') as handle:
            json.dump({'identity': identity, 'diagnostic': diagnostic, 'validation_predictions': rows}, handle, indent=2)
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if verify_completed(output, identity, args.smoke_only):
            print('already_completed'); return
        if any(p.name != '.run.lock' for p in output.iterdir()):
            raise ValueError('partial output preserved; explicit recovery required')
        execute(cfg, phases, payload, identity, output, args.fold, args.condition, args.device, args.smoke_only)


if __name__ == '__main__':
    main()
