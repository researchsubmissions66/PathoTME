"""Matched TME-only and probability-fusion comparators; no Torch dependency."""
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from pathotme.locked_tcga import atomic_json, identity, sha, verify_files
from pathotme.fusion import binary_metrics

POLICY = {
    'version': 'matched_cpu_baselines_v1',
    'c_grid': [0.001, 0.01, 0.1, 1.0, 10.0],
    'alpha_grid': [0.0, 0.25, 0.5, 0.75, 1.0],
    'selection': 'patient_AUROC_then_patient_NLL_then_smaller_C_or_alpha',
    'fit': 'training_slides_only_L2_logistic_no_class_weight',
    'preprocessing': 'fixed_panel_transforms_train_median_mean_population_std_no_indicators',
    'fusion': '(1-alpha)*native_probability + alpha*TME_probability',
    'patient_aggregation': 'arithmetic_mean_of_all_held_out_slide_probabilities',
    'no_train_val_refit': True,
    'exploratory': True,
}


def load_json(path):
    return json.loads(Path(path).read_text())


def csv_write(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.csv')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def split_frame(path, labels):
    frame = pd.read_csv(path, dtype={'slide_id': str, 'case_id': str})
    source = frame['label_id'] if 'label_id' in frame else frame['label'].map(lambda x: labels.get(x, x))
    frame = frame[['slide_id', 'case_id']].assign(label=source.astype(int))
    if frame.isna().any().any() or frame.slide_id.duplicated().any() or set(frame.label) != {0, 1}:
        raise ValueError('Invalid, duplicate or nonbinary split')
    if (frame.groupby('case_id').label.nunique() != 1).any():
        raise ValueError('Conflicting labels within a patient')
    return frame.sort_values('slide_id').reset_index(drop=True)


def read_splits(task):
    frames = {phase: split_frame(path, task['label_dict']) for phase, path in task['splits'].items()}
    for i, phase in enumerate(frames):
        for other in list(frames)[i+1:]:
            if set(frames[phase].case_id) & set(frames[other].case_id):
                raise ValueError('Patient leakage across splits')
    return frames


def checked_predictions(path, expected):
    frame = pd.read_csv(path, dtype={'slide_id': str, 'case_id': str})
    if 'label' not in frame and 'label_id' in frame:
        frame = frame.rename(columns={'label_id': 'label'})
    if frame.slide_id.duplicated().any():
        raise ValueError('Duplicate prediction slide')
    frame = frame.sort_values('slide_id').reset_index(drop=True)
    if len(frame) != len(expected) or not frame[['slide_id', 'case_id', 'label']].equals(expected):
        raise ValueError('Predictions do not cover the exact labeled split')
    probabilities = frame[['probability_0', 'probability_1']].to_numpy(float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-6)):
        raise ValueError('Invalid saved probabilities')
    return frame


def metrics(frame, probability):
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(frame),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Invalid positive-class probabilities')
    def score(y, p):
        result = binary_metrics(y, p)
        result['brier'] = float(np.mean((p-y)**2))
        return result
    patient = frame[['case_id', 'label']].assign(p=probability).groupby('case_id', sort=True).agg(label=('label', 'first'), p=('p', 'mean'))
    return {'slide_metrics': score(frame.label.to_numpy(), probability),
            'patient_metrics': score(patient.label.to_numpy(), patient.p.to_numpy()),
            'slides': len(frame), 'patients': len(patient)}


def selection_key(score, complexity):
    patient = score['patient_metrics']
    return (-patient['auroc_ovr'], patient['nll'], complexity)


def fit_scaler(train):
    train = np.asarray(train, dtype=np.float64)
    if np.isinf(train).any() or np.isnan(train).all(0).any():
        raise ValueError('Infinite or all-missing training feature')
    median = np.nanmedian(train, axis=0)
    imputed = np.where(np.isnan(train), median, train)
    scale = imputed.std(0, ddof=0)
    scale[scale < 1e-8] = 1
    return {'median': median.tolist(), 'mean': imputed.mean(0).tolist(), 'scale': scale.tolist()}


def scale_values(values, state):
    values = np.asarray(values, dtype=np.float64)
    result = (np.where(np.isnan(values), state['median'], values)-np.asarray(state['mean']))/state['scale']
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite standardized feature')
    return result


def feature_matrix(task, frame):
    from pathotme.brca_features import transform_rows as breast_transform
    from pathotme.shared_panel import transform_rows as shared_transform
    table = pd.read_csv(task['tme_csv'], dtype={'slide_id': str}).set_index('slide_id', verify_integrity=True)
    if set(frame.slide_id)-set(table.index):
        raise ValueError('Missing TME slide; no inner-join dropping is allowed')
    values = table.loc[frame.slide_id].reset_index().to_dict('records')
    transform = (lambda rows: breast_transform(rows, task['panel'])) if task['panel'].startswith('brca_') else shared_transform
    result = np.asarray(transform(values), dtype=np.float64)
    if result.shape != (len(frame), len(task['feature_names'])):
        raise ValueError('Panel dimensions differ')
    return result


def complete(path, expected_identity):
    if not Path(path).exists():
        return None
    saved = load_json(path)
    if saved.get('status') != 'completed' or saved.get('identity') != expected_identity:
        raise ValueError('Existing baseline output has a different identity')
    verify_files(saved['artifact_sha256'])
    return saved


def run_tme(task, root):
    out = Path(root)/'tme_only'/task['id']
    verify_files(task['input_sha256'])
    existing = complete(out/'metrics.json', task['id'])
    if existing:
        return existing
    started = time.monotonic()
    frames = read_splits(task)
    x_train, x_val = (feature_matrix(task, frames[p]) for p in ['train', 'val'])
    scaler = fit_scaler(x_train)
    train, val = scale_values(x_train, scaler), scale_values(x_val, scaler)
    candidates, fitted = [], {}
    for c in POLICY['c_grid']:
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=5000, tol=1e-8, random_state=task['seed'])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train, frames['train'].label.to_numpy())
        if list(model.classes_) != [0, 1]:
            raise ValueError('Wrong classifier class order')
        score = metrics(frames['val'], model.predict_proba(val)[:, 1])
        candidates.append({'C': c, 'metrics': score})
        fitted[c] = model
    chosen = min(candidates, key=lambda x: selection_key(x['metrics'], x['C']))
    model = fitted[chosen['C']]
    out.mkdir(parents=True, exist_ok=True)
    # Freeze selection before transforming, predicting or scoring any test rows.
    atomic_json(out/'selection.json', {'C': chosen['C'], 'candidates': candidates, 'policy': POLICY})
    atomic_json(out/'model.json', {'C': chosen['C'], 'classes': [0, 1], 'scaler': scaler,
                'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
                'feature_names': task['feature_names'], 'precision': 'float64_CPU', 'task': task})
    result = {'status': 'completed', 'identity': task['id'], 'condition': 'tme_only',
              'cohort': task['cohort'], 'fold': task['fold'], 'panel': task['panel'],
              'selection': chosen, 'policy': POLICY, 'metrics': {}, 'device': 'cpu'}
    for phase in ['val', 'test']:
        values = val if phase == 'val' else scale_values(feature_matrix(task, frames[phase]), scaler)
        p = model.predict_proba(values)[:, 1]
        csv_write(out/f'{phase}_predictions.csv', frames[phase].assign(probability_0=1-p, probability_1=p))
        result['metrics'][phase] = metrics(frames[phase], p)
    result['wall_seconds'] = time.monotonic()-started
    result['artifact_sha256'] = {str(out/name): sha(out/name) for name in ['selection.json', 'model.json', 'val_predictions.csv', 'test_predictions.csv']}
    atomic_json(out/'metrics.json', result)
    return result


def fuse(visual, tme, alpha):
    visual, tme = np.asarray(visual), np.asarray(tme)
    if visual.shape != tme.shape or not 0 <= alpha <= 1:
        raise ValueError('Fusion shape or weight is invalid')
    return (1-alpha)*visual + alpha*tme


def select_alpha(frame, visual, tme):
    candidates = [{'alpha': a, 'metrics': metrics(frame, fuse(visual, tme, a))} for a in POLICY['alpha_grid']]
    selected = min(candidates, key=lambda x: selection_key(x['metrics'], x['alpha']))
    return selected, candidates


def run_fusion(task, tme_task, root):
    from native_export import native_inputs, export_validation
    native = native_inputs(task)
    out = Path(root)/'fusion'/task['name']
    run_id = identity({'task': task['id'], 'native': native, 'policy': POLICY})
    existing = complete(out/'metrics.json', run_id)
    if existing:
        return existing
    started = time.monotonic()
    tme = run_tme(tme_task, root)
    frames = read_splits(tme_task)
    out.mkdir(parents=True, exist_ok=True)
    val_path = export_validation(task, out, native)
    visual_val = checked_predictions(val_path, frames['val'])
    tme_root = Path(root)/'tme_only'/tme_task['id']
    tme_val = checked_predictions(tme_root/'val_predictions.csv', frames['val'])
    chosen, candidates = select_alpha(frames['val'], visual_val.probability_1, tme_val.probability_1)
    atomic_json(out/'selection.json', {'selected': chosen, 'candidates': candidates, 'policy': POLICY,
                'tme_metrics_sha256': sha(tme_root/'metrics.json'), 'native': native})
    # Test values enter only after alpha and the standalone TME C are frozen.
    native_test = Path(task['plan']['native_dir'])/f"fold{task['fold']}_predictions.csv"
    visual_test = checked_predictions(native_test, frames['test'])
    tme_test = checked_predictions(tme_root/'test_predictions.csv', frames['test'])
    result = {'status': 'completed', 'identity': run_id, 'task_id': task['id'],
              'cohort': task['cohort'], 'method': task['method'], 'encoder': task['encoder'],
              'fold': task['fold'], 'panel': tme_task['panel'], 'policy': POLICY,
              'selected_alpha': chosen['alpha'], 'selected_C': tme['selection']['C'],
              'native': native, 'tme_id': tme_task['id'], 'conditions': {}, 'device': 'cpu'}
    prediction = frames['test'].copy()
    for condition, p in [('native', visual_test.probability_1), ('tme_only', tme_test.probability_1),
            ('late_fusion_fixed_half', fuse(visual_test.probability_1, tme_test.probability_1, 0.5)),
            ('late_fusion_validation', fuse(visual_test.probability_1, tme_test.probability_1, chosen['alpha']))]:
        prediction[condition+'_probability_1'] = np.asarray(p)
        result['conditions'][condition] = metrics(frames['test'], p)
    csv_write(out/'predictions.csv', prediction)
    verify_files(native)
    result['wall_seconds'] = time.monotonic()-started
    result['artifact_sha256'] = {str(out/name): sha(out/name) for name in ['selection.json', 'predictions.csv', 'native_validation.csv', 'native_validation.json']}
    result['artifact_sha256'][str(tme_root/'metrics.json')] = sha(tme_root/'metrics.json')
    atomic_json(out/'metrics.json', result)
    return result
