"""Read-only CPU reconstruction of the exact seven native PathoTME methods."""
from pathlib import Path
import importlib
import json
import sys

from pathotme.locked_tcga import atomic_json, sha, verify_files


class NativePending(Exception):
    pass


def expected_native_config(task):
    plan = task['plan']
    if not plan['native_reused']:
        return plan['config']
    # The original locked ViLa/MGPATH runner used this explicit PGVL source
    # convention before later launch plans added source_base_config.
    source = plan.get('source_base_config')
    if source is None:
        if task['method'] not in ('vila_mil', 'mgpath') or task['cohort'] != 'brca':
            raise ValueError('Missing native reuse recipe')
        pgvl = Path(__file__).resolve().parents[2].parent/'PGVL-Gym'
        source = str(pgvl/f"benchmarks/tcga_{task['cohort']}/configs/{task['method']}/{task['cohort']}_16shot.yaml")
    return source


def native_inputs(task):
    from common.configuration import load_yaml_config
    from common.run_state import validate_resume_state
    plan, fold = task['plan'], task['fold']
    root = Path(plan['native_dir'])
    files = [root/'config.json', root/'metrics.json', root/f'fold{fold}_best.pt', root/f'fold{fold}_predictions.csv']
    if not all(p.is_file() for p in files):
        raise NativePending('Waiting for the matching completed native checkpoint and predictions')
    source = expected_native_config(task)
    launch = json.loads(Path(task['launch']).read_text())
    if launch['identity'] != task['launch_identity']:
        raise ValueError('Parent launch identity changed')
    expected = launch['file_sha256'].get(source)
    if expected is None or sha(source) != expected:
        raise ValueError('Native source recipe is not bound to its original launch')
    prompt_bindings = {p: digest for p, digest in launch['file_sha256'].items()
                       if '/text_prompts/' in p or '/text_assets/' in p}
    verify_files(prompt_bindings)
    cfg = load_yaml_config(source)
    state = json.loads((root/'metrics.json').read_text())
    records = validate_resume_state(state, root/'config.json', task['method'], cfg)
    found = next((r for r in records if r['fold'] == fold), None)
    if found is None:
        raise NativePending('Native fold has no completed metrics record')
    if any(found.get('sample_failures', {}).values()):
        raise ValueError('Native fold has sample failures')
    # Do not reuse a checkpoint from a one-bag smoke fixture.
    if (root/'fixture.json').exists() or 'native_fixture' in root.parts:
        raise ValueError('A smoke fixture is not a research baseline')
    return {str(p): sha(p) for p in files}


def native_class(task):
    method, encoder = task['method'], task['encoder']
    if task['cohort'] in ('crc', 'blca'):
        path = Path(__file__).resolve().parents[1]/'crc_blca_core62_20260914'
        sys.path.insert(0, str(path))
        from runtime_models import native_type
        return native_type(method, encoder)
    if method == 'vila_mil':
        from methods.vila_mil.adapter import ViLaMILMethod
        return ViLaMILMethod
    if method == 'mgpath':
        if encoder == 'clip-rn50':
            from pathotme.cross_encoder_models import CLIPMGPathMethod
            return CLIPMGPathMethod
        from methods.mgpath.adapter import MGPathMethod
        return MGPathMethod
    module, name = {
        'focus': ('focus_tme', 'FocusNativeMethod'),
        'muse': ('muse_tme', 'MUSENativeMethod'),
        'hive_mil': ('hive_tme', 'HiVENativeMethod'),
        'dyko': ('dyko_tme', 'DyKoNativeMethod'),
        'mscpt': ('mscpt_tme', 'MSCPTNativeMethod'),
    }[method]
    return getattr(importlib.import_module('pathotme.'+module), name)


def export_validation(task, output, bindings):
    from baseline_core import checked_predictions, split_frame, csv_write
    output = Path(output)
    metadata_path, path = output/'native_validation.json', output/'native_validation.csv'
    if metadata_path.exists():
        cached = json.loads(metadata_path.read_text())
        if cached['native_sha256'] != bindings or cached['task_id'] != task['id'] or sha(path) != cached['csv_sha256']:
            raise ValueError('Native validation cache binding changed')
        return path
    import torch
    import numpy as np
    import pandas as pd
    from train import build_loaders, set_seed, _batch_metadata
    from common.configuration import load_yaml_config
    cfg = load_yaml_config(task['plan']['config'])
    set_seed(cfg.get('seed', 1)+task['fold'])
    bridge = native_class(task)(cfg, 'cpu')
    model = bridge.build_model()
    checkpoint = Path(task['plan']['native_dir'])/f"fold{task['fold']}_best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    bridge.on_checkpoint_loaded(model, 'best', task['fold'])
    model.eval()
    # Only loader concurrency changes; the constructor sees the exact native recipe.
    loaders = build_loaders(task['method'], {**cfg, 'num_workers': 0, 'pin_memory': False}, task['fold'])
    predictions = []
    with torch.no_grad():
        for batch in loaders[1]:
            result = bridge.eval_step(batch, model)
            logits = result['logits'].detach().float().cpu()
            labels = result['label'].detach().cpu().reshape(-1).numpy()
            if logits.shape != (len(labels), 2) or not torch.isfinite(logits).all():
                raise ValueError('Invalid native validation logits')
            probs = logits.softmax(-1).numpy()
            meta = _batch_metadata(batch)
            if meta is None or len(meta['slide_id']) != len(labels):
                raise ValueError('Missing exact native slide metadata')
            for i in range(len(labels)):
                predictions.append({'slide_id': str(meta['slide_id'][i]), 'case_id': str(meta['case_id'][i]),
                    'label': int(labels[i]), 'probability_0': float(probs[i, 0]), 'probability_1': float(probs[i, 1])})
    frame = pd.DataFrame(predictions)
    # Checkpoint compatibility check on one saved test prediction. This changes no
    # recipe, parameter or validation selection and does not optimize against test.
    batch = next(iter(loaders[2]))
    with torch.no_grad():
        p = bridge.eval_step(batch, model)['logits'].detach().float().cpu().softmax(-1).numpy()
    test = pd.read_csv(Path(task['plan']['native_dir'])/f"fold{task['fold']}_predictions.csv").set_index('slide_id')
    ids = _batch_metadata(batch)['slide_id']
    reference = test.loc[ids, ['probability_0', 'probability_1']].to_numpy(float)
    if not np.allclose(p, reference, atol=1e-4, rtol=1e-4):
        raise ValueError(f'CPU native replay differs from saved checkpoint predictions: {float(np.max(np.abs(p-reference)))}')
    verify_files(bindings)
    csv_write(path, frame)
    expected = split_frame(task['splits']['val'], task['label_dict'])
    checked_predictions(path, expected)
    atomic_json(metadata_path, {'task_id': task['id'], 'native_sha256': bindings,
        'csv_sha256': sha(path), 'device': 'cpu', 'checkpoint_load': 'strict',
        'replayed_test_slide_ids': ids, 'replay_max_probability_difference': float(np.max(np.abs(p-reference))),
        'validation_slide_count': len(frame)})
    return path
