#!/usr/bin/env python3
"""One smoke or one four-arm fold from the immutable TCGA study contract."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym']
from pathotme.locked_tcga import ROOT, PGVL, PYTHON, atomic_json, identity, sha, rows, membership, load_launch, verify_completion
from common.configuration import load_dotenv, load_yaml_config


def selected(launch, cohort, method, fold):
    return next(p for p in launch['plans'] if (p['cohort'], p['method'], p['fold']) == (cohort, method, fold))


def check_features(plan, cfg):
    inventory = json.loads(Path(plan['feature_inventory']).read_text())
    for phase in ['train', 'val', 'test']:
        for row in rows(Path(cfg['split_dir']) / f"fold{plan['fold']}/{phase}.csv"):
            for field in ['feature_path_column_s', 'feature_path_column_l']:
                path = Path(os.path.expandvars(row[cfg[field]]))
                item = inventory[str(path)]; stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
                    raise ValueError(f'feature changed since header audit: {path}')


def checkpoint_files(plan):
    native = Path(plan['native_dir']); fold = plan['fold']
    return [native / 'config.json', native / 'metrics.json', native / f'fold{fold}_best.pt', native / f'fold{fold}_predictions.csv']


def validate_native(plan, cfg):
    from common.run_state import validate_resume_state
    native = Path(plan['native_dir'])
    state = json.loads((native / 'metrics.json').read_text())
    expected = (load_yaml_config(PGVL / f"benchmarks/tcga_{plan['cohort']}/configs/{plan['method']}/{plan['cohort']}_16shot.yaml")
                if plan['native_reused'] else cfg)
    valid = validate_resume_state(state, native / 'config.json', plan['method'], expected)
    record = next((r for r in valid if r['fold'] == plan['fold']), None)
    if record is None or any(record.get('sample_failures', {}).values()):
        raise ValueError('native fold missing or has sample failures')
    test = rows(Path(cfg['split_dir']) / f"fold{plan['fold']}/test.csv")
    prediction = rows(native / f"fold{plan['fold']}_predictions.csv")
    if sorted(membership(test)) != sorted(membership(prediction)):
        raise ValueError('native predictions/test identity mismatch')
    return {str(p): sha(p) for p in checkpoint_files(plan)}


def evaluate(loader, bridge, model, metric_fn):
    import numpy as np
    import pandas as pd
    import torch
    from run_vila_guided import _metric_bundle
    probabilities = []; labels = []; metadata = []; attention = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            details = bridge.eval_step_with_details(batch, model)
            probabilities.append(details['probabilities'][0].detach().cpu().numpy())
            labels.append(int(batch[-1].reshape(-1)[0])); metadata.append(details['metadata'])
            values = {}
            for scale in ['low', 'high']:
                scores = details[f'{scale}_tme_group_attention'].mean(dim=(0, 1)).cpu().tolist()
                names = getattr(model.conditioner.tokenizer, f'{scale.upper()}_TOKEN_NAMES')
                if len(scores) != len(names):
                    raise ValueError('attention token role mismatch')
                values.update({f'tme_attention_{scale}_{n}': v for n, v in zip(names, scores)})
            attention.append(values)
    probabilities = np.asarray(probabilities); labels = np.asarray(labels)
    if probabilities.shape != (len(labels), 2) or not np.isfinite(probabilities).all():
        raise ValueError('invalid test probabilities')
    frame = pd.DataFrame(metadata); frame['label'] = labels
    frame['prediction'] = probabilities.argmax(-1)
    frame[['probability_0', 'probability_1']] = probabilities
    frame = pd.concat([frame, pd.DataFrame(attention)], axis=1)
    return frame, _metric_bundle(probabilities, labels, metadata, metric_fn)


def arm_run(launch, plan, base_cfg, arm, smoke, device, native_hashes):
    import numpy as np
    import pandas as pd
    import torch
    from train import build_loaders, classification_metrics, set_seed
    from run_vila_guided import _atomic_torch, _atomic_csv, _run_epoch
    from pathotme.locked_models import make_method
    spec = launch['protocol']; fold = plan['fold']; method = plan['method']
    out = (Path(launch['output']) / f"smokes/{plan['cohort']}/{method}/{arm}"
           if smoke else Path(plan['output']) / arm)
    run_identity = identity({'launch': launch['identity'], 'plan': plan, 'arm': arm,
                             'smoke': smoke, 'native_sha256': native_hashes})
    if (out / 'metrics.json').exists():
        return verify_completion(out / 'metrics.json', run_identity)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**base_cfg, '_fold_index': fold, 'results_dir': str(out),
           'base_checkpoint_dir': plan['smoke_checkpoint_dir'] if smoke else plan['native_dir'],
           'tme_feature_csv': plan['tme_csv'], 'tme_panel': spec['panels'][plan['cohort']],
           'tme_mode': 'zero' if arm == 'zero' else 'actual',
           **{'tme_' + k: v for k, v in spec['adapter'].items() if k in ['hidden_dim', 'attention_heads', 'dropout', 'initial_gate']},
           'optimizer': 'adam', 'lr': spec['adapter']['lr'], 'weight_decay': spec['adapter']['weight_decay'],
           'lr_scheduler': None}
    atomic_json(out / 'config.json', {'identity': run_identity, 'launch_identity': launch['identity'],
                                    'config': cfg, 'arm': arm, 'native_sha256': native_hashes})
    maps = json.loads(Path(plan['donor_maps']).read_text())
    set_seed(spec['seed'] + fold)
    loaders = build_loaders(method, cfg, fold)
    bridge = make_method(cfg, plan['cohort'], method, arm, maps, device)
    model = bridge.build_model(); bridge.prepare_fold(fold, model, loaders[0])
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith('conditioner.') for n, _ in trainable):
        raise ValueError('only conditioner parameters may train')
    initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for _, p in trainable)).hexdigest()
    optimizer = bridge.build_optimizer(model)
    if smoke:
        batch = next(iter(loaders[0])); model.eval()
        inputs = bridge.inputs(batch)[0] if method == 'mgpath' else bridge._inputs(batch, model)[0]
        with torch.no_grad():
            native = model.base(*inputs[:4]) if method == 'mgpath' else model.base(*inputs[:5])[0]
            projection = model.conditioner.output_projection.weight.clone()
            model.conditioner.output_projection.weight.zero_()
            observed = model(*inputs) if method == 'mgpath' else model(*inputs)[0]
            torch.testing.assert_close(observed, native, rtol=1e-5, atol=1e-5)
            model.conditioner.output_projection.weight.copy_(projection)
            varied = (*inputs[:-1], torch.full_like(inputs[-1], 1e3))
            observed = model(*inputs) if method == 'mgpath' else model(*inputs)[0]
            other = model(*varied) if method == 'mgpath' else model(*varied)[0]
            if arm == 'zero':
                torch.testing.assert_close(observed, other)
            elif torch.allclose(observed, other, atol=1e-7, rtol=1e-7):
                raise ValueError('real conditioner does not affect outputs')
        model.train(); step = bridge.train_step(batch, model, optimizer, None)
        gradients = [p.grad for _, p in trainable if p.grad is not None]
        if (not np.isfinite(step['loss']) or not gradients or
                not all(torch.isfinite(g).all() for g in gradients) or
                not any(g.abs().sum() > 0 for g in gradients) or
                any(p.grad is not None for p in model.base.parameters())):
            raise ValueError('invalid adapter gradients or frozen-base violation')
        model.eval()
        with torch.no_grad():
            check = bridge.eval_step(batch, model)
            if not torch.isfinite(check['logits']).all():
                raise ValueError('nonfinite post-update logits')
        result = {'status': 'completed', 'identity': run_identity, 'initial_conditioner_sha256': initial,
                  'training_bag_only': True, 'zero_residual_native_equivalence': True,
                  'only_adapter_gradients': True, 'loss': step['loss'],
                  'trainable_parameters': sum(p.numel() for _, p in trainable),
                  'artifact_sha256': {str(out / 'config.json'): sha(out / 'config.json')}}
        atomic_json(out / 'metrics.json', result)
        return result
    selection = spec['selection'][method]
    best = float('inf'); best_epoch = -1; stale = 0; start = 0; history = []
    last_path = out / 'resume.pt'; best_path = out / 'best.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved['identity'] != run_identity:
            raise ValueError('resume checkpoint identity mismatch')
        model.load_adapter_state_dict(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        best, best_epoch, stale, history = saved['best'], saved['best_epoch'], saved['stale'], saved['history']
        start = saved['epoch'] + 1
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        if sha(best_path) != saved['best_sha256']:
            raise ValueError('best checkpoint changed during resume')
    for epoch in range(start, spec['adapter']['epochs']):
        if history and stale >= selection['patience'] and history[-1]['epoch'] > selection['min_epoch']:
            break
        before = time.monotonic()
        tr = _run_epoch(loaders[0], bridge, model, classification_metrics, optimizer)
        with torch.no_grad():
            val = _run_epoch(loaders[1], bridge, model, classification_metrics)
        monitor = (float(1 - val['metrics']['accuracy']) if method == 'vila_mil'
                   else -float(val['metrics']['macro_f1']))
        if not np.isfinite(monitor):
            raise ValueError('invalid validation monitor')
        improved = monitor <= best if selection['ties_improve'] else monitor < best
        if improved:
            best = monitor; best_epoch = epoch; stale = 0
            _atomic_torch(best_path, {'identity': run_identity, 'epoch': epoch, 'state': model.adapter_state_dict()})
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_loss': tr['loss'], 'val_loss': val['loss'],
                        'minimized_validation_monitor': monitor, 'improved': improved,
                        'wall_seconds': time.monotonic() - before})
        _atomic_torch(last_path, {'identity': run_identity, 'epoch': epoch, 'state': model.adapter_state_dict(),
            'optimizer': optimizer.state_dict(), 'best': best, 'best_epoch': best_epoch, 'stale': stale,
            'history': history, 'best_sha256': sha(best_path), 'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all()})
        _atomic_csv(out / 'training_history.csv', pd.DataFrame(history))
        print(json.dumps({'cohort': plan['cohort'], 'method': method, 'fold': fold, 'arm': arm, **history[-1]}), flush=True)
    saved = torch.load(best_path, map_location=device, weights_only=True)
    if saved['identity'] != run_identity:
        raise ValueError('best checkpoint identity mismatch')
    model.load_adapter_state_dict(saved['state'])
    prediction, metrics = evaluate(loaders[2], bridge, model, classification_metrics)
    expected = rows(Path(cfg['split_dir']) / f'fold{fold}/test.csv')
    if sorted(membership(prediction.to_dict('records'))) != sorted(membership(expected)):
        raise ValueError('adapter predictions/test identity mismatch')
    _atomic_csv(out / 'predictions.csv', prediction)
    artifacts = [best_path, last_path, out / 'predictions.csv', out / 'training_history.csv', out / 'config.json']
    result = {'status': 'completed', 'identity': run_identity, 'launch_identity': launch['identity'],
              'cohort': plan['cohort'], 'method': method, 'fold': fold, 'arm': arm,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'best_epoch': best_epoch,
              'validation_monitor': selection['monitor'], 'best_minimized_monitor': best,
              'initial_conditioner_sha256': initial, 'metrics': metrics,
              'trainable_parameters': sum(p.numel() for _, p in trainable),
              'artifact_sha256': {str(p): sha(p) for p in artifacts}}
    atomic_json(out / 'metrics.json', result)
    return result


def execute(args, launch, plan):
    import torch
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('execution requires an allocated GPU job')
    cfg = load_yaml_config(plan['config']); check_features(plan, cfg)
    smoke_root = Path(launch['output']) / f"smokes/{plan['cohort']}/{plan['method']}"
    report_path = smoke_root / 'smoke_report.json'
    if not args.smoke:
        smoke = json.loads(report_path.read_text())
        if smoke.get('status') != 'smoke_passed' or smoke.get('launch_identity') != launch['identity']:
            raise ValueError('matching successful smoke required')
        from pathotme.locked_tcga import verify_files
        verify_files(smoke['artifact_sha256'])
        if not plan['native_reused']:
            subprocess.run([PYTHON, '-u', str(PGVL / 'train.py'), '--method', plan['method'],
                            '--config', plan['config'], '--device', args.device], cwd=PGVL, check=True)
        native_hashes = validate_native(plan, cfg)
    else:
        fixture = Path(plan['smoke_checkpoint_dir']) / f"fold{plan['fold']}_best.pt"
        native_hashes = {str(fixture): sha(fixture)}
    results = {}
    for arm in ['zero', 'actual', 'shuffled']:
        results[arm] = arm_run(launch, plan, cfg, arm, args.smoke, args.device, native_hashes)
        gc.collect(); torch.cuda.empty_cache()
    if len({r['initial_conditioner_sha256'] for r in results.values()}) != 1:
        raise ValueError('adapter arms have different initial weights')
    if args.smoke:
        atomic_json(report_path, {'status': 'smoke_passed', 'launch_identity': launch['identity'],
            'cohort': plan['cohort'], 'method': plan['method'], 'arms': results,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'note': 'historical checkpoint is a structural fixture only; no test bags used',
            'artifact_sha256': {str(smoke_root / arm / 'metrics.json'): sha(smoke_root / arm / 'metrics.json') for arm in results}})
    else:
        atomic_json(Path(plan['output']) / 'fold_complete.json', {'status': 'completed',
            'launch_identity': launch['identity'], 'plan': plan, 'native_sha256': native_hashes,
            'arms': results, 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--cohort', choices=['nsclc', 'brca'], required=True)
    parser.add_argument('--method', choices=['vila_mil', 'mgpath'], required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); load_dotenv(PGVL / '.env')
    launch = load_launch(args.launch); plan = selected(launch, args.cohort, args.method, args.fold)
    if args.smoke and args.fold != 0:
        raise ValueError('one fold-zero training-only smoke per pair')
    if not args.execute:
        print(json.dumps({'launch_identity': launch['identity'], 'plan': plan, 'smoke': args.smoke}))
    else:
        lock_dir = Path(launch['output']) / 'locks'; lock_dir.mkdir(exist_ok=True)
        with (lock_dir / f"{args.cohort}_{args.method}_{'smoke' if args.smoke else args.fold}.lock").open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(args, launch, plan)
