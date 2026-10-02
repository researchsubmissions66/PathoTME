#!/usr/bin/env python3
"""Run one matched MSCPT fold or one training-bag-only smoke."""
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
sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym']
from pathotme.locked_tcga import ROOT,PGVL,PYTHON,atomic_json,identity,sha,rows,membership,load_launch,verify_completion,verify_files
from common.configuration import load_dotenv,load_yaml_config
from pathotme.mscpt_runtime import evaluate, check_features, selected

def arm_run(launch, plan, base_cfg, arm, smoke, device, native_hashes):
    import numpy as np
    import pandas as pd
    import torch
    from train import build_loaders, classification_metrics, set_seed
    from run_vila_guided import _atomic_torch, _atomic_csv, _run_epoch
    from pathotme.mscpt_tme import MSCPTTMEMethod
    verify_files(native_hashes)
    spec = launch['protocol']; fold = plan['fold']; method = plan['method']
    out = (Path(launch['output']) / f"smokes/{plan['cohort']}/{plan['encoder']}/{arm}"
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
    bridge = MSCPTTMEMethod(cfg, maps, arm, device)
    model = bridge.build_model(); bridge.prepare_fold(fold, model, loaders[0])
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith('conditioner.') for n, _ in trainable):
        raise ValueError('only conditioner parameters may train')
    initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for _, p in trainable)).hexdigest()
    optimizer = bridge.build_optimizer(model)
    if smoke:
        from pathotme.mscpt_smoke_checks import check_model
        batch = next(iter(loaders[0])); model.eval()
        inputs = bridge.inputs(batch, model)[0]
        diagnostic = check_model(model, inputs, arm)
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
                  'only_adapter_gradients': True, 'label_independent_predictions': True, 'loss': step['loss'], 'diagnostic': diagnostic,
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
        monitor = bridge.validation_monitor({'val_macro_f1':val['metrics']['macro_f1'], 'val_auroc_ovr':val['metrics']['auroc_ovr'], 'val_accuracy':val['metrics']['accuracy']})[1]
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
        print(json.dumps({'cohort': plan['cohort'], 'method': method, 'encoder': plan['encoder'], 'fold': fold, 'arm': arm, **history[-1]}), flush=True)
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
              'cohort': plan['cohort'], 'method': method, 'encoder': plan['encoder'], 'fold': fold, 'arm': arm,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'best_epoch': best_epoch,
              'validation_monitor': selection['monitor'], 'best_minimized_monitor': best,
              'initial_conditioner_sha256': initial, 'metrics': metrics,
              'trainable_parameters': sum(p.numel() for _, p in trainable),
              'artifact_sha256': {str(p): sha(p) for p in artifacts}}
    atomic_json(out / 'metrics.json', result)
    return result


def validate_native(plan,cfg):
    from common.run_state import validate_resume_state
    native=Path(plan['native_dir']);fold=plan['fold']
    expected=load_yaml_config(plan['source_base_config']) if plan['native_reused'] else cfg
    state=json.loads((native/'metrics.json').read_text())
    valid=validate_resume_state(state,native/'config.json',plan['method'],expected)
    record=next((r for r in valid if r['fold']==fold),None)
    if record is None or any(record.get('sample_failures',{}).values()):raise ValueError('native fold incomplete')
    prediction=native/f'fold{fold}_predictions.csv'
    if sorted(membership(rows(prediction)))!=sorted(membership(rows(Path(cfg['split_dir'])/f'fold{fold}/test.csv'))):
        raise ValueError('native predictions differ from exact common test split')
    paths=[native/'config.json',native/'metrics.json',native/f'fold{fold}_best.pt',prediction]
    return {str(p):sha(p) for p in paths}


def native_stage(plan,cfg,device):
    if plan['native_reused']:return validate_native(plan,cfg)
    out=Path(plan['native_dir']);out.mkdir(parents=True,exist_ok=True)
    if not (out/'metrics.json').exists():
        before = time.monotonic()
        from train import train_one_fold,SummaryWriter,_write_metrics
        from pathotme.mscpt_tme import MSCPTNativeMethod
        if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=cfg:
            raise ValueError('native output config changed')
        atomic_json(out/'config.json',cfg)
        writer=SummaryWriter(log_dir=str(out/'tensorboard'))
        try:result=train_one_fold(plan['fold'],cfg,MSCPTNativeMethod(cfg,device),writer)
        finally:writer.flush();writer.close()
        _write_metrics(out/'metrics.json','mscpt',cfg,[{'fold':plan['fold'],**result}])
        atomic_json(out/'producer.json',{'extension':cfg['pathotme_extension'],
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'runner_sha256':sha(__file__), 'native_wall_seconds': time.monotonic() - before})
    return validate_native(plan,cfg)


def smoke_fixture(plan,cfg,device,launch_identity):
    import torch
    from train import set_seed, build_loaders
    from pathotme.mscpt_tme import MSCPTNativeMethod
    from run_vila_guided import _atomic_torch
    fixture = Path(plan['smoke_checkpoint_dir'])
    checkpoint = fixture / f"fold{plan['fold']}_best.pt"
    fixture_identity = identity({'launch': launch_identity, 'plan': plan,
                                 'policy': 'one_training_bag_step_only_not_a_paper_baseline'})
    if (fixture/'fixture.json').exists():
        saved = json.loads((fixture/'fixture.json').read_text())
        if saved.get('identity') != fixture_identity:
            raise ValueError('smoke fixture identity mismatch')
        verify_files(saved['artifact_sha256'])
        return saved['artifact_sha256']
    if checkpoint.exists():
        raise ValueError('unfinished fixture export requires inspection')
    set_seed(cfg['seed'] + plan['fold'])
    method = MSCPTNativeMethod(cfg, device)
    model = method.build_model()
    loader = build_loaders('mscpt', cfg, plan['fold'])[0]
    model.train()
    before = time.monotonic()
    step = method.train_step(next(iter(loader)), model, method.build_optimizer(model), torch.nn.CrossEntropyLoss())
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if (not torch.isfinite(step['logits']).all() or not gradients
            or not all(torch.isfinite(g).all() for g in gradients)
            or model.Custom_model.prompt_learner.p_input.grad is None
            or not model.Custom_model.prompt_learner.p_input.grad.abs().sum() > 0):
        raise ValueError('native MSCPT forward/gradient smoke failed')
    fixture.mkdir(parents=True, exist_ok=True)
    _atomic_torch(checkpoint, model.state_dict())
    atomic_json(fixture/'fixture.json', {'policy': 'one_training_bag_step_only_not_a_paper_baseline',
        'identity': fixture_identity, 'artifact_sha256': {str(checkpoint): sha(checkpoint)},
        'config': cfg, 'native_loss': step['loss'], 'native_step_wall_seconds': time.monotonic() - before,
        'native_objective': 'sum_three_branch_cross_entropy', 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    del model, method
    gc.collect(); torch.cuda.empty_cache()
    return {str(checkpoint): sha(checkpoint)}


def execute(args,launch,plan):
    import torch
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('allocated GPU job required')
    cfg=load_yaml_config(plan['config']);check_features(plan,cfg)
    root=Path(launch['output'])/f"smokes/{plan['cohort']}/{plan['encoder']}"
    if args.smoke and (root/'smoke_report.json').exists():
        report=json.loads((root/'smoke_report.json').read_text())
        if report.get('status')!='smoke_passed' or report.get('launch_identity')!=launch['identity']:
            raise ValueError('smoke identity mismatch')
        verify_files(report['artifact_sha256'])
        for result in report['arms'].values():
            verify_files(result['artifact_sha256'])
        return
    if args.smoke:native_hashes=smoke_fixture(plan,cfg,args.device,launch['identity'])
    else:
        smoke=json.loads((root/'smoke_report.json').read_text())
        if smoke.get('status')!='smoke_passed' or smoke.get('launch_identity')!=launch['identity']:
            raise ValueError('matching successful encoder smoke required')
        verify_files(smoke['artifact_sha256'])
        for result in smoke['arms'].values():
            verify_files(result['artifact_sha256'])
        native_hashes=native_stage(plan,cfg,args.device)
    results={}
    for arm in ['zero','actual','shuffled']:
        results[arm]=arm_run(launch,plan,cfg,arm,args.smoke,args.device,native_hashes)
        gc.collect();torch.cuda.empty_cache()
    if len({r['initial_conditioner_sha256'] for r in results.values()})!=1:
        raise ValueError('adapter initializations differ')
    if args.smoke:
        atomic_json(root/'smoke_report.json',{'status':'smoke_passed','launch_identity':launch['identity'],
            'cohort':plan['cohort'],'method':plan['method'],'encoder':cfg['backbone'],'arms':results,
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'artifact_sha256':{**native_hashes, **{str(root/arm/'metrics.json'):sha(root/arm/'metrics.json') for arm in results}}})
    else:
        atomic_json(Path(plan['output'])/'fold_complete.json',{'status':'completed',
            'launch_identity':launch['identity'],'plan':plan,'native_sha256':native_hashes,
            'arms':results,'slurm_job_id':os.environ.get('SLURM_JOB_ID')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch',type=Path,required=True)
    parser.add_argument('--cohort',choices=['nsclc','brca'],required=True)
    parser.add_argument('--encoder', choices=['plip','clip-rn50'], required=True)
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();load_dotenv(PGVL/'.env')
    launch=load_launch(args.launch);plan=selected(launch,args.cohort,args.encoder,args.fold)
    if args.smoke and args.fold!=0:raise ValueError('smokes use fold-zero training bags only')
    if not args.execute:print(json.dumps({'launch_identity':launch['identity'],'plan':plan,'smoke':args.smoke}))
    else:
        locks=Path(launch['output'])/'locks';locks.mkdir(exist_ok=True)
        with (locks/f"{args.cohort}_{args.encoder}_{'smoke' if args.smoke else args.fold}.lock").open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            execute(args,launch,plan)
