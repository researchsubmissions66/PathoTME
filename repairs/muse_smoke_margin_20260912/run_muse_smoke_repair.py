"""Amend only failed MUSE structural smokes; retain the original fold runtime."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym']
from pathotme.locked_tcga import atomic_json, identity, load_launch, sha, verify_files


def load_amendment(path):
    amendment = json.loads(Path(path).read_text())
    expected = amendment.pop('identity')
    if identity(amendment) != expected:
        raise ValueError('smoke amendment identity mismatch')
    amendment['identity'] = expected
    verify_files(amendment['file_sha256'])
    launch = load_launch(amendment['parent_launch'])
    if launch['identity'] != amendment['parent_launch_identity']:
        raise ValueError('parent experiment identity mismatch')
    return amendment, launch


def run_arm(amendment, launch, plan, cfg, arm, device, native_hashes):
    import numpy as np
    import torch
    from train import build_loaders, set_seed
    from pathotme.muse_tme import MUSETMEMethod
    from muse_smoke_checks import check_model
    spec = launch['protocol']
    output = Path(amendment['output'])/'smokes'/plan['cohort']/plan['encoder']/arm
    output.mkdir(parents=True, exist_ok=True)
    run_id = identity({'amendment':amendment['identity'], 'launch':launch['identity'],
                       'plan':plan, 'arm':arm, 'native_sha256':native_hashes})
    config = {**cfg, '_fold_index':plan['fold'], 'results_dir':str(output),
        'base_checkpoint_dir':plan['smoke_checkpoint_dir'],
        'tme_feature_csv':plan['tme_csv'], 'tme_panel':spec['panels'][plan['cohort']],
        'tme_mode':'zero' if arm=='zero' else 'actual',
        **{'tme_'+k:v for k,v in spec['adapter'].items()
           if k in ['hidden_dim','attention_heads','dropout','initial_gate']},
        'optimizer':'adam', 'lr':spec['adapter']['lr'],
        'weight_decay':spec['adapter']['weight_decay'], 'lr_scheduler':None}
    atomic_json(output/'config.json', {'identity':run_id, 'amendment_identity':amendment['identity'],
        'launch_identity':launch['identity'], 'config':config, 'arm':arm,
        'native_sha256':native_hashes})
    maps = json.loads(Path(plan['donor_maps']).read_text())
    set_seed(spec['seed']+plan['fold'])
    loaders = build_loaders('muse',config,plan['fold'])
    bridge = MUSETMEMethod(config,maps,arm,device)
    model = bridge.build_model()
    bridge.prepare_fold(plan['fold'],model,loaders[0])
    parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not parameters or any(not n.startswith('conditioner.') for n,p in parameters):
        raise ValueError('only conditioner parameters may train')
    initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for n,p in parameters)).hexdigest()
    optimizer = bridge.build_optimizer(model)
    batch = next(iter(loaders[0]))
    inputs, metadata = bridge.inputs(batch,model)
    evidence = check_model(model,inputs,arm)
    model.train()
    step = bridge.train_step(batch,model,optimizer,None)
    gradients = [p.grad for n,p in parameters if p.grad is not None]
    if (not np.isfinite(step['loss']) or not gradients
            or not all(torch.isfinite(g).all() for g in gradients)
            or not any(g.abs().sum()>0 for g in gradients)
            or any(p.grad is not None for p in model.base.parameters())):
        raise ValueError('invalid adapter gradients or frozen-base violation')
    model.eval()
    with torch.no_grad():
        check = bridge.eval_step(batch,model)
        if not torch.isfinite(check['logits']).all():
            raise ValueError('nonfinite post-update logits')
    result = {'status':'completed', 'identity':run_id, 'amendment_identity':amendment['identity'],
        'initial_conditioner_sha256':initial, 'training_bag_only':True,
        'zero_residual_native_equivalence':True, 'only_adapter_gradients':True,
        'label_independent_predictions':True, 'loss':step['loss'],
        'trainable_parameters':sum(p.numel() for n,p in parameters),
        'training_slide_id':metadata['slide_id'], 'probe':evidence,
        'artifact_sha256':{str(output/'config.json'):sha(output/'config.json')}}
    atomic_json(output/'metrics.json',result)
    print(json.dumps({'cohort':plan['cohort'],'encoder':plan['encoder'],'arm':arm,
                      'loss':step['loss'],'probe':evidence}),flush=True)
    return result


def execute(args, amendment, launch, plan):
    import torch
    from common.configuration import load_dotenv, load_yaml_config
    from pathotme.muse_runtime import check_features
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('allocated GPU job required')
    load_dotenv('/path/to/PGVL-Gym/.env')
    cfg = load_yaml_config(plan['config'])
    check_features(plan,cfg)
    fixture_path = Path(plan['smoke_checkpoint_dir'])/'fixture.json'
    fixture = json.loads(fixture_path.read_text())
    expected = identity({'launch':launch['identity'],'plan':plan,
                         'policy':'one_training_bag_step_only_not_a_paper_baseline'})
    if fixture['identity'] != expected or fixture['config'] != cfg:
        raise ValueError('original structural fixture provenance mismatch')
    verify_files(fixture['artifact_sha256'])
    native_hashes = fixture['artifact_sha256']
    canonical = Path(launch['output'])/'smokes'/plan['cohort']/plan['encoder']/'smoke_report.json'
    if canonical.exists():
        raise FileExistsError('canonical smoke report already exists; inspect before retrying')
    results = {}
    for arm in ['zero','actual','shuffled']:
        results[arm] = run_arm(amendment,launch,plan,cfg,arm,args.device,native_hashes)
        gc.collect(); torch.cuda.empty_cache()
    if len({r['initial_conditioner_sha256'] for r in results.values()}) != 1:
        raise ValueError('adapter initializations differ')
    root = Path(amendment['output'])/'smokes'/plan['cohort']/plan['encoder']
    report = {'status':'smoke_passed','launch_identity':launch['identity'],
        'amendment_identity':amendment['identity'],'amendment_path':str(args.amendment.resolve()),
        'cohort':plan['cohort'],'method':'muse','encoder':plan['encoder'],'arms':results,
        'slurm_job_id':os.environ['SLURM_JOB_ID'],
        'policy':'same_experiment_new_structural_smoke_check_only',
        'native_fixture_reused':str(fixture_path),
        'artifact_sha256':{**native_hashes, str(fixture_path):sha(fixture_path),
            str(args.amendment.resolve()):sha(args.amendment),
            **amendment['file_sha256'],
            **{str(root/arm/'metrics.json'):sha(root/arm/'metrics.json') for arm in results}}}
    atomic_json(root/'smoke_report.json',report)
    # The unchanged fold runner verifies the original experiment identity and
    # every amendment/source/validation/artifact hash through this envelope.
    atomic_json(canonical,report)
    print(json.dumps({'status':'smoke_passed','canonical_report':str(canonical)}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment',type=Path,required=True)
    parser.add_argument('--cohort',choices=['nsclc','brca'],required=True)
    parser.add_argument('--encoder',choices=['plip','clip-rn50'],required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    amendment,launch = load_amendment(args.amendment)
    matches = [p for p in launch['plans'] if p['fold']==0
               and (p['cohort'],p['encoder'])==(args.cohort,args.encoder)]
    if len(matches)!=1 or [args.cohort,args.encoder] not in amendment['affected_pairs']:
        raise ValueError('only the three failed smoke pairs may be repaired')
    if not args.execute:
        print(json.dumps({'amendment_identity':amendment['identity'],'plan':matches[0]}));return
    locks = Path(launch['output'])/'locks'
    with (locks/f'{args.cohort}_{args.encoder}_smoke.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args,amendment,launch,matches[0])


if __name__ == '__main__':
    main()
