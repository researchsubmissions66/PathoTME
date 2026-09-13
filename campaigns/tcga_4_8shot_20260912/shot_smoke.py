"""Training-bag checks for both shot values; full training loops are unchanged."""
import gc
import hashlib
import json
from pathlib import Path
import sys

from shot_contract import tag_for
from pathotme.locked_tcga import atomic_json,identity,sha,verify_files


def arm_smoke(launch,plan,cfg,arm,device,native_hashes):
    import numpy as np
    import torch
    from train import build_loaders,set_seed
    from pathotme.locked_models import make_method as native_factory
    from pathotme.cross_encoder_models import make_method as cross_factory
    sys.path.insert(0,'/path/to/PathoTME/repairs/muse_smoke_margin_20260912')
    from muse_smoke_checks import require_tme_margin_effect
    spec=launch['protocol'];method=plan['method'];fold=plan['fold']
    output=Path(launch['output'])/'smokes'/plan['cohort']/method/arm
    output.mkdir(parents=True,exist_ok=True)
    run_id=identity({'launch':launch['identity'],'plan':plan,'arm':arm,
                     'smoke':True,'native_sha256':native_hashes})
    config={**cfg,'_fold_index':fold,'results_dir':str(output),
        'base_checkpoint_dir':plan['smoke_checkpoint_dir'],'tme_feature_csv':plan['tme_csv'],
        'tme_panel':spec['panels'][plan['cohort']], 'tme_mode':'zero' if arm=='zero' else 'actual',
        **{'tme_'+k:v for k,v in spec['adapter'].items() if k in ['hidden_dim','attention_heads','dropout','initial_gate']},
        'optimizer':'adam','lr':spec['adapter']['lr'],'weight_decay':spec['adapter']['weight_decay'],
        'lr_scheduler':None}
    atomic_json(output/'config.json',{'identity':run_id,'launch_identity':launch['identity'],
                                    'config':config,'arm':arm,'native_sha256':native_hashes})
    set_seed(spec['seed']+fold)
    loaders=build_loaders(method,config,fold)
    factory=native_factory if tag_for(method,plan['encoder'])=='native' else cross_factory
    bridge=factory(config,plan['cohort'],method,arm,json.loads(Path(plan['donor_maps']).read_text()),device)
    model=bridge.build_model();bridge.prepare_fold(fold,model,loaders[0])
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith('conditioner.') for n,p in trainable):
        raise ValueError('only conditioner parameters may train')
    initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for n,p in trainable)).hexdigest()
    optimizer=bridge.build_optimizer(model);batch=next(iter(loaders[0]));model.eval()
    inputs=bridge.inputs(batch)[0] if method=='mgpath' else bridge._inputs(batch,model)[0]
    with torch.no_grad():
        native=model.base(*inputs[:4]) if method=='mgpath' else model.base(*inputs[:5])[0]
        projection=model.conditioner.output_projection.weight
        saved=projection.clone()
        try:
            projection.zero_()
            observed=model(*inputs) if method=='mgpath' else model(*inputs)[0]
            torch.testing.assert_close(observed,native,rtol=1e-5,atol=1e-5)
        finally:projection.copy_(saved)
        observed=model(*inputs,return_details=True)['logits']
        varied=(*inputs[:-1],torch.full_like(inputs[-1],1e3))
        other=model(*varied,return_details=True)['logits']
        if method=='vila_mil':
            relabeled=(*inputs[:4],1-inputs[4],inputs[-1])
            torch.testing.assert_close(observed,model(*relabeled,return_details=True)['logits'],rtol=0,atol=0)
    if arm=='zero':
        torch.testing.assert_close(observed,other,rtol=0,atol=0)
        probe={'policy':'binary_decision_margin_and_gradient','zero_input_independent':True}
    else:
        values=inputs[-1].detach().clone().requires_grad_()
        logits=model(*inputs[:-1],values,return_details=True)['logits']
        gradient=torch.autograd.grad((logits[:,1]-logits[:,0]).sum(),values,allow_unused=True)[0]
        probe={'policy':'binary_decision_margin_and_gradient',**require_tme_margin_effect(observed,other,gradient)}
    model.train();step=bridge.train_step(batch,model,optimizer,None)
    gradients=[p.grad for n,p in trainable if p.grad is not None]
    if (not np.isfinite(step['loss']) or not gradients or not all(torch.isfinite(g).all() for g in gradients)
            or not any(g.abs().sum()>0 for g in gradients) or any(p.grad is not None for p in model.base.parameters())):
        raise ValueError('invalid adapter gradients or frozen-base violation')
    model.eval()
    with torch.no_grad():
        if not torch.isfinite(bridge.eval_step(batch,model)['logits']).all():raise ValueError('nonfinite post-update logits')
    result={'status':'completed','identity':run_id,'initial_conditioner_sha256':initial,
        'training_bag_only':True,'zero_residual_native_equivalence':True,'only_adapter_gradients':True,
        'loss':step['loss'],'probe':probe,'shots':plan['shots'],
        'trainable_parameters':sum(p.numel() for n,p in trainable),
        'artifact_sha256':{str(output/'config.json'):sha(output/'config.json')}}
    atomic_json(output/'metrics.json',result)
    print(json.dumps({'cohort':plan['cohort'],'method':method,'encoder':plan['encoder'],
                      'shots':plan['shots'],'arm':arm,'probe':probe}),flush=True)
    return result


def run_smoke(launch,plan,device):
    import torch
    from common.configuration import load_yaml_config
    from run_locked_tcga import check_features
    cfg=load_yaml_config(plan['config']);check_features(plan,cfg)
    checkpoint=Path(plan['smoke_checkpoint_dir'])/'fold0_best.pt'
    native_hashes={str(checkpoint):plan['smoke_checkpoint_sha256']};verify_files(native_hashes)
    root=Path(launch['output'])/'smokes'/plan['cohort']/plan['method']
    if (root/'smoke_report.json').exists():raise FileExistsError('inspect existing smoke before repeating it')
    results={}
    for arm in ['zero','actual','shuffled']:
        results[arm]=arm_smoke(launch,plan,cfg,arm,device,native_hashes)
        gc.collect();torch.cuda.empty_cache()
    if len({r['initial_conditioner_sha256'] for r in results.values()})!=1:raise ValueError('initial conditioners differ')
    import os
    atomic_json(root/'smoke_report.json',{'status':'smoke_passed','launch_identity':launch['identity'],
        'cohort':plan['cohort'],'method':plan['method'],'encoder':plan['encoder'],'shots':plan['shots'],
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'arms':results,
        'note':'16-shot native checkpoint used only as a structural fixture; this shot uses its own training data and fitted statistics',
        'artifact_sha256':{**native_hashes,**{str(root/arm/'metrics.json'):sha(root/arm/'metrics.json') for arm in results}}})
