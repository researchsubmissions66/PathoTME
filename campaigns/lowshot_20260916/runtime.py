"""Reuse frozen native training and original per-method adapter loops at new shots."""
import argparse
import contextlib
import fcntl
import gc
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import FunctionType
from contract import *

COHORT_RUNTIME = ROOT/'campaigns/crc_blca_core62_20260914'
sys.path[:0] = [str(ROOT/'scripts'),str(COHORT_RUNTIME),
               str(PGVL/'benchmarks/pathotme_blca_dyko_directional_20260915')]

def rebound(fn, **bindings):
    result=FunctionType(fn.__code__,{**fn.__globals__,**bindings},fn.__name__,fn.__defaults__,fn.__closure__)
    result.__kwdefaults__=fn.__kwdefaults__
    return result

@contextlib.contextmanager
def bindings(plan):
    """Only validators/task factories are scoped; restore every original on exit."""
    changes=[]
    def patch(module,key,value):
        changes.append((module,key,getattr(module,key)));setattr(module,key,value)
    try:
        for name in ['focus','muse','hive','dyko','mscpt']:
            patch(importlib.import_module('pathotme.'+name+'_tme'),'validate_config',validate_config)
        cross=importlib.import_module('pathotme.cross_encoder_models')
        patch(cross,'validate_clip_config',validate_config)
        locked=importlib.import_module('pathotme.locked_models')
        from pathotme.brca_mgpath_index_repair import IndexedBreastGuidedMGPathMethod
        patch(locked,'BreastGuidedMGPathMethod',IndexedBreastGuidedMGPathMethod)
        cohort=importlib.import_module('runtime_models')
        patch(cohort,'validate_config',validate_config)
        patch(cohort,'_NATIVE',{});patch(cohort,'_ADAPTER',{})
        yield
    finally:
        for module,key,value in reversed(changes):setattr(module,key,value)

def native_class(plan):
    method=plan['method'];encoder=plan['encoder']
    if plan['cohort'] in ['crc','blca']:
        from runtime_models import native_type
        return native_type(method,encoder)
    if method=='vila_mil':
        from methods.vila_mil.adapter import ViLaMILMethod
        return ViLaMILMethod
    if method=='mgpath':
        if encoder=='clip-rn50':
            from pathotme.cross_encoder_models import CLIPMGPathMethod
            return CLIPMGPathMethod
        from methods.mgpath.adapter import MGPathMethod
        return MGPathMethod
    names={'focus':('focus','Focus'),'muse':('muse','MUSE'),'hive_mil':('hive','HiVE'),
           'dyko':('dyko','DyKo'),'mscpt':('mscpt','MSCPT')}
    name,prefix=names[method]
    return getattr(importlib.import_module('pathotme.'+name+'_tme'),prefix+'NativeMethod')

def adapter(plan,cfg,maps,arm,device):
    validate_config(cfg)
    if plan['cohort'] in ['crc','blca']:
        from runtime_models import adapter_type
        return adapter_type(plan['method'],plan['encoder'])(cfg,maps,arm,device)
    if plan['method'] in ['vila_mil','mgpath']:
        native=(plan['method'],plan['encoder']) in [('vila_mil','clip-rn50'),('mgpath','plip')]
        module=importlib.import_module('pathotme.locked_models' if native else 'pathotme.cross_encoder_models')
        return module.make_method(cfg,plan['cohort'],plan['method'],arm,maps,device)
    names={'focus':('focus','Focus'),'muse':('muse','MUSE'),'hive_mil':('hive','HiVE'),
           'dyko':('dyko','DyKo'),'mscpt':('mscpt','MSCPT')}
    name,prefix=names[plan['method']]
    return getattr(importlib.import_module('pathotme.'+name+'_tme'),prefix+'TMEMethod')(cfg,maps,arm,device)

def parent_runner(plan):
    if plan['cohort'] in ['crc','blca']:return importlib.import_module('run_campaign')
    if plan['method'] in ['vila_mil','mgpath']:
        native=(plan['method'],plan['encoder']) in [('vila_mil','clip-rn50'),('mgpath','plip')]
        return importlib.import_module('run_locked_tcga' if native else 'run_cross_encoder_tcga')
    name={'hive_mil':'hive'}.get(plan['method'],plan['method'])
    return importlib.import_module('run_'+name+'_tcga')

def native_stage(plan,cfg,device):
    from run_campaign import validate_native
    from train import train_one_fold,SummaryWriter,_write_metrics
    out=Path(plan['native_dir']);out.mkdir(parents=True,exist_ok=True)
    if not (out/'metrics.json').exists():
        if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=cfg:
            raise ValueError('Native config changed')
        atomic_json(out/'config.json',cfg);before=time.monotonic()
        writer=SummaryWriter(log_dir=str(out/'tensorboard'))
        try:result=train_one_fold(plan['fold'],cfg,native_class(plan)(cfg,device),writer)
        finally:writer.flush();writer.close()
        _write_metrics(out/'metrics.json',plan['method'],cfg,[{'fold':plan['fold'],**result}])
        atomic_json(out/'producer.json',{'runner_sha256':sha(__file__),'shots':plan['shots'],
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'native_wall_seconds':time.monotonic()-before,
            'parent_native_recipe':plan['parent_config'],'training':'unchanged train.train_one_fold'})
    return validate_native(plan,cfg)

def smoke_check(plan,bridge,model,batch,cfg,arm):
    import runtime_checks
    from repair_checks import amended_source,require_invariant_logits,require_directional_margin
    # The previously validated numerical amendment, with only native task dispatch rebound.
    evidence=[]
    def invariant(a,b):evidence.append(require_invariant_logits(a,b))
    namespace={**runtime_checks.check_model.__globals__,
               'native_type':lambda method,encoder:native_class(plan),
               'require_invariant_logits':invariant,'require_directional_margin':require_directional_margin}
    exec(compile(amended_source(runtime_checks.check_model),__file__,'exec'),namespace)
    return {**namespace['check_model'](bridge,model,batch,cfg,arm),'invariance_checks':evidence}

def arm_smoke(launch,plan,cfg,arm,device,native_hashes):
    import numpy as np
    import torch
    from train import build_loaders,set_seed
    config=adapter_config(plan,cfg,arm,True);validate_config(config)
    out=Path(config['results_dir']);out.mkdir(parents=True,exist_ok=True)
    run_id=identity({'launch':launch['identity'],'plan':plan,'arm':arm,'smoke':True,'native_sha256':native_hashes})
    atomic_json(out/'config.json',{'identity':run_id,'config':config,'native_sha256':native_hashes})
    set_seed(1+plan['fold']);loaders=build_loaders(plan['method'],config,plan['fold'])
    bridge=adapter(plan,config,json.loads(Path(plan['donor_maps']).read_text()),arm,device)
    model=bridge.build_model();bridge.prepare_fold(plan['fold'],model,loaders[0])
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith('conditioner.') for n,p in trainable):raise ValueError('Unfrozen native parameters')
    initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for n,p in trainable)).hexdigest()
    batch=next(iter(loaders[0]));diagnostic=smoke_check(plan,bridge,model,batch,cfg,arm)
    model.train();step=bridge.train_step(batch,model,bridge.build_optimizer(model),None)
    gradients=[p.grad for n,p in trainable if p.grad is not None]
    if (not np.isfinite(step['loss']) or not gradients or not all(torch.isfinite(g).all() for g in gradients)
        or not any(g.abs().sum()>0 for g in gradients) or any(p.grad is not None for p in model.base.parameters())):
        raise ValueError('Invalid adapter gradient or frozen-base violation')
    model.eval()
    with torch.no_grad():
        if not torch.isfinite(bridge.eval_step(batch,model)['logits']).all():raise ValueError('Nonfinite updated logits')
    result={'status':'completed','identity':run_id,'initial_conditioner_sha256':initial,
        'training_bag_only':True,'shots':plan['shots'],'loss':step['loss'],'diagnostic':diagnostic,
        'only_adapter_gradients':True,'artifact_sha256':{str(out/'config.json'):sha(out/'config.json')}}
    atomic_json(out/'metrics.json',result)
    print(json.dumps({'smoke':plan['id'],'arm':arm,'diagnostic':diagnostic}),flush=True)
    return result

def smoke_report(plan,launch):
    path=Path(plan['smoke_output'])/'smoke_report.json'
    report=json.loads(path.read_text())
    if report['status']!='smoke_passed' or report['launch_identity']!=launch['identity']:
        raise ValueError('Matching low-shot smoke required')
    verify_files(report['artifact_sha256'])
    for arm in report['arms'].values():verify_files(arm['artifact_sha256'])
    return report

def execute(campaign,plan,smoke,device):
    import torch
    from run_campaign import smoke_fixture
    cfg=read_config(plan['config']);check_features(plan,cfg)
    launch={'identity':campaign['identity'],'output':plan['group_output'],'protocol':plan['protocol']}
    with bindings(plan):
        if smoke:
            if (Path(plan['smoke_output'])/'smoke_report.json').exists():return smoke_report(plan,launch)
            fixture=rebound(smoke_fixture,native_type=lambda method,encoder:native_class(plan))
            native_hashes=fixture(plan,cfg,device,launch['identity'])
        else:
            smoke_report(plan,launch)
            native_hashes=native_stage(plan,cfg,device)
        results={}
        for arm in ['zero','actual','shuffled']:
            if smoke:results[arm]=arm_smoke(launch,plan,cfg,arm,device,native_hashes)
            else:results[arm]=parent_runner(plan).arm_run(launch,plan,cfg,arm,False,device,native_hashes)
            gc.collect();torch.cuda.empty_cache()
        if len({r['initial_conditioner_sha256'] for r in results.values()})!=1:raise ValueError('Conditioner initializations differ')
        if smoke:
            root=Path(plan['smoke_output'])
            atomic_json(root/'smoke_report.json',{'status':'smoke_passed','launch_identity':launch['identity'],
                'plan':plan,'arms':results,'slurm_job_id':os.environ['SLURM_JOB_ID'],
                'artifact_sha256':{**native_hashes,**{str(root/a/'metrics.json'):sha(root/a/'metrics.json') for a in results}}})
        else:
            atomic_json(Path(plan['output'])/'fold_complete.json',{'status':'completed','launch_identity':launch['identity'],
                'plan':plan,'native_sha256':native_hashes,'arms':results,'slurm_job_id':os.environ['SLURM_JOB_ID']})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=OUTPUT/'campaign.json')
    parser.add_argument('--group',required=True)
    parser.add_argument('--shots',type=int,choices=[4,8])
    parser.add_argument('--fold',type=int,choices=range(5),default=0)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    campaign=load_campaign(args.campaign)
    plans=[p for p in campaign['plans'] if p['group']==args.group and p['fold']==args.fold
           and (args.smoke or p['shots']==args.shots)]
    group=next(g for g in campaign['groups'] if g['id']==args.group)
    if (args.smoke and (args.fold!=0 or args.shots is not None or len(plans)!=len(group['shots']))) or (not args.smoke and len(plans)!=1):
        raise ValueError('Invalid request')
    if not args.execute:print(json.dumps({'plans':[p['id'] for p in plans],'smoke':args.smoke}));return
    import torch
    from common.configuration import load_dotenv
    load_dotenv(PGVL/'.env')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
    lockdir=OUTPUT/'locks';lockdir.mkdir(exist_ok=True)
    key=args.group+('_smoke' if args.smoke else f'_{args.shots}_{args.fold}')
    with (lockdir/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for plan in plans:
            execute(campaign,plan,args.smoke,args.device);gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
