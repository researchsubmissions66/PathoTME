"""Focused cached-model checks; GPU smokes remain required for actual training."""
import argparse,faulthandler,gc,json,sys,time,traceback
from pathlib import Path
sys.path[:0]=[str(Path(__file__).parent),'/path/to/PathoTME','/path/to/PGVL-Gym','/path/to/PathoTME/scripts']
faulthandler.dump_traceback_later(120,repeat=True)

def bounded_batch(batch,method):
    """Small CPU probe copied from one training bag, never a research result."""
    import torch
    result=list(batch)
    # DyKo's WAKI clustering is defined on the complete bag; a short spatial
    # prefix can collapse its visual variation. Keep that architecture intact.
    if method=='dyko':return result
    if method=='mgpath':
        for i in [0,2]:result[i]=batch[i][:,:64].clone()
        for i in [1,3]:
            edge=batch[i];keep=(edge<64).all(dim=-2)
            result[i]=edge[:,:,keep[0]].clone()
    else:
        for i,value in enumerate(batch[:-2]):
            if isinstance(value,torch.Tensor):result[i]=value[:,:32].clone()
    return result

def main(a):
    from pathotme.locked_tcga import load_launch,atomic_json,sha
    from common.configuration import load_yaml_config,load_dotenv
    from train import build_loaders,set_seed
    from runtime_models import validate_config,native_type,adapter_type,guided_model
    from runtime_checks import check_features,check_model
    import torch
    launch=load_launch(a.launch);load_dotenv('/path/to/PGVL-Gym/.env')
    report={'status':'running','launch_identity':launch['identity'],'scope':'CPU cached native construction, real training-bag loader, bounded adapter forward/gradient; no GPU result','groups':[],'source_sha256':{str(Path(__file__)):sha(__file__)}}
    out=Path(launch['output'])/'cpu_validation.json'
    if a.reuse_non_dyko:
        amendment=launch['preflight_amendment']
        if amendment.get('reuse_non_dyko_validated') is not True:raise ValueError('No proven unchanged non-DyKo branch')
        prior_path=Path(amendment['prior_validation'])
        if sha(prior_path)!=amendment['prior_validation_sha256']:raise ValueError('Prior validation changed')
        prior=json.loads(prior_path.read_text())
        if prior['launch_identity']!=amendment['prior_identity']:raise ValueError('Prior validation identity mismatch')
        kept=[g for g in prior['groups'] if g['method']!='dyko']
        if len(kept)!=12 or any(g['status']!='passed' for g in kept):raise ValueError('Twelve passing unchanged groups required')
        report['groups']=[{**g,'reused_from_launch':prior['launch_identity']} for g in kept]
        report['source_sha256'][str(prior_path)]=sha(prior_path)
    for p in launch['plans']:validate_config(load_yaml_config(p['config']))
    for p in launch['plans']:
        if p['fold']!=0 or (a.reuse_non_dyko and p['method']!='dyko'):continue
        item={'method':p['method'],'encoder':p['encoder'],'cohort':p['cohort']};start=time.monotonic()
        print('CHECK',item,flush=True)
        try:
            cfg=load_yaml_config(p['config']);check_features(p,cfg);set_seed(1)
            loaders=build_loaders(p['method'],cfg,0);batch=next(iter(loaders[0]));tiny=bounded_batch(batch,p['method'])
            item['actual_training_bag_shapes']=[list(x.shape) for x in batch if isinstance(x,torch.Tensor)]
            native=native_type(p['method'],p['encoder'])(cfg,'cpu');base=native.build_model()
            adapter=p['adapter'];maps=json.loads(Path(p['donor_maps']).read_text());initial=[];checks={}
            for arm in ['zero','actual','shuffled']:
                arm_cfg={**cfg,'pathotme_arm':arm,'base_checkpoint_dir':p['native_dir'],'tme_feature_csv':p['tme_csv'],'tme_panel':'shared_core62_v1','tme_mode':'zero' if arm=='zero' else 'actual','optimizer':'adam','lr':adapter['lr'],'weight_decay':adapter['weight_decay'],'lr_scheduler':None,**{'tme_'+k:adapter[k] for k in ['hidden_dim','attention_heads','dropout','initial_gate']}}
                bridge=adapter_type(p['method'],p['encoder'])(arm_cfg,maps,arm,'cpu')
                set_seed(1);model=guided_model(p['method'],base,arm_cfg);bridge.prepare_fold(0,model,loaders[0])
                checks[arm]=check_model(bridge,model,tiny,cfg,arm)
                model.train();optimizer=bridge.build_optimizer(model);step=bridge.train_step(tiny,model,optimizer,None)
                gradients=[x.grad for x in model.conditioner.parameters() if x.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum()>0 for g in gradients):raise ValueError('Invalid conditioner gradients')
                if any(x.grad is not None for x in base.parameters()):raise ValueError('Frozen native base received gradients')
                if not torch.isfinite(step['logits']).all():raise ValueError('Invalid CPU probe logits')
                del model,bridge,optimizer
            item.update(status='passed',checks=checks)
            del base,native,loaders,batch,tiny;gc.collect()
        except Exception as e:
            item.update(status='failed',error=str(e),traceback=traceback.format_exc());traceback.print_exc()
        item['seconds']=time.monotonic()-start;report['groups'].append(item);atomic_json(out,report)
        print('RESULT',item['method'],item['encoder'],item['status'],item['seconds'],flush=True)
    report['status']='passed' if len(report['groups'])==14 and all(x['status']=='passed' for x in report['groups']) else 'failed'
    atomic_json(out,report);print(json.dumps({'status':report['status'],'path':str(out)}),flush=True)
    if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--launch',type=Path,required=True);p.add_argument('--reuse-non-dyko',action='store_true');main(p.parse_args())
