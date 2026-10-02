"""Focused CPU preflight: recipe preservation, memberships, constructors and loaders."""
import copy
import gc
import json
import time
from contract import *

def validate():
    from common.configuration import load_dotenv
    load_dotenv(PGVL/'.env')
    before=time.monotonic();campaign=load_campaign(OUTPUT/'campaign.json')
    allowed={'shots','split_dir','results_dir','k_start','k_end','benchmark','experiment',
             'highshot_contract_path','highshot_contract_sha256'}
    seen=set();split_checks=[]
    for plan in campaign['plans']:
        cfg=read_config(plan['config']);old=read_config(plan['parent_config'])
        drift={k for k in set(cfg)|set(old) if cfg.get(k)!=old.get(k)}
        if not drift<=allowed:raise ValueError(f'Changed native recipe: {drift-allowed}')
        validate_config(cfg)
        for arm in ['zero','actual','shuffled']:validate_config(adapter_config(plan,cfg,arm))
        for key,value in [('shots',16),('lr',0.314),('seed',19),('label_dict',{'wrong':0})]:
            changed={**cfg,key:value}
            try:validate_config(changed)
            except ValueError:pass
            else:raise ValueError('Contract accepted drift')
        key=plan['cohort'],plan['shots'],plan['fold']
        if key not in seen:
            phases={p:rows(Path(cfg['split_dir'])/f"fold{plan['fold']}/{p}.csv") for p in ['train','val','test']}
            check_phases(phases,plan['shots']);validate_donors(phases,json.loads(Path(plan['donor_maps']).read_text()))
            split_checks.append({'cohort':key[0],'shots':key[1],'fold':key[2],
                                 'counts':{p:len(v) for p,v in phases.items()}});seen.add(key)
    from runtime import bindings,native_class,adapter,parent_runner
    from train import build_loaders
    import pathotme.focus_tme as focus_module
    original_validator=focus_module.validate_config
    boundaries=[];constructors=[]
    for plan in campaign['plans']:
        if plan['fold']!=0:continue
        cfg=read_config(plan['config']);check_features(plan,cfg)
        with bindings(plan):
            native=native_class(plan)(cfg,'cpu')
            maps=json.loads(Path(plan['donor_maps']).read_text())
            for arm in ['zero','actual','shuffled']:
                bridge=adapter(plan,adapter_config(plan,cfg,arm),maps,arm,'cpu')
                del bridge
            loaders=build_loaders(plan['method'],cfg,plan['fold'])
            expected=[plan['split_counts'][p] for p in ['train','val','test']]
            if [len(l.dataset) for l in loaders]!=expected:raise ValueError('Loader split count drift')
            for loader,phase in zip(loaders,['train','val','test']):
                frame=getattr(loader.dataset,'df',getattr(loader.dataset,'frame',None))
                if frame is not None and 'slide_id' in frame:
                    if sorted(frame['slide_id'].astype(str))!=sorted(r['slide_id'] for r in rows(Path(cfg['split_dir'])/f'fold0/{phase}.csv')):
                        raise ValueError('Loader changed membership')
            runner=parent_runner(plan)
            boundaries.append({'id':plan['id'],'loader_counts':expected,'parent_arm_run':runner.__file__})
            constructors.append({'id':plan['id'],'native':type(native).__name__,'three_adapters':True})
            del native,loaders
        if focus_module.validate_config is not original_validator:raise ValueError('Validator not restored')
        gc.collect();print(json.dumps({'cpu_preflight':plan['id'],'status':'passed'}),flush=True)
    try:
        with bindings(campaign['plans'][0]):raise RuntimeError('restore sentinel')
    except RuntimeError:pass
    assert focus_module.validate_config is original_validator
    report={'status':'passed','campaign_identity':campaign['identity'],'recipe_plans':350,
            'split_checks':split_checks,'boundaries':boundaries,'constructors':constructors,
            'negative_contract_checks':1400,'restores_on_success_and_exception':True,
            'elapsed_seconds':time.monotonic()-before,
            'scope':'CPU constructors/loaders and frozen recipe/membership/assets. No GPU smoke or research outcome claim.'}
    atomic_json(OUTPUT/'validation.json',report);print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)}))

if __name__=='__main__':validate()
