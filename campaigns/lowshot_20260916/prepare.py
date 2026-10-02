"""Derive the authorized low-shot grid from frozen native recipes and LR splits."""
import copy
import csv
import json
import math
import re
import yaml
from pathlib import Path
from collections import defaultdict
from contract import *
from pathotme.controlled_vila import donor_maps

def prepare():
    if OUTPUT.exists(): raise FileExistsError('Preserve existing campaign')
    baseline = json.loads(BASELINE.read_text()); allshots = json.loads(ALLSHOTS.read_text())
    assert baseline['identity'] == 'addccf688b73210efe87329e0fd034946c8a4ddebaf58644f3556fab1ded962e'
    assert allshots['identity'] == 'ef543376d8aab20ef176c4dc50e5979c14a516418504e881f12f083129577ed9'
    for manifest in [baseline, allshots]:
        assert identity({k:v for k,v in manifest.items() if k!='identity'}) == manifest['identity']
    OUTPUT.mkdir(parents=True)
    bindings = {str(p): sha(p) for p in [BASELINE, ALLSHOTS]}
    historical_only = {}
    def bind(path, expected=None):
        if str(path) in bindings:
            if expected is not None and bindings[str(path)] != expected: raise ValueError('Conflicting parent binding')
            return
        digest = sha(path)
        if expected is not None and digest != expected: raise ValueError(f'Parent binding changed: {path}')
        if str(path) in bindings and bindings[str(path)] != digest: raise ValueError('Inconsistent binding')
        bindings[str(path)] = digest
    parents = {}; tasks = {}
    for task in baseline['fusion_tasks']:
        if task['cohort'] not in SCOPE: continue
        tasks[task['cohort'],task['method'],task['encoder'],task['fold']] = task
        for p,d in task['input_sha256'].items(): bind(p,d)
        if task['launch'] not in parents:
            launch = json.loads(Path(task['launch']).read_text())
            assert identity({k:v for k,v in launch.items() if k!='identity'}) == task['launch_identity'] == launch['identity']
            bind(task['launch']); parents[task['launch']] = launch
            # Preserve all exact parent source/asset bindings, without editing any.
            for p,d in launch['file_sha256'].items():
                if re.search(r'/fold[0-4]_(best|final)\.pt$',p):
                    historical_only[p]=d
                else:bind(p,d)
            print(json.dumps({'verified_parent':task['launch'],'bindings':len(bindings)}),flush=True)
    for directory in [ROOT/'pathotme', ROOT/'scripts', PGVL/'common', PGVL/'methods', RUNTIME,
                      ROOT/'campaigns/crc_blca_core62_20260914']:
        for p in directory.rglob('*.py'): bind(p)
    for p in [PGVL/'train.py',PGVL/'benchmarks/pathotme_blca_dyko_directional_20260915/repair_checks.py']:
        bind(p)
    for name in ['parent_jobs.json','parent_sacct.txt','queue_before.txt']:
        src=Path('/tmp')/('pathotme_lowshot_'+name)
        dest=OUTPUT/'evidence'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(src.read_bytes());bind(dest)
    ids=json.loads((OUTPUT/'evidence/parent_jobs.json').read_text())
    accounting=list(csv.DictReader((OUTPUT/'evidence/parent_sacct.txt').open(),delimiter='|'))
    timings=defaultdict(list);memory=defaultdict(list)
    for row in accounting:
        jid=row['JobIDRaw'].split('.')[0]
        if jid not in ids: continue
        task=ids[jid][0];key=(task['cohort'],task['method'],task['encoder'])
        if '.' not in row['JobIDRaw'] and row['State']=='COMPLETED': timings[key].append(int(row['ElapsedRaw']))
        rss=row['MaxRSS']
        if rss:
            units={'K':1/1024**2,'M':1/1024,'G':1,'T':1024}
            memory[key].append(float(rss[:-1])*units[rss[-1]] if rss[-1] in units else float(rss)/1024**3)
    splits={}
    for cohort, shots_list in SCOPE.items():
        reference=tasks[cohort,'vila_mil','clip-rn50',0]
        cfg=read_config(reference['plan']['config'])
        manifest=rows(cfg['dataset_csv']);by_slide={r['slide_id']:r for r in manifest}
        assert len(by_slide)==len(manifest); bind(cfg['dataset_csv'])
        for shot in shots_list:
            for fold in range(5):
                lr=next(t for t in allshots['tasks'] if (t['cohort'],t['shots'],t['fold'])==(cohort,shot,fold))
                phases={}
                for phase, members in lr['memberships'].items():
                    values=[{**by_slide[m['slide_id']], 'shots':str(shot)} for m in members]
                    if sorted(membership(values))!=sorted(membership(members)): raise ValueError('LR/feature manifest mismatch')
                    path=OUTPUT/f'splits/{cohort}/{shot}shot/fold{fold}/{phase}.csv'
                    write_rows(path,values);bind(path);phases[phase]=values
                parent_task=tasks[cohort,'vila_mil','clip-rn50',fold]
                for phase in ['train','val','test']:
                    old=set(membership(rows(parent_task['splits'][phase])));new=set(membership(phases[phase]))
                    assert old==new if phase=='test' else new<=old
                check_phases(phases,shot)
                maps=donor_maps(phases,1,fold);validate_donors(phases,maps)
                parent_maps=json.loads(Path(parent_task['plan']['donor_maps']).read_text())
                assert maps['test']==parent_maps['test']
                donor=OUTPUT/f'splits/{cohort}/{shot}shot/fold{fold}/donors.json';atomic_json(donor,maps);bind(donor)
                splits[cohort,shot,fold]=(donor.parent.parent,donor,{p:len(v) for p,v in phases.items()})
    # Verify the 80 already-completed low-shot folds before excluding them.
    prior_path=OUTPUT.parent/'tcga_4_8shot_20260912_v1/campaign.json'
    prior=json.loads(prior_path.read_text());bind(prior_path)
    reused=[]
    for group in prior['groups']:
        child=json.loads(Path(group['launch']).read_text());bind(group['launch'])
        assert child['identity']==group['identity']==identity({k:v for k,v in child.items() if k!='identity'})
        for previous in child['plans']:
            complete=Path(previous['output'])/'fold_complete.json'
            record=json.loads(complete.read_text());bind(complete)
            assert record['status']=='completed' and record['launch_identity']==child['identity']
            cohort,shot,fold=previous['cohort'],previous['shots'],previous['fold']
            previous_cfg=read_config(previous['config']);bind(previous['config'])
            split,donor,_=splits[cohort,shot,fold]
            for phase in ['train','val','test']:
                previous_split=Path(previous_cfg['split_dir'])/f'fold{fold}/{phase}.csv'
                bind(previous_split)
                assert sorted(membership(rows(previous_split)))==sorted(membership(rows(split/f'fold{fold}/{phase}.csv')))
            assert json.loads(Path(previous['donor_maps']).read_text())==json.loads(donor.read_text())
            bind(previous['donor_maps'])
            reused.append({'cohort':cohort,'method':previous['method'],'encoder':previous['encoder'],
                           'shots':shot,'fold':fold,'completion':str(complete),'launch_identity':child['identity']})
    assert len(reused)==80
    plans=[];groups=[]
    for cohort,shots_list in SCOPE.items():
        for method in METHODS:
            if cohort in ['nsclc','brca'] and method in ['vila_mil','mgpath']: continue
            for encoder in ENCODERS:
                group=f'{cohort}_{method}_{encoder}'
                groups.append({'id':group,'cohort':cohort,'method':method,'encoder':encoder,'shots':shots_list})
                for shot in shots_list:
                    for fold in range(5):
                        task=tasks[cohort,method,encoder,fold];old=task['plan'];parent=parents[task['launch']]
                        spec=copy.deepcopy(parent.get('protocol',{}))
                        adapter=copy.deepcopy(old.get('adapter',spec.get('adapter')))
                        selection=copy.deepcopy(old.get('selection',spec.get('selection',{}).get(method)))
                        panel='shared_core62_v1' if cohort in ['crc','blca'] else spec['panels'][cohort]
                        split,donor,counts=splits[cohort,shot,fold]
                        root=OUTPUT/f'groups/{group}/{shot}shot'
                        out=root/f'runs/fold{fold}';native=out/'native';smoke=root/'smokes'
                        cfg=read_config(old['config']);new=copy.deepcopy(cfg)
                        # Keep all scientific/optimizer/prompt/epoch fields byte-equivalent in value.
                        new.update(shots=shot,split_dir=str(split),results_dir=str(native),
                            k_start=fold,k_end=fold+1,benchmark='pathotme_tcga_4_8shot_allmodels_20260916_v1',
                            experiment=f'pathotme_{group}_{shot}shot')
                        config=OUTPUT/f'configs/{group}_{shot}shot_fold{fold}.yaml'
                        plan={**old,'cohort':cohort,'method':method,'encoder':encoder,'fold':fold,'shots':shot,
                            'id':f'{group}_{shot}shot_fold{fold}','group':group,'config':str(config),
                            'native_reused':False,'native_dir':str(native),'source_base_config':None,
                            'smoke_checkpoint_dir':str(smoke/'native_fixture'),'smoke_output':str(smoke),
                            'donor_maps':str(donor),'output':str(out),'split_counts':counts,
                            'adapter':adapter,'selection':selection,'panel':panel,'group_output':str(root),
                            'parent_launch':task['launch'],'parent_launch_identity':task['launch_identity'],
                            'parent_config':old['config'],'parent_native_reused':old['native_reused']}
                        for stale in ['historical_native_provenance','smoke_checkpoint_sha256']:plan.pop(stale,None)
                        plan['protocol']={**spec,'shots':shot,'seed':1,'panels':{cohort:panel},'adapter':adapter,'selection':{method:selection}}
                        key=(cohort,method,encoder);timing=timings[key];proxy='same_cohort_method_encoder'
                        if not timing:
                            timing=[n for k,v in timings.items() if k[1:]==key[1:] for n in v];proxy='other_cohort_same_method_encoder'
                        if not timing:raise ValueError('No runtime evidence')
                        measured=max(timing);factor=4/3 if old['native_reused'] else 1
                        minimum=max(3600,math.ceil((1.5*measured*max(shot/16,0.5)*factor+1800)/1800)*1800)
                        maximum=max(minimum,math.ceil((2*measured*factor+3600)/1800)*1800)
                        if maximum>48*3600:raise ValueError('Estimate exceeds partition limit')
                        rss=max([n for k,v in memory.items() if k[1:]==key[1:] for n in v],default=16)
                        ram=max(32,math.ceil((1.5*rss+8)/8)*8)
                        plan['resources']={'cpus':8,'mem_gib':ram,'time_min_seconds':minimum,'time_limit_seconds':maximum,
                            'measured_max_16shot_seconds':measured,'timing_proxy':proxy,'native_reuse_adjustment':factor,
                            'measured_same_arch_encoder_maxrss_gib':rss,
                            'policy':'TimeMin=1.5*max16*max(shots/16,.5)*native_factor+30m; TimeLimit=2*max16*native_factor+60m; rounded30m. Sampled RSS with50%+8GiB margin. Not a timeout guarantee.'}
                        assets={str(donor):sha(donor),old['tme_csv']:sha(old['tme_csv'])}
                        assets.update({str(split/f'fold{fold}/{p}.csv'):sha(split/f'fold{fold}/{p}.csv') for p in ['train','val','test']})
                        contract_path=OUTPUT/f'contracts/{plan["id"]}.json'
                        atomic_json(contract_path,{'native_config':new,'plan':plan,'assets':assets});bind(contract_path)
                        new.update(lowshot_contract_path=str(contract_path),lowshot_contract_sha256=sha(contract_path))
                        config.parent.mkdir(exist_ok=True)
                        config.write_text(yaml.safe_dump(new,sort_keys=False))
                        if read_config(config)!=new:raise ValueError('Config serialization drift')
                        if any(type(read_config(config)[k]) is not float for k in ['lr','weight_decay']):raise TypeError('Optimizer values must remain floats')
                        bind(config);validate_config(new)
                        plans.append(plan)
    campaign={'scope':SCOPE,'methods':METHODS,'encoders':ENCODERS,'groups':groups,'plans':plans,
        'counts':{'fold_jobs':480,'smoke_jobs':48,'total_jobs':528,'condition_fold_results':1920},
        'policy':'Exploratory shot extension; preserve every native recipe and exact LR shot memberships. Each fold freshly fits native plus zero/actual/shuffled adapters. All outcomes retained. Full-shot explicitly excluded.',
        'baseline_manifest_identity':baseline['identity'],'allshot_manifest_identity':allshots['identity'],
        'reused_completed_lowshot_folds':reused,
        'unused_historical_checkpoint_bindings':historical_only,
        'file_sha256':bindings,'output':str(OUTPUT)}
    assert len(plans)==480 and len(groups)==48
    campaign['identity']=identity(campaign);atomic_json(OUTPUT/'campaign.json',campaign)
    print(json.dumps({'identity':campaign['identity'],'counts':campaign['counts'],'bindings':len(bindings)}),flush=True)

if __name__=='__main__':prepare()
