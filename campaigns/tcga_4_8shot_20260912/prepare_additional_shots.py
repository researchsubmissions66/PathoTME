"""Prepare exactly the user-selected 4/8-shot ViLa/MGPATH follow-up."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym']
from shot_contract import SHOTS,COHORTS,phase_subset,write_rows,tag_for
from common.configuration import load_dotenv,load_yaml_config
from common.run_state import validate_resume_state
from pathotme.locked_tcga import ROOT,PGVL,atomic_json,identity,sha,rows,membership,verify_files
from pathotme.controlled_vila import donor_maps
from pathotme.yaml_runtime_config import write_runtime_config

REPORT=PGVL/'benchmarks/pathotme_completed_results_20260912.json'
OUTPUT=Path('/path/to/shared/PathoTME-results/tcga_4_8shot_20260912_v1')
STUDY='pathotme_tcga_4_8shot_20260912_v1'
IGNORED_CONFIG={'dataset_csv','split_dir','results_dir','experiment','benchmark','k_start','k_end'}


def read_config(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.suffix=='.json' else load_yaml_config(path)


def compare_recipes(actual,expected):
    return {k:[actual.get(k),expected.get(k)] for k in set(actual)|set(expected)
            if k not in IGNORED_CONFIG and actual.get(k)!=expected.get(k)}


def prepare():
    load_dotenv(PGVL/'.env')
    if OUTPUT.exists():raise FileExistsError('preserve existing campaign; inspect its ledger')
    OUTPUT.mkdir(parents=True)
    report=json.loads(REPORT.read_text())
    assert report['fold_allocations_complete']==40 and report['condition_fold_results']==160
    parent_launches={}
    for item in report['launch_evidence']:
        assert sha(item['path'])==item['sha256']
        launch=json.loads(Path(item['path']).read_text());expected=launch.pop('identity')
        assert identity(launch)==expected==item['identity'];launch['identity']=expected
        parent_launches[expected]=(Path(item['path']),launch)
    groups={(g['cohort'],g['method'],g['encoder']):g for g in report['groups']}
    assert len(groups)==8 and all(len(g['folds'])==5 for g in groups.values())
    common={str(REPORT):sha(REPORT)}
    for path,launch in parent_launches.values():common[str(path)]=sha(path)
    sources=[PGVL/'train.py',PGVL/'scripts/tcga_benchmark.py',
             ROOT/'repairs/muse_smoke_margin_20260912/muse_smoke_checks.py']
    for directory in [ROOT/'pathotme',ROOT/'scripts',PGVL/'common',PGVL/'methods',Path(__file__).parent]:
        sources.extend(directory.rglob('*.py'))
    for path in set(sources):
        digest=sha(path)
        historical={l['file_sha256'][str(path)] for p,l in parent_launches.values() if str(path) in l['file_sha256']}
        if historical and digest not in historical:raise ValueError(f'completed runtime source changed: {path}')
        common[str(path)]=digest
    completion_records=[]
    for group in groups.values():
        for fold in group['folds']:
            path=Path(fold['completion_path'])
            if sha(path)!=fold['completion_sha256']:raise ValueError('completed fold record changed')
            complete=json.loads(path.read_text())
            assert complete['status']=='completed' and complete['plan']==fold['plan']
            common[str(path)]=sha(path)
            completion_records.append(str(path))
    splits={};statistics=[];feature_paths={}
    for cohort in COHORTS:
        reference=groups[cohort,'vila_mil','clip-rn50']['folds'][0]['plan']
        cfg=read_config(reference['config']);manifest=rows(cfg['dataset_csv'])
        common[cfg['dataset_csv']]=sha(cfg['dataset_csv'])
        common[reference['feature_inventory']]=sha(reference['feature_inventory'])
        inventory=json.loads(Path(reference['feature_inventory']).read_text())
        feature_paths.update(inventory)
        table=rows(reference['tme_csv']);columns=list(table[0])[1:]
        assert len(columns)==(62 if cohort=='nsclc' else 64)
        tme={r['slide_id']:[float(r[k]) if r[k] else float('nan') for k in columns] for r in table}
        common[reference['tme_csv']]=sha(reference['tme_csv'])
        for fold in range(5):
            parent={phase:rows(Path(cfg['split_dir'])/f'fold{fold}/{phase}.csv') for phase in ['train','val','test']}
            for shot in SHOTS:
                phases=phase_subset(manifest,parent,cohort,fold,shot)
                location=OUTPUT/f'splits/{cohort}/{shot}shot/fold{fold}'
                for phase,values in phases.items():
                    path=location/f'{phase}.csv';write_rows(path,values);common[str(path)]=sha(path)
                    assert phase=='test' or {r['slide_id'] for r in values}<={r['slide_id'] for r in parent[phase]}
                assert membership(phases['test'])==membership(parent['test'])
                donors=donor_maps(phases,1,fold)
                donor_path=location/'donors.json';atomic_json(donor_path,donors);common[str(donor_path)]=sha(donor_path)
                values=np.asarray([tme[r['slide_id']] for r in phases['train']],dtype=float)
                assert not np.isinf(values).any() and not np.isnan(values).all(axis=0).any()
                statistics.append({'cohort':cohort,'shots':shot,'fold':fold,'training_rows':len(values),
                    'columns':len(columns),'no_all_missing_training_column':True,
                    'donor_path':str(donor_path),'test_membership_unchanged':True,'original_16shot_sampling_reproduced':True})
                splits[cohort,shot,fold]=(location,phases,donor_path)
    for path,item in feature_paths.items():
        stat=Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(item['size'],item['mtime_ns']):raise ValueError(f'feature changed: {path}')
    print(json.dumps({'split_groups':len(splits),'reused_feature_headers_and_current_stats':len(feature_paths)}),flush=True)
    raw_timing=json.loads(Path(args.walltimes).read_text())
    common[str(Path(args.walltimes).resolve())]=sha(args.walltimes)
    timings={r['job_id']:r for r in raw_timing['jobs']}
    children=[];reuse_records=[];all_plans=[]
    for tag in ['native','cross']:
        for shot in SHOTS:
            output=OUTPUT/f'{tag}/{shot}shot';plans=[];bound=dict(common)
            reference_group=groups['nsclc','vila_mil','clip-rn50' if tag=='native' else 'plip']
            parent_path,parent=parent_launches[reference_group['folds'][0]['launch_identity']]
            spec=deepcopy(parent['protocol']);spec.update(study=STUDY,shots=shot)
            spec['analysis']={**spec['analysis'],'status':'exploratory_shot_followup_after_all_16shot_outcomes_seen',
                'reporting':'all_sixteen_new_cohort_model_encoder_shot_groups_and_all_48_new_actual_minus_control_contrasts_including_null_negative_results',
                'multiplicity':'48 new contrasts; 72 across ViLa/MGPATH at 4,8,16 shots; account for additional architectures if jointly analyzed',
                'external_evaluation':False,'inference_requires_TME':True}
            for cohort in COHORTS:
                for method in ['vila_mil','mgpath']:
                    encoder=('clip-rn50' if method=='vila_mil' else 'plip') if tag=='native' else ('plip' if method=='vila_mil' else 'clip-rn50')
                    group=groups[cohort,method,encoder]
                    max_wall=max(int(timings[f['slurm_job_id']]['elapsed_seconds']) for f in group['folds'])
                    seconds=max(3600,math.ceil((max_wall*max(.35,shot/16)*1.5+900)/1800)*1800)
                    for entry in group['folds']:
                        fold=entry['fold'];old=entry['plan'];base=read_config(old['config'])
                        location,phases,donor_path=splits[cohort,shot,fold]
                        out=output/f'runs/{cohort}/{method}/fold{fold}'
                        config={**base,'shots':shot,'split_dir':str(location.parent),
                            'results_dir':str(out/'native'),'benchmark':STUDY,
                            'experiment':f'{STUDY}_{cohort}_{method}_{encoder}_{shot}shot',
                            'k_start':fold,'k_end':fold+1}
                        assert type(config['lr']) is float and type(config['weight_decay']) is float
                        config_path=output/f'configs/{cohort}_{method}_fold{fold}.yaml'
                        write_runtime_config(config_path,config)
                        assert load_yaml_config(config_path)==config
                        bound[str(config_path)]=sha(config_path);bound[old['config']]=sha(old['config'])
                        smoke_plan=group['folds'][0]['plan']
                        fixture=Path(smoke_plan['native_dir'])/'fold0_best.pt'
                        fixture_completion=json.loads(Path(group['folds'][0]['completion_path']).read_text())
                        assert str(fixture) in fixture_completion['native_sha256']
                        fixture_hash=fixture_completion['native_sha256'][str(fixture)]
                        if str(fixture) not in bound:
                            assert sha(fixture)==fixture_hash;bound[str(fixture)]=fixture_hash
                        plan={k:deepcopy(old[k]) for k in ['cohort','method','fold','tme_csv','feature_inventory']}
                        plan.update(encoder=encoder,shots=shot,config=str(config_path),native_reused=False,
                            native_dir=config['results_dir'],source_base_config=None,
                            smoke_checkpoint_dir=str(fixture.parent),smoke_checkpoint_sha256=fixture_hash,
                            donor_maps=str(donor_path),output=str(out),
                            parent_completed_launch=entry['launch_identity'],parent_completed_fold=entry['completion_path'],
                            split_counts={p:len(v) for p,v in phases.items()},
                            time_limit_seconds=seconds,walltime_basis={'completed_16shot_max_seconds':max_wall,
                                'formula':'ceil_30min(max_wall * max(0.35, shots/16) * 1.5 + 15min), minimum 1h'})
                        # RN50 MGPATH has no registered native PGVL baseline.
                        source_name='vila_mil_plip' if method=='vila_mil' and encoder=='plip' else method
                        candidate=PGVL/f'benchmarks/tcga_{cohort}/configs/{source_name}/{cohort}_{shot}shot.yaml'
                        reason='no registered RN50 MGPATH native baseline'
                        if not (method=='mgpath' and encoder=='clip-rn50'):
                            candidate_cfg=load_yaml_config(candidate)
                            candidate_root=Path(candidate_cfg['results_dir'])
                            drift=compare_recipes(config,candidate_cfg)
                            matched=all(membership(phases[p])==membership(rows(Path(candidate_cfg['split_dir'])/f'fold{fold}/{p}.csv')) for p in phases)
                            reason='different common-cohort splits' if not matched else ('recipe mismatch: '+','.join(sorted(drift)) if drift else 'missing/incomplete native artifacts')
                            if matched and not drift and (candidate_root/'metrics.json').exists():
                                valid=validate_resume_state(json.loads((candidate_root/'metrics.json').read_text()),candidate_root/'config.json',method,candidate_cfg)
                                record=next((x for x in valid if x['fold']==fold),None)
                                prediction=candidate_root/f'fold{fold}_predictions.csv'
                                if record is not None and not any(record.get('sample_failures',{}).values()) and sorted(membership(rows(prediction)))==sorted(membership(phases['test'])):
                                    plan.update(native_reused=True,native_dir=str(candidate_root),source_base_config=str(candidate))
                                    reason='exact recipe and ordered train/val/test membership; complete native checkpoint/metrics/predictions'
                                    for p in [candidate,candidate_root/'config.json',candidate_root/'metrics.json',candidate_root/f'fold{fold}_best.pt',prediction]:
                                        if str(p) not in bound:bound[str(p)]=sha(p)
                        reuse_records.append({'cohort':cohort,'method':method,'encoder':encoder,'shots':shot,'fold':fold,'reused':plan['native_reused'],'reason':reason})
                        for key in ['text_prompt_path','backbone_weights']:
                            asset=Path(base[key]);assets=sorted(p for p in asset.rglob('*') if p.is_file()) if asset.is_dir() else [asset]
                            for p in assets:
                                if str(p) not in bound:bound[str(p)]=sha(p)
                        plans.append(plan);all_plans.append(plan)
                    print(json.dumps({'prepared': [cohort,method,encoder,shot],
                        'reused':sum(p['native_reused'] for p in plans if p['cohort']==cohort and p['method']==method),
                        'fold_limit_minutes':seconds//60}),flush=True)
            spec['resources']['smoke_time']='02:00:00'
            spec['job_packaging']='eight campaign smoke allocations each covering 4 and 8 shots, then eighty single-shot fold allocations'
            launch={'protocol':spec,'output':str(output),'plans':plans,'file_sha256':bound,
                'counts':{'fold_jobs':20,'smoke_reports':4,'native_reused':sum(p['native_reused'] for p in plans),
                          'native_new':sum(not p['native_reused'] for p in plans),'adapter_fits':60,'comparisons':80},
                'runtime_amendment':{'kind':'4_8shot_followup_with_existing_brca_index_and_numeric_repairs',
                    'parent_source':str(parent_path),'training_loop':'unchanged_native_and_cross_encoder_loops',
                    'native_reuse_validation':'shot-aware source config lookup replaces prior 16-shot-only lookup',
                    'smoke_policy':'binary decision margin and nonzero finite TME gradient; 16shot native checkpoint is a structural fixture only'}}
            launch['identity']=identity(launch);path=output/'launch.json';atomic_json(path,launch)
            children.append({'tag':tag,'shots':shot,'launch':str(path),'identity':launch['identity']})
    preparation={'recorded_at':datetime.now(timezone.utc).isoformat(),'split_audit':statistics,
        'feature_files_verified_unchanged':len(feature_paths),'reuse':reuse_records,
        'completed_parent_folds_verified':len(completion_records),'walltimes':raw_timing}
    atomic_json(OUTPUT/'preparation.json',preparation)
    campaign={'recorded_at':datetime.now(timezone.utc).isoformat(),'study':STUDY,'output':str(OUTPUT),
        'authorization':['submit other shots for completed ViLa/MGPATH','User selected 4 and 8'],
        'shots':SHOTS,'cohorts':COHORTS,'methods':['vila_mil','mgpath'],'encoders':['clip-rn50','plip'],
        'groups':children,'counts':{'smoke_jobs':8,'fold_jobs':80,'condition_fold_results':320,
            'native_reused':sum(p['native_reused'] for p in all_plans),
            'native_new':sum(not p['native_reused'] for p in all_plans),'adapter_fits':240},
        'file_sha256':{str(OUTPUT/'preparation.json'):sha(OUTPUT/'preparation.json'),
                       **{g['launch']:sha(g['launch']) for g in children},
                       **{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}}}
    campaign['identity']=identity(campaign);atomic_json(OUTPUT/'prepared_campaign.json',campaign)
    print(json.dumps({'prepared_identity':campaign['identity'],'counts':campaign['counts']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walltimes',type=Path,required=True)
    args=parser.parse_args();prepare()
