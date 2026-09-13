#!/usr/bin/env python3
"""Prepare the exact two-cohort/two-encoder DyKo addition, without submitting."""
import argparse,json,os,shutil,sys
from copy import deepcopy
from pathlib import Path
sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym']
from common.configuration import load_dotenv,load_yaml_config
from methods.dyko.prompts import load_prompt_bank
from pathotme.dyko_contract import build_config
from pathotme.focus_contract import validate_donors
from pathotme.locked_tcga import ROOT,PGVL,atomic_json,sha,identity,rows,membership,check_phases,verify_files
from pathotme.yaml_runtime_config import write_runtime_config


def prepare(spec_path,output):
    from pathotme.muse_data import load_tme
    import numpy as np
    spec=json.loads(spec_path.read_text())
    if (spec['cohorts']!=['nsclc','brca'] or spec['encoders']!=['plip','clip-rn50']
            or spec['methods']!=['dyko'] or spec['folds']!=list(range(5)) or spec['shots']!=16):
        raise ValueError('only the fixed four-group DyKo addition is supported')
    parent_path=Path(spec['parent_launch']);parent=json.loads(parent_path.read_text())
    parent_id=parent.pop('identity')
    if identity(parent)!=parent_id or parent_id!=spec['parent_launch_identity']:
        raise ValueError('parent identity mismatch')
    feature_parent_path=Path(spec['feature_audit_launch'])
    feature_parent=json.loads(feature_parent_path.read_text());feature_id=feature_parent.pop('identity')
    if identity(feature_parent)!=feature_id or feature_id!=spec['feature_audit_launch_identity']:
        raise ValueError('feature audit parent identity mismatch')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    bound={}
    def bind(path,inherited=False):
        path=str(path)
        if path not in bound:
            digest=sha(path)
            if inherited and parent['file_sha256'].get(path)!=digest:
                raise ValueError(f'parent asset changed: {path}')
            bound[path]=digest
    for path in [spec_path,parent_path,feature_parent_path]:bind(path)
    assets=spec['brca_assets'];encoding=json.loads(Path(assets['encoding_path']).read_text())
    if encoding['status']!='completed' or encoding['shape']!=[64,768]:
        raise ValueError('successful real BRCA TITAN encoding required')
    verify_files(encoding['source_sha256'])
    for path in encoding['source_sha256']:bind(path)
    verify_files({assets[k]:assets[k.replace('_path','_sha256')]
                  for k in ('class_prompt_path','tensor_path','source_bank_path','encoding_path')})
    for key in ('class_prompt_path','tensor_path','source_bank_path','encoding_path'):bind(assets[key])
    if assets['tensor_sha256']!=encoding['tensor_sha256']:
        raise ValueError('BRCA tensor differs from encoding record')
    for path in (ROOT/'text_prompts/dyko_brca_v1').rglob('*'):
        if path.is_file():bind(path)
    source_bank=json.loads(Path(assets['source_bank_path']).read_text())
    for item in source_bank['sources'].values():
        if 'path' in item:
            verify_files({item['path']:item['sha256']});bind(item['path'])
    plans,coverage,token_audits,reuse_audit=[],{},{},{}
    base_path=PGVL/'benchmarks/tcga_nsclc/configs/dyko_plip/nsclc_16shot.yaml'
    base=load_yaml_config(base_path);bind(base_path)
    for cohort in spec['cohorts']:
        old=[p for p in parent['plans'] if p['cohort']==cohort and p['method']=='vila_mil']
        old_cfg=load_yaml_config(old[0]['config'])
        manifest,split_dir,tme=Path(old_cfg['dataset_csv']),Path(old_cfg['split_dir']),Path(old[0]['tme_csv'])
        bind(manifest,True);bind(tme,True)
        frame=rows(manifest);transformed=load_tme(tme,cohort)
        coverage[cohort]=dict(slides=len(frame),patients=len({r['case_id'] for r in frame}))
        protocol_path=PGVL/f'benchmarks/tcga_{cohort}/protocol.yaml';bind(protocol_path)
        protocol=load_yaml_config(protocol_path)
        paired_path=PGVL/f'benchmarks/tcga_{cohort}/configs/muse_paired_clip_rn50/{cohort}_16shot.yaml'
        paired=load_yaml_config(paired_path);bind(paired_path)
        for encoder in spec['encoders']:
            key=encoder.replace('-','_')+'_20x';source=protocol['feature_sources'][key]
            cfg_template=build_config(base,paired,source,spec['study'],cohort,encoder,assets)
            bank=load_prompt_bank(cfg_template['text_prompt_path'],cfg_template['label_dict'],
                cfg_template['prompt_file_classnames'],cfg_template['prompt_class_bindings'])
            if bank.file_sha256!=cfg_template['text_prompt_file_sha256']:
                raise ValueError('class prompt hash mismatch')
            bind(cfg_template['text_prompt_path']);bind(cfg_template['concept_feature_path'])
            from prepare_cross_encoder_tcga import clip_tokens
            audit=clip_tokens(bank.descriptions)
            if max(audit['original_token_counts_including_specials'])>77:
                raise ValueError('class bank must encode in full; no added token context')
            for k in ('tokenizer_source','vocabulary_source'):bind(audit[k])
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(base['backbone_weights'],local_files_only=True)
            ids=tokenizer(list(bank.descriptions),padding=False,truncation=False)['input_ids']
            if max(map(len,ids))>77:raise ValueError('PLIP prompt overflow')
            audit.update(policy='unchanged_class_text_full_native_encoding_feature_context_v1',
                class_order=list(cfg_template['label_dict']),plip_token_ids_unpadded=ids,
                plip_counts_including_specials=list(map(len,ids)))
            token_audits[f'{cohort}/{encoder}']=audit
            weights=Path(cfg_template['backbone_weights'])
            for path in sorted(weights.rglob('*')) if weights.is_dir() else [weights]:
                if path.is_file():bind(path)
            audit_plan=next(p for p in feature_parent['plans'] if p['cohort']==cohort and p['encoder']==encoder)
            audit_path=Path(audit_plan['feature_inventory'])
            if sha(audit_path)!=feature_parent['file_sha256'][str(audit_path)]:
                raise ValueError('prior single-scale feature audit changed')
            bind(audit_path)
            inventory=json.loads(audit_path.read_text())
            selected_inventory={}
            for row in frame:
                path=os.path.expandvars(row[f'feature__{key}']);item=inventory[path]
                st=Path(path).stat()
                if (st.st_size,st.st_mtime_ns)!=(item['size'],item['mtime_ns']):
                    raise ValueError(f'feature changed since full header audit: {path}')
                selected_inventory[path]=item
            inventory_path=output/f'features/{cohort}_{encoder}.json';atomic_json(inventory_path,selected_inventory)
            for fold in spec['folds']:
                original=next(p for p in old if p['fold']==fold)
                phases={phase:rows(split_dir/f'fold{fold}/{phase}.csv') for phase in ['train','val','test']}
                check_phases(phases,transformed.index)
                donors=Path(original['donor_maps']);validate_donors(phases,json.loads(donors.read_text()))
                for path in [donors,*[split_dir/f'fold{fold}/{p}.csv' for p in phases]]:bind(path,True)
                values=transformed.loc[[r['slide_id'] for r in phases['train']]].to_numpy(dtype=float)
                if np.isinf(values).any() or np.isnan(values).all(0).any():raise ValueError('invalid training TME values')
                if cohort=='nsclc' and encoder=='plip':
                    matches={p:membership(phases[p])==membership(rows(Path(base['split_dir'])/f'fold{fold}/{p}.csv')) for p in phases}
                    reuse_audit[str(fold)]=matches
                    if all(matches.values()):
                        raise ValueError('historical baseline might be reusable; review before requesting new fits')
                out=output/f'runs/{cohort}/{encoder}/fold{fold}'
                cfg={**deepcopy(cfg_template),'dataset_csv':str(manifest),'split_dir':str(split_dir),
                     'k_start':fold,'k_end':fold+1,'results_dir':str(out/'native')}
                path=output/f'configs/{cohort}_{encoder}_fold{fold}.yaml';write_runtime_config(path,cfg)
                plans.append(dict(cohort=cohort,method='dyko',encoder=encoder,fold=fold,config=str(path),
                    native_reused=False,native_dir=str(out/'native'),source_base_config=None,
                    smoke_checkpoint_dir=str(output/f'smokes/{cohort}/{encoder}/native_fixture'),
                    tme_csv=str(tme),donor_maps=str(donors),feature_inventory=str(inventory_path),
                    output=str(out),split_counts={p:len(v) for p,v in phases.items()},
                    native_reuse_exclusion='NSCLC PLIP historical splits differ; RN50 and BRCA are new PathoTME extensions'))
            print(f'Prepared {cohort}/{encoder}: five native folds and fifteen adapters',flush=True)
    atomic_json(output/'prompt_audit.json',token_audits)
    for directory in [ROOT/'pathotme',PGVL/'common',PGVL/'methods',PGVL/'clip']:
        for path in directory.rglob('*.py'):bind(path)
    for path in [PGVL/'train.py',ROOT/'scripts/run_vila_guided.py',ROOT/'scripts/prepare_cross_encoder_tcga.py',
                 ROOT/'README.md',ROOT/'tests/test_dyko_tme.py',
                 ROOT/'scripts/build_dyko_brca_knowledge.py',*sorted((ROOT/'scripts').glob('*dyko_tcga.py'))]:bind(path)
    for path in output.rglob('*'):
        if path.is_file():bind(path)
    launch=dict(protocol=spec,output=str(output),plans=plans,coverage=coverage,
        parent_launch_identity=parent_id,file_sha256=bound,historical_nsclc_plip_membership_matches=reuse_audit,
        counts=dict(smoke_jobs=4,fold_jobs=20,comparisons=80,adapter_fits=60,native_reused=0,native_new=20))
    launch['identity']=identity(launch);atomic_json(output/'launch.json',launch)
    print(json.dumps({'launch':str(output/'launch.json'),'counts':launch['counts'],'coverage':coverage}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/dyko_tcga_16shot_20260912.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    load_dotenv(PGVL/'.env')
    if a.prepare:prepare(a.config.resolve(),a.output.resolve())
    else:print(json.dumps({'would_prepare':str(a.output),'spec':json.loads(a.config.read_text())}))
