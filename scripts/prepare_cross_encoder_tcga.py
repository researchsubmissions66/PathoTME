#!/usr/bin/env python3
"""Prepare the missing encoder pairings on the parent's identical common folds."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym']
from common.configuration import load_dotenv,load_yaml_config
from common.run_state import validate_resume_state
from pathotme.locked_tcga import ROOT,PGVL,sha,identity,atomic_json,rows,membership,load_launch
from pathotme.cross_encoder_contract import CLIP_PORT,CLIP_TEXT_POLICY,validate_clip_config


def clip_tokens(prompts):
    # Load the native tokenizer implementation without importing Torch/weights.
    clip_spec=importlib.util.find_spec('clip')
    if clip_spec is None or clip_spec.origin is None:raise ImportError('native CLIP tokenizer unavailable')
    source=Path(clip_spec.origin).parent/'simple_tokenizer.py'
    spec=importlib.util.spec_from_file_location('pathotme_clip_token_audit',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tokenizer=module.SimpleTokenizer()
    start=tokenizer.encoder['<|startoftext|>'];end=tokenizer.encoder['<|endoftext|>']
    original=[[start,*tokenizer.encode(text),end] for text in prompts]
    consumed=[ids[:76]+[end] if len(ids)>77 else ids+[0]*(77-len(ids)) for ids in original]
    return {'original_token_counts_including_specials':[len(x) for x in original],
        'consumed_token_ids':consumed,'policy':CLIP_TEXT_POLICY,
        'tokenizer_source':str(source),'tokenizer_sha256':sha(source),
        'vocabulary_source':module.default_bpe(),'vocabulary_sha256':sha(module.default_bpe())}


def build_clip_cfg(base,clip,study,cohort):
    cfg=copy.deepcopy(base)
    for key in ['backbone','backbone_weights','encoder','feature_space_id','feature_dim','feature_sources','feature_resolutions','feature_input_kinds']:
        cfg[key]=copy.deepcopy(clip[key])
    cfg.update({'experiment':f'{study}_{cohort}_mgpath_clip_rn50','benchmark':study,
        'feature_path_column_l':clip['feature_path_column_s'],'feature_path_column_s':clip['feature_path_column_l'],
        'data_folder_l':clip['data_folder_s'],'data_folder_s':clip['data_folder_l'],
        'feature_projection':'none','prompt_feature_space_id':clip['feature_space_id'],
        'pathotme_encoder_extension':CLIP_PORT,'pathotme_text_policy':CLIP_TEXT_POLICY,
        'mgpath_runtime':CLIP_PORT,'encoder_provenance':'adapted',
        'implementation_provenance':'pathotme_extended_paired_feature_context','upstream_fidelity':'partial',
        'fidelity_note':'PathoTME-owned RN50 MGPATH port: native 1024D CLIP shared features and frozen matching text tower; four zero-initialized feature contexts replace PLIP token contexts. Native graph, 64 centers, aggregation and Sinkhorn are retained. Full prompt banks remain unchanged; explicit 77-position prefix/EOT consumption. Not upstream PLIP-G or a registered native PGVL condition.'})
    validate_clip_config(cfg)
    return cfg


def prepare(spec_path,output):
    from common.preflight import preflight
    from methods.mgpath.prompts import load_prompt_bank
    spec=json.loads(spec_path.read_text());parent=load_launch(spec['parent_launch'])
    if parent['identity']!=spec['parent_launch_identity']:raise ValueError('parent study mismatch')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol=copy.deepcopy(parent['protocol']);protocol['study']=spec['study']
    protocol['analysis']['multiplicity']='Bonferroni simultaneous 99.7916666667 percent intervals for 24 combined primary contrasts'
    protocol['analysis']['master_amendment']=spec['analysis_amendment']
    protocol['additional_pairings']=spec['additional_pairings']
    bound=dict(parent['file_sha256']);bound[spec['parent_launch']]=sha(spec['parent_launch']);bound[str(spec_path)]=sha(spec_path)
    plans=[]
    for original in parent['plans']:
        cohort,method,fold=original['cohort'],original['method'],original['fold']
        old_cfg=load_yaml_config(original['config'])
        if method=='vila_mil':
            source=PGVL/f'benchmarks/tcga_{cohort}/configs/vila_mil_plip/{cohort}_16shot.yaml'
            base=load_yaml_config(source);historical=Path(base['results_dir'])
            valid=validate_resume_state(json.loads((historical/'metrics.json').read_text()),historical/'config.json',method,base)
            record=next((r for r in valid if r['fold']==fold),None)
            if record is None or any(record.get('sample_failures',{}).values()):raise ValueError('PLIP baseline invalid')
            reusable=all(membership(rows(Path(old_cfg['split_dir'])/f'fold{fold}/{phase}.csv'))==membership(rows(Path(base['split_dir'])/f'fold{fold}/{phase}.csv')) for phase in ['train','val','test'])
            paths=[source,Path(base['text_prompt_path'])]
            if reusable:
                pred=historical/f'fold{fold}_predictions.csv'
                if sorted(membership(rows(pred)))!=sorted(membership(rows(Path(old_cfg['split_dir'])/f'fold{fold}/test.csv'))):raise ValueError('PLIP prediction membership mismatch')
                paths += [historical/'config.json',historical/'metrics.json',historical/f'fold{fold}_best.pt',pred]
            if fold==0:paths.append(historical/'fold0_best.pt')
            for p in paths:
                if str(p) not in bound:bound[str(p)]=sha(p)
            cfg=copy.deepcopy(base)
            cfg['experiment']=f"{spec['study']}_{cohort}_vila_plip";cfg['benchmark']=spec['study']
            fixture=str(historical);source_base=str(source)
        else:
            source=PGVL/f'benchmarks/tcga_{cohort}/configs/mgpath/{cohort}_16shot.yaml'
            base=load_yaml_config(source)
            clip=load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/configs/vila_mil/{cohort}_16shot.yaml')
            cfg=build_clip_cfg(base,clip,spec['study'],cohort);reusable=False;source_base=None
            fixture=str(output/f'smokes/{cohort}/mgpath/native_fixture')
            if fold==0:
                bank=load_prompt_bank(cfg['text_prompt_path'],cfg['label_dict'],cfg['prompt_class_bindings'])
                audit=clip_tokens(bank.prompts)
                atomic_json(output/f'prompt_audit/{cohort}_mgpath_clip.json',audit)
                for k in ['tokenizer_source','vocabulary_source']:bound[audit[k]]=sha(audit[k])
        cfg.update({'dataset_csv':old_cfg['dataset_csv'],'split_dir':old_cfg['split_dir'],
            'k_start':fold,'k_end':fold+1,'results_dir':str(output/f'runs/{cohort}/{method}/fold{fold}/native')})
        path=output/f'configs/{cohort}_{method}_fold{fold}.json';atomic_json(path,cfg)
        if method=='vila_mil':
            report=preflight(cfg,check_features=True)
            if not report.ok:raise ValueError(report.as_dict())
        else:validate_clip_config(cfg)
        plan={**original,'config':str(path),'native_reused':reusable,
            'native_dir':str(historical) if reusable else cfg['results_dir'],
            'source_base_config':source_base,'smoke_checkpoint_dir':fixture,
            'encoder':cfg['backbone'],'output':str(output/f'runs/{cohort}/{method}/fold{fold}'),
            'condition_provenance':'registered_PLIP_ViLa_port' if method=='vila_mil' else CLIP_PORT}
        plans.append(plan)
        print(f'prepared {cohort}/{method}/{cfg["backbone"]}/fold{fold} reused={reusable}',flush=True)
    for p in [ROOT/'pathotme/cross_encoder_contract.py',ROOT/'pathotme/cross_encoder_models.py',ROOT/'pathotme/brca_mgpath_index_repair.py',
              ROOT/'scripts/prepare_cross_encoder_tcga.py',ROOT/'scripts/run_cross_encoder_tcga.py',
              ROOT/'scripts/launch_cross_encoder_tcga.py',ROOT/'TCGA_CROSS_ENCODERS.md',ROOT/'tests/test_cross_encoder_tcga.py']:
        bound[str(p)]=sha(p)
    for p in output.rglob('*'):
        if p.is_file():bound[str(p)]=sha(p)
    launch={'protocol':protocol,'cross_encoder_spec':spec,'parent_launch_identity':parent['identity'],
        'parent_jobs':'21933657-21933680','output':str(output),'plans':plans,'file_sha256':bound,
        'counts':{'fold_jobs':20,'smoke_jobs':4,'comparisons':80,'native_reused':sum(p['native_reused'] for p in plans),
                  'native_new':sum(not p['native_reused'] for p in plans),'adapter_fits':60,'combined_slurm_jobs':48}}
    launch['identity']=identity(launch);atomic_json(output/'launch.json',launch)
    print(json.dumps({'launch':str(output/'launch.json'),'counts':launch['counts']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/tcga_cross_encoders_16shot_20260909.json')
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args();load_dotenv(PGVL/'.env')
    if args.prepare:prepare(args.config.resolve(),args.output.resolve())
    else:print(json.dumps({'would_prepare':str(args.output),'spec':json.loads(args.config.read_text())}))
