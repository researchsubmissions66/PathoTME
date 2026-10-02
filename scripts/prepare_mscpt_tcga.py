#!/usr/bin/env python3
"""Prepare exactly four MSCPT groups on unchanged PathoTME cohorts and folds."""
import argparse,json,os,shutil,sys
from copy import deepcopy
from pathlib import Path
sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym']
from common.configuration import load_dotenv,load_yaml_config
from pathotme.mscpt_contract import build_config,TEXT_POLICY
from pathotme.hive_geometry import array_sha
from pathotme.focus_contract import validate_donors
from pathotme.locked_tcga import ROOT,PGVL,atomic_json,sha,identity,rows,check_phases,verify_files
from pathotme.yaml_runtime_config import write_runtime_config


def prompt_audit(cfg,bind):
    source=json.loads(Path(cfg['description_prompt_path']).read_text())
    bank={c:source[c] for c in cfg['label_dict']}
    bind(cfg['description_prompt_path'])
    cfg['description_prompt_sha256']=sha(cfg['description_prompt_path'])
    prefix=' '.join(['X']*12)
    strings=[text for branch in bank.values() for text in branch['small_mag']]
    strings += [f'{prefix} {text}' for branch in bank.values() for text in branch['big_mag']]
    if any(len(b['small_mag'])!=10 or not b['big_mag'] for b in bank.values()):
        raise ValueError('MSCPT scale description dimensions changed')
    if cfg.get('selection_prompt_path'):
        bind(cfg['selection_prompt_path']);cfg['selection_prompt_sha256']=sha(cfg['selection_prompt_path'])
        selectors=json.loads(Path(cfg['selection_prompt_path']).read_text())
        for index in sorted(selectors,key=int):
            for c in bank:
                strings.extend(t.replace('CLASSNAME',selectors[index]['classnames'][c]) for t in selectors[index]['templates'])
    if cfg['backbone']=='plip':
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(cfg['backbone_weights'],local_files_only=True)
        original=tokenizer(strings,padding=False,truncation=False)['input_ids']
        consumed=tokenizer(strings,padding='max_length',truncation=True,max_length=64)['input_ids']
    else:
        from clip.simple_tokenizer import SimpleTokenizer
        tokenizer=SimpleTokenizer()
        original=[[49406,*tokenizer.encode(text),49407] for text in strings]
        consumed=[(ids[:63]+[49407] if len(ids)>64 else ids+[0]*(64-len(ids))) for ids in original]
        bind(PGVL/'clip/simple_tokenizer.py');bind(PGVL/'clip/bpe_simple_vocab_16e6.txt.gz')
    return dict(policy=TEXT_POLICY,classnames=list(bank),
        descriptions_per_class={c:{k:len(v) for k,v in b.items()} for c,b in bank.items()},
        original_token_counts_including_specials=list(map(len,original)),
        consumed_ids_sha256=array_sha(consumed),source_strings_sha256=identity(strings),
        truncated_rows=sum(len(ids)>64 for ids in original),
        frozen_selector=bool(cfg.get('selection_prompt_path')),
        note='Original description and selector bytes retained; native MSCPT 64-position prefix/EOT truncation is disclosed.')


def prepare(spec_path,output,audit_root):
    from pathotme.muse_data import load_tme
    import numpy as np
    spec=json.loads(spec_path.read_text())
    if (spec['cohorts']!=['nsclc','brca'] or spec['encoders']!=['plip','clip-rn50']
            or spec['methods']!=['mscpt'] or spec['folds']!=list(range(5)) or spec['shots']!=16):
        raise ValueError('only the fixed four-group MSCPT addition is supported')
    parent_path=Path(spec['parent_launch']);parent=json.loads(parent_path.read_text())
    parent_id=parent.pop('identity')
    if identity(parent)!=parent_id or parent_id!=spec['parent_launch_identity']:
        raise ValueError('parent identity mismatch')
    audit=json.loads((audit_root/'audit_complete.json').read_text())
    if (audit['status']!='passed' or audit['parent_identity']!=parent_id
            or audit['helper_sha256']!=sha(ROOT/'pathotme/hive_geometry.py')):
        raise ValueError('completed matching hierarchy audit required')
    verify_files({str(audit_root/k):v for k,v in audit['group_sha256'].items()})
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
    bind(spec_path);bind(parent_path)
    (output/'features').mkdir()
    shutil.copyfile(audit_root/'audit_complete.json',output/'features/audit_complete.json')
    plans,coverage,token_audits=[],{},{}
    for cohort in spec['cohorts']:
        old=[p for p in parent['plans'] if p['cohort']==cohort and p['method']=='vila_mil']
        old_cfg=load_yaml_config(old[0]['config'])
        manifest,split_dir,tme=Path(old_cfg['dataset_csv']),Path(old_cfg['split_dir']),Path(old[0]['tme_csv'])
        bind(manifest,True);bind(tme,True)
        frame=rows(manifest);transformed=load_tme(tme,cohort)
        coverage[cohort]=dict(slides=len(frame),patients=len({r['case_id'] for r in frame}))
        base_path=PGVL/f'benchmarks/tcga_{cohort}/configs/mscpt_5x20x/{cohort}_16shot.yaml'
        base=load_yaml_config(base_path);bind(base_path)
        protocol_path=PGVL/f'benchmarks/tcga_{cohort}/protocol.yaml';bind(protocol_path)
        protocol=load_yaml_config(protocol_path)
        for encoder in spec['encoders']:
            key=encoder.replace('-','_')
            paired_path=PGVL/f'benchmarks/tcga_{cohort}/configs/muse_paired_{key}/{cohort}_16shot.yaml'
            paired=load_yaml_config(paired_path);bind(paired_path)
            cfg_template=build_config(base,paired,spec['study'],cohort,encoder)
            prompt=prompt_audit(cfg_template,bind);token_audits[f'{cohort}/{encoder}']=prompt
            cfg_template['mscpt_token_ids_sha256']=prompt['consumed_ids_sha256']
            weights=Path(cfg_template['backbone_weights'])
            for path in sorted(weights.rglob('*')) if weights.is_dir() else [weights]:
                if path.is_file():bind(path)
            group_path=audit_root/f'{cohort}_{key}.json';group=json.loads(group_path.read_text())
            if group['manifest_sha256']!=sha(manifest) or set(group['slides'])!={r['slide_id'] for r in frame}:
                raise ValueError('hierarchy audit changed the common cohort')
            # Pin both coordinate-bearing feature stores to the declared encoder.
            for scale in [5,20]:
                source=protocol['feature_sources'][f'{key}_{scale}x']
                if source['feature_space_id']!=cfg_template['feature_space_id']:
                    raise ValueError('registry feature space differs from paired MSCPT config')
            for item in group['slides'].values():
                for path,stat in item['files'].items():
                    observed=Path(path).stat()
                    if (observed.st_size,observed.st_mtime_ns)!=(stat['size'],stat['mtime_ns']):
                        raise ValueError(f'feature changed after audit: {path}')
            inventory_path=output/f'features/{cohort}_{key}.json';shutil.copyfile(group_path,inventory_path)
            for fold in spec['folds']:
                original=next(p for p in old if p['fold']==fold)
                phases={phase:rows(split_dir/f'fold{fold}/{phase}.csv') for phase in ['train','val','test']}
                check_phases(phases,transformed.index)
                donors=Path(original['donor_maps']);validate_donors(phases,json.loads(donors.read_text()))
                for path in [donors,*[split_dir/f'fold{fold}/{p}.csv' for p in phases]]:bind(path,True)
                values=transformed.loc[[r['slide_id'] for r in phases['train']]].to_numpy(dtype=float)
                if np.isinf(values).any() or np.isnan(values).all(0).any():raise ValueError('invalid train-only TME normalization')
                out=output/f'runs/{cohort}/{encoder}/fold{fold}'
                cfg={**deepcopy(cfg_template),'dataset_csv':str(manifest),'split_dir':str(split_dir),
                     'k_start':fold,'k_end':fold+1,'results_dir':str(out/'native')}
                path=output/f'configs/{cohort}_{key}_fold{fold}.yaml';write_runtime_config(path,cfg)
                plans.append(dict(cohort=cohort,method='mscpt',encoder=encoder,fold=fold,config=str(path),
                    native_reused=False,native_dir=str(out/'native'),source_base_config=None,
                    smoke_checkpoint_dir=str(output/f'smokes/{cohort}/{encoder}/native_fixture'),
                    tme_csv=str(tme),donor_maps=str(donors),feature_inventory=str(inventory_path),
                    output=str(out),split_counts={p:len(v) for p,v in phases.items()},
                    native_reuse_exclusion='Existing PGVL folds use a different cohort and/or scale/port; fit all twenty matched common-cohort native controls'))
            print(f'Prepared {cohort}/{encoder}: five native folds and fifteen adapters',flush=True)
    atomic_json(output/'prompt_audit.json',token_audits)
    for directory in [ROOT/'pathotme',PGVL/'common',PGVL/'methods',PGVL/'clip']:
        for path in directory.rglob('*.py'):bind(path)
    for path in [PGVL/'train.py',ROOT/'scripts/run_vila_guided.py',ROOT/'scripts/prepare_cross_encoder_tcga.py',
                 ROOT/'TME_GUIDED_HIVE.md',ROOT/'tests/test_mscpt_tme.py',
                 *sorted((ROOT/'scripts').glob('*mscpt_tcga.py'))]:bind(path)
    for path in output.rglob('*'):
        if path.is_file():bind(path)
    launch=dict(protocol=spec,output=str(output),plans=plans,coverage=coverage,
        parent_launch_identity=parent_id,file_sha256=bound,
        counts=dict(smoke_jobs=4,fold_jobs=20,comparisons=80,adapter_fits=60,native_reused=0,native_new=20))
    launch['identity']=identity(launch);atomic_json(output/'launch.json',launch)
    print(json.dumps({'launch':str(output/'launch.json'),'counts':launch['counts'],'coverage':coverage}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/mscpt_tcga_16shot_20260914.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hierarchy-audit',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');args=p.parse_args()
    load_dotenv(PGVL/'.env')
    if args.prepare:prepare(args.config.resolve(),args.output.resolve(),args.hierarchy_audit.resolve())
    else:print(json.dumps({'would_prepare':str(args.output),'spec':json.loads(args.config.read_text())}))
