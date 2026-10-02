#!/usr/bin/env python3
"""Prepare four FOCUS groups on exact existing PathoTME folds; never submit."""
import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym']
from common.configuration import load_dotenv, load_yaml_config
from common.run_state import validate_resume_state
from common.prompts import load_focus_prompt_bank
from pathotme.focus_contract import build_config, validate_donors
from pathotme.locked_tcga import ROOT, PGVL, atomic_json, sha, identity, rows, membership, check_phases
from pathotme.yaml_runtime_config import write_runtime_config
from prepare_locked_tcga import inspect_feature


def prepare(spec_path, output):
    import numpy as np
    from pathotme.focus_tme import load_tme
    spec = json.loads(spec_path.read_text())
    if (spec['cohorts'] != ['nsclc', 'brca'] or spec['encoders'] != ['plip', 'clip-rn50']
            or spec['methods'] != ['focus'] or spec['folds'] != list(range(5)) or spec['shots'] != 16):
        raise ValueError('only the frozen four-group 16-shot boundary is supported')
    parent_path = Path(spec['parent_launch'])
    parent = json.loads(parent_path.read_text())
    parent_id = parent.pop('identity')
    if identity(parent) != parent_id or parent_id != spec['parent_launch_identity']:
        raise ValueError('parent identity mismatch')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    bound = {str(spec_path): sha(spec_path), str(parent_path): sha(parent_path)}
    def bind(path, inherited=False):
        path = str(path)
        if path not in bound:
            digest = sha(path)
            if inherited and parent['file_sha256'].get(path) != digest:
                raise ValueError(f'parent asset changed: {path}')
            bound[path] = digest
    plans, coverage, reuse_audit, token_audits = [], {}, {}, {}
    for cohort in spec['cohorts']:
        old = [p for p in parent['plans'] if p['cohort'] == cohort and p['method'] == 'vila_mil']
        old_cfg = load_yaml_config(old[0]['config'])
        manifest = Path(old_cfg['dataset_csv'])
        split_dir = Path(old_cfg['split_dir'])
        tme = Path(old[0]['tme_csv'])
        for p in (manifest, tme):
            bind(p, inherited=True)
        metadata = tme.with_suffix('.csv.metadata.json')
        if metadata.exists():
            bind(metadata, inherited=str(metadata) in parent['file_sha256'])
        transformed = load_tme(tme, cohort)
        frame = rows(manifest)
        coverage[cohort] = {'slides': len(frame), 'patients': len({r['case_id'] for r in frame})}
        protocol_path = PGVL / f'benchmarks/tcga_{cohort}/protocol.yaml'
        protocol = load_yaml_config(protocol_path)
        base_path = PGVL / f'benchmarks/tcga_{cohort}/configs/focus_plip/{cohort}_16shot.yaml'
        clip_path = PGVL / f'benchmarks/tcga_{cohort}/configs/vila_mil/{cohort}_16shot.yaml'
        base, clip = load_yaml_config(base_path), load_yaml_config(clip_path)
        for p in (protocol_path, base_path, clip_path, base['text_prompt_path'], PGVL/'text_prompts/PROVENANCE.json'):
            bind(p)
        bank = load_focus_prompt_bank(base['text_prompt_path'], class_names=list(base['label_dict']),
            file_class_names=base['focus_prompt_file_classnames'], expected_provenance=base['prompt_provenance'],
            expected_file_sha256=base['focus_prompt_file_sha256'],
            expected_ordered_prompt_bank_sha256=base['focus_prompt_bank_sha256'])
        from prepare_cross_encoder_tcga import clip_tokens
        audit = clip_tokens(bank.prompts)
        if max(audit['original_token_counts_including_specials']) > 77:
            raise ValueError('FOCUS original bank exceeds native RN50 context; no truncation allowed')
        # Both towers use CLIP BPE but each tokenizer is validated separately.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base['backbone_weights'], local_files_only=True)
        ids = tokenizer(list(bank.prompts), padding=False, truncation=False)['input_ids']
        if max(map(len, ids)) > 77:
            raise ValueError('FOCUS original bank exceeds PLIP context')
        audit['policy'] = 'unchanged_full_bank_native_tokenization_no_truncation_v1'
        audit['plip_counts_including_specials'] = list(map(len, ids))
        audit['plip_token_ids_unpadded'] = ids
        token_audits[cohort] = audit
        for k in ('tokenizer_source', 'vocabulary_source'):
            bind(audit[k])
        historical = Path(base['results_dir'])
        valid, historical_error = [], None
        try:
            valid = validate_resume_state(json.loads((historical/'metrics.json').read_text()),
                                          historical/'config.json', 'focus', base)
        except (ValueError, FileNotFoundError) as error:
            historical_error = str(error)
        for encoder in spec['encoders']:
            key = 'plip_20x' if encoder == 'plip' else 'clip_rn50_20x'
            source = protocol['feature_sources'][key]
            cfg_template = build_config(base, clip, source, spec['study'], cohort, encoder)
            weights = Path(cfg_template['backbone_weights'])
            if weights.is_dir():
                weight_files = sorted(p for p in weights.rglob('*') if p.is_file())
                if not weight_files:
                    raise ValueError('empty paired checkpoint directory')
                for p in weight_files:
                    bind(p)
            else:
                bind(weights)
            inventory = {}
            tasks = [(os.path.expandvars(r[f'feature__{key}']), key.rsplit('_20x', 1)[0],
                      cfg_template['feature_dim'], 20, source) for r in frame]
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i, (path, item) in enumerate(pool.map(inspect_feature, tasks), 1):
                    inventory[path] = item
                    if i % 500 == 0 or i == len(tasks):
                        print(f'{cohort}/{encoder}: {i}/{len(tasks)} feature headers checked', flush=True)
            inventory_path = output / f'features/{cohort}_{encoder}.json'
            atomic_json(inventory_path, inventory)
            for fold in spec['folds']:
                original = next(p for p in old if p['fold'] == fold)
                phases = {phase: rows(split_dir/f'fold{fold}/{phase}.csv') for phase in ('train', 'val', 'test')}
                check_phases(phases, transformed.index)
                donors = Path(original['donor_maps'])
                validate_donors(phases, json.loads(donors.read_text()))
                for p in [donors, *[split_dir/f'fold{fold}/{phase}.csv' for phase in phases]]:
                    bind(p, inherited=True)
                values = transformed.loc[[r['slide_id'] for r in phases['train']]].to_numpy(dtype=float)
                if np.isinf(values).any() or np.isnan(values).all(0).any():
                    raise ValueError('invalid train-fold normalization inputs')
                record = next((r for r in valid if r['fold'] == fold), None)
                reusable = (encoder == 'plip' and record is not None
                    and not any(record.get('sample_failures', {}).values())
                    and all(membership(phases[p]) == membership(rows(Path(base['split_dir'])/f'fold{fold}/{p}.csv')) for p in phases)
                    and (historical/f'fold{fold}_best.pt').is_file())
                if reusable:
                    pred = historical/f'fold{fold}_predictions.csv'
                    if sorted(membership(rows(pred))) != sorted(membership(phases['test'])):
                        raise ValueError('historical predictions do not match test membership')
                    for p in [historical/'config.json', historical/'metrics.json', historical/f'fold{fold}_best.pt', pred]:
                        bind(p)
                out = output/f'runs/{cohort}/{encoder}/fold{fold}'
                cfg = {**deepcopy(cfg_template), 'dataset_csv': str(manifest), 'split_dir': str(split_dir),
                       'k_start': fold, 'k_end': fold+1, 'results_dir': str(out/'native')}
                path = output/f'configs/{cohort}_{encoder}_fold{fold}.yaml'
                write_runtime_config(path, cfg)
                plans.append(dict(cohort=cohort, method='focus', encoder=encoder, fold=fold, config=str(path),
                    native_reused=reusable, native_dir=str(historical if reusable else out/'native'),
                    source_base_config=str(base_path) if reusable else None,
                    smoke_checkpoint_dir=str(output/f'smokes/{cohort}/{encoder}/native_fixture'),
                    tme_csv=str(tme), donor_maps=str(donors), feature_inventory=str(inventory_path),
                    output=str(out), split_counts={p: len(v) for p, v in phases.items()},
                    historical_native_provenance='config/artifact bound; historical producer source revision not independently reattested' if reusable else None))
                print(f'prepared {cohort}/{encoder}/fold{fold}; native_reused={reusable}', flush=True)
            reuse_audit[cohort] = {'historical_validation_error': historical_error}
    atomic_json(output/'prompt_audit.json', token_audits)
    # Bind current imported code; ancestor runtime is not implicitly re-attested.
    # All relevant package files are bound conservatively to cover dynamic imports.
    for directory in (ROOT/'pathotme', PGVL/'common', PGVL/'methods', PGVL/'clip'):
        for p in directory.rglob('*.py'):
            bind(p)
    for p in [PGVL/'train.py', ROOT/'scripts/run_vila_guided.py', ROOT/'scripts/prepare_locked_tcga.py',
              ROOT/'scripts/prepare_cross_encoder_tcga.py', ROOT/'TME_GUIDED_FOCUS.md', ROOT/'tests/test_focus_tme.py',
              *sorted((ROOT/'scripts').glob('*focus_tcga.py'))]:
        bind(p)
    for p in output.rglob('*'):
        if p.is_file():
            bind(p)
    launch = dict(protocol=spec, output=str(output), plans=plans, coverage=coverage,
                  historical_reuse_audit=reuse_audit, parent_launch_identity=parent_id, file_sha256=bound,
                  counts=dict(smoke_jobs=4, fold_jobs=20, comparisons=80, adapter_fits=60,
                              native_reused=sum(p['native_reused'] for p in plans),
                              native_new=sum(not p['native_reused'] for p in plans)))
    launch['identity'] = identity(launch)
    atomic_json(output/'launch.json', launch)
    print(json.dumps({'launch': str(output/'launch.json'), 'counts': launch['counts'], 'coverage': coverage}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/focus_tcga_16shot_20260911.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    load_dotenv(PGVL/'.env')
    if args.prepare:
        prepare(args.config.resolve(), args.output.resolve())
    else:
        print(json.dumps({'would_prepare': str(args.output), 'spec': json.loads(args.config.read_text())}))
