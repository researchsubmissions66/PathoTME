#!/usr/bin/env python3
"""Prepare four MUSE groups on the parent's exact TCGA folds; never submit."""
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
from common.prompts.muse import load_muse_prompt_bank
from pathotme.muse_contract import build_config
from pathotme.focus_contract import validate_donors
from pathotme.locked_tcga import ROOT, PGVL, atomic_json, sha, identity, rows, membership, check_phases
from pathotme.yaml_runtime_config import write_runtime_config
from prepare_locked_tcga import inspect_feature


def prompt_audit(base, bind):
    """Bind unchanged source strings and each tower's registered consumption."""
    provenance = json.loads((PGVL/'text_prompts/PROVENANCE.json').read_text())['assets']
    records = {path: provenance[str(Path(path).relative_to(PGVL/'text_prompts'))]
               for path in base['prompt_csvs'].values()}
    if any(r['provenance'] != 'upstream' for r in records.values()):
        raise ValueError('original upstream MUSE banks required')
    bank = load_muse_prompt_bank(base['prompt_csvs'], classnames=base['classnames'], records=records)
    descriptions = [text for branch in bank.descriptions for text in branch]
    for path in base['prompt_csvs'].values():
        bind(path)
    template_path = PGVL/'text_prompts/muse/class_template.json'
    bind(template_path)
    template = json.loads(template_path.read_text())['template']
    class_texts = [template.format(name) for name in bank.classnames]
    if base['backbone'] == 'clip-rn50':
        from prepare_cross_encoder_tcga import clip_tokens
        audit = clip_tokens(descriptions)
        class_audit = clip_tokens(class_texts)
        for key in ('tokenizer_source', 'vocabulary_source'):
            bind(audit[key])
        class_counts = class_audit['original_token_counts_including_specials']
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base['backbone_weights'], local_files_only=True)
        original = tokenizer(descriptions, padding=False, truncation=False)['input_ids']
        consumed = tokenizer(descriptions, padding=False, truncation=True, max_length=77)['input_ids']
        if any(len(ids) > 77 or ids[-1] != tokenizer.eos_token_id for ids in consumed):
            raise ValueError('PLIP EOT-preserving context policy failed')
        audit = {'original_token_counts_including_specials': list(map(len, original)),
                 'consumed_token_ids': consumed}
        class_counts = list(map(len, tokenizer(class_texts, padding=False, truncation=False)['input_ids']))
    if max(class_counts) > 77:
        raise ValueError('inference class template exceeds native context')
    audit.update(policy=base['muse_text_context_policy'], classnames=list(bank.classnames),
                 class_template=str(template_path), class_template_token_counts=class_counts,
                 rows_per_class=[branch.row_count for branch in bank.branches],
                 prompt_csvs=records,
                 ordered_loaded_description_sha256=identity(bank.descriptions),
                 truncated_description_rows=sum(n > 77 for n in audit['original_token_counts_including_specials']),
                 note='Native registered 77-token prefix/EOT consumption; source CSV rows are unchanged. '
                      'Descriptions serve only native training retrieval; inference uses class-template semantics.')
    return audit


def prepare(spec_path, output):
    import numpy as np
    from pathotme.muse_data import load_tme
    spec = json.loads(spec_path.read_text())
    if (spec['cohorts'] != ['nsclc', 'brca'] or spec['encoders'] != ['plip', 'clip-rn50']
            or spec['methods'] != ['muse'] or spec['folds'] != list(range(5)) or spec['shots'] != 16):
        raise ValueError('only the fixed four-group 16-shot addition is supported')
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
        manifest, split_dir, tme = Path(old_cfg['dataset_csv']), Path(old_cfg['split_dir']), Path(old[0]['tme_csv'])
        for path in (manifest, tme):
            bind(path, inherited=True)
        metadata = tme.with_suffix('.csv.metadata.json')
        if metadata.exists():
            bind(metadata, inherited=str(metadata) in parent['file_sha256'])
        transformed = load_tme(tme, cohort)
        frame = rows(manifest)
        coverage[cohort] = {'slides': len(frame), 'patients': len({r['case_id'] for r in frame})}
        protocol_path = PGVL/f'benchmarks/tcga_{cohort}/protocol.yaml'
        protocol = load_yaml_config(protocol_path)
        bind(protocol_path)
        bind(PGVL/'text_prompts/PROVENANCE.json')
        for encoder in spec['encoders']:
            base_path = PGVL/f'benchmarks/tcga_{cohort}/configs/muse_paired_{encoder.replace("-", "_")}/{cohort}_16shot.yaml'
            base = load_yaml_config(base_path)
            bind(base_path)
            token_audits[f'{cohort}/{encoder}'] = prompt_audit(base, bind)
            key = 'plip_10x' if encoder == 'plip' else 'clip_rn50_10x'
            source = protocol['feature_sources'][key]
            cfg_template = build_config(base, spec['study'], cohort, encoder)
            weights = Path(cfg_template['backbone_weights'])
            weight_files = sorted(p for p in weights.rglob('*') if p.is_file()) if weights.is_dir() else [weights]
            if not weight_files:
                raise ValueError('empty paired checkpoint directory')
            for path in weight_files:
                bind(path)
            tasks = [(os.path.expandvars(r[f'feature__{key}']), key.rsplit('_10x', 1)[0],
                      cfg_template['feature_dim'], 10, source) for r in frame]
            inventory = {}
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i, (path, item) in enumerate(pool.map(inspect_feature, tasks), 1):
                    inventory[path] = item
                    if i % 500 == 0 or i == len(tasks):
                        print(f'{cohort}/{encoder}: {i}/{len(tasks)} feature headers checked', flush=True)
            inventory_path = output/f'features/{cohort}_{encoder}.json'
            atomic_json(inventory_path, inventory)
            historical = Path(base['results_dir'])
            valid, historical_error = [], None
            try:
                valid = validate_resume_state(json.loads((historical/'metrics.json').read_text()),
                                              historical/'config.json', 'muse', base)
            except (ValueError, RuntimeError, FileNotFoundError) as error:
                historical_error = str(error)
            fold_audit = []
            for fold in spec['folds']:
                original = next(p for p in old if p['fold'] == fold)
                phases = {phase: rows(split_dir/f'fold{fold}/{phase}.csv') for phase in ('train', 'val', 'test')}
                check_phases(phases, transformed.index)
                donors = Path(original['donor_maps'])
                validate_donors(phases, json.loads(donors.read_text()))
                for path in [donors, *[split_dir/f'fold{fold}/{phase}.csv' for phase in phases]]:
                    bind(path, inherited=True)
                values = transformed.loc[[r['slide_id'] for r in phases['train']]].to_numpy(dtype=float)
                if np.isinf(values).any() or np.isnan(values).all(0).any():
                    raise ValueError('invalid train-fold normalization inputs')
                record = next((r for r in valid if r['fold'] == fold), None)
                same_splits = all(membership(phases[p]) == membership(rows(Path(base['split_dir'])/f'fold{fold}/{p}.csv')) for p in phases)
                reusable = (record is not None and not any(record.get('sample_failures', {}).values())
                            and same_splits and (historical/f'fold{fold}_best.pt').is_file())
                if reusable:
                    pred = historical/f'fold{fold}_predictions.csv'
                    if sorted(membership(rows(pred))) != sorted(membership(phases['test'])):
                        raise ValueError('historical predictions do not match common test membership')
                    for path in [historical/'config.json', historical/'metrics.json', historical/f'fold{fold}_best.pt', pred]:
                        bind(path)
                out = output/f'runs/{cohort}/{encoder}/fold{fold}'
                cfg = {**deepcopy(cfg_template), 'dataset_csv': str(manifest), 'split_dir': str(split_dir),
                       'k_start': fold, 'k_end': fold+1, 'results_dir': str(out/'native')}
                path = output/f'configs/{cohort}_{encoder}_fold{fold}.yaml'
                write_runtime_config(path, cfg)
                plans.append(dict(cohort=cohort, method='muse', encoder=encoder, fold=fold, config=str(path),
                    native_reused=reusable, native_dir=str(historical if reusable else out/'native'),
                    source_base_config=str(base_path) if reusable else None,
                    smoke_checkpoint_dir=str(output/f'smokes/{cohort}/{encoder}/native_fixture'),
                    tme_csv=str(tme), donor_maps=str(donors), feature_inventory=str(inventory_path),
                    output=str(out), split_counts={p: len(v) for p, v in phases.items()},
                    historical_native_provenance='config/artifact bound; historical producer source revision not independently reattested' if reusable else None))
                fold_audit.append({'fold': fold, 'matching_splits': same_splits,
                                   'complete_record': record is not None, 'reused': reusable})
                print(f'prepared {cohort}/{encoder}/fold{fold}; native_reused={reusable}', flush=True)
            reuse_audit[f'{cohort}/{encoder}'] = {'historical_validation_error': historical_error, 'folds': fold_audit}
    atomic_json(output/'prompt_audit.json', token_audits)
    for directory in (ROOT/'pathotme', PGVL/'common', PGVL/'methods', PGVL/'clip'):
        for path in directory.rglob('*.py'):
            bind(path)
    for path in [PGVL/'train.py', ROOT/'scripts/run_vila_guided.py', ROOT/'scripts/prepare_locked_tcga.py',
                 ROOT/'scripts/prepare_cross_encoder_tcga.py', ROOT/'TME_GUIDED_MUSE.md', ROOT/'tests/test_muse_tme.py',
                 *sorted((ROOT/'scripts').glob('*muse_tcga.py'))]:
        bind(path)
    for path in output.rglob('*'):
        if path.is_file():
            bind(path)
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
    parser.add_argument('--config', type=Path, default=ROOT/'configs/muse_tcga_16shot_20260912.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    load_dotenv(PGVL/'.env')
    if args.prepare:
        prepare(args.config.resolve(), args.output.resolve())
    else:
        print(json.dumps({'would_prepare': str(args.output), 'spec': json.loads(args.config.read_text())}))
