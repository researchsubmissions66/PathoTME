#!/usr/bin/env python3
"""Seal a prepared, unsubmitted study with tests and cached encoder digests."""
import argparse
import json
import importlib.metadata
from pathlib import Path
import sys

sys.path[:0] = ['/path/to/PathoTME', '/path/to/PGVL-Gym']
from pathotme.locked_tcga import ROOT, PGVL, sha, identity, atomic_json, load_launch, verify_files
from common.configuration import load_dotenv, load_yaml_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    args=parser.parse_args();load_dotenv(PGVL/'.env')
    launch=load_launch(args.launch)
    output=Path(launch['output']);sealed=output/'launch.sealed.json'
    if sealed.exists() or (output/'submission.json').exists():
        raise FileExistsError('preserve existing seal/submission')
    validation=json.loads(args.validation.read_text())
    if validation['status']!='passed':raise ValueError('tests must pass')
    verify_files(validation['source_sha256'])
    paths=[args.launch.resolve(),args.validation.resolve(),ROOT/'README.md',Path(__file__).resolve(),
           PGVL/'methods/mscpt/dataset.py',PGVL/'text_prompts/PROVENANCE.json']
    paths+=list((ROOT/'tests').glob('test_*vila*.py'))
    paths += [ROOT/'tests/test_locked_tcga.py',ROOT/'tests/test_guided_mgpath.py',ROOT/'tests/test_brca_features.py']
    for cohort in ['nsclc','brca']:
        for method in ['vila_mil','mgpath']:
            cfg=load_yaml_config(PGVL/f'benchmarks/tcga_{cohort}/configs/{method}/{cohort}_16shot.yaml')
            weights=Path(cfg['backbone_weights'])
            paths += sorted(p for p in weights.rglob('*') if p.is_file()) if weights.is_dir() else [weights]
    nsclc_meta=Path('/path/to/shared/PathoTME-data/processed/opentme_nsclc_core62.csv.metadata.json')
    from pathotme.features import SCHEMA_SHA256
    meta=json.loads(nsclc_meta.read_text())
    if (meta['selected_table_sha256']!=sha(nsclc_meta.with_name('opentme_nsclc_core62.csv'))
            or meta['feature_schema_sha256']!=SCHEMA_SHA256
            or meta['source_revision']!='9262bc0cd0cd7774d0f7bcbfe6ae5f4665898b25'):
        raise ValueError('NSCLC TME provenance mismatch')
    paths.append(nsclc_meta)
    for p in paths:
        if str(p) not in launch['file_sha256']:
            launch['file_sha256'][str(p)]=sha(p)
    launch['prepared_parent_identity']=launch.pop('identity')
    launch['validation_path']=str(args.validation.resolve())
    launch['runtime_versions']={name:importlib.metadata.version(name) for name in
        ['torch','transformers','numpy','pandas','scikit-learn','h5py','scipy']}
    launch['seal_note']='pre-submission seal adds focused tests, documentation and exact cached encoder files; scientific plans unchanged'
    launch['identity']=identity(launch)
    atomic_json(sealed,launch)
    print(json.dumps({'sealed_launch':str(sealed),'identity':launch['identity'],'counts':launch['counts']}))


if __name__=='__main__':main()
