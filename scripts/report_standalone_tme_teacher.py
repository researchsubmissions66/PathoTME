#!/usr/bin/env python3
"""Report new students against hash-validated reused controls, without fitting."""
import argparse
import json
from pathlib import Path

from run_standalone_tme_teacher import (ROOT, CONDITIONS, NEW_CONDITIONS, environment,
    load_yaml_config, preflight, sha, verify_completed)


def matched_pair(left, right):
    """Require identical slides, patient IDs and labels before any comparison."""
    a = left.set_index('slide_id', verify_integrity=True).sort_index()
    b = right.set_index('slide_id', verify_integrity=True).sort_index()
    if not a.index.equals(b.index) or not a[['case_id','label']].equals(b[['case_id','label']]):
        raise ValueError('comparison cohorts or labels differ')
    return a.reset_index(), b.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=ROOT/'configs/vila_conch_brca_tme_teacher_16shot.yaml')
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(); environment()
    if args.output.exists():
        raise FileExistsError('preserve previous report')
    import numpy as np
    import pandas as pd
    from common.statistical_analysis import aggregate_patients, extended_metrics, paired_bootstrap_differences
    contract = load_yaml_config(args.config); matched = load_yaml_config(contract['matched_contract'])
    frames = {}; ensembles = {}; status = []; scores = []; intervals = []; hashes = {}
    for condition in CONDITIONS:
        source = []; external = []
        for fold in contract['folds']:
            _, _, _, identity, root = preflight(contract, fold, condition)
            record = verify_completed(root, identity)
            status.append({'condition': condition, 'fold': fold, 'identity': identity, 'completed': bool(record),
                           'reused': condition not in NEW_CONDITIONS, 'output': str(root)})
            if not record:
                continue
            path = root/f'fold{fold}_predictions.csv'; data = pd.read_csv(path); data['fold'] = fold
            hashes[str(path)] = sha(path); source.append(data)
            scores.append({'domain':'TCGA_source_fold','condition':condition,'fold':fold,
                           **extended_metrics(aggregate_patients(data))})
            external_root = Path(contract['results_root'] if condition in NEW_CONDITIONS else matched['results_root'])
            directory = external_root/'external/cptac_brca'/condition/f'fold{fold}'
            if (directory/'metrics.json').exists():
                from eval_wsi_only import preflight as external_preflight
                *_, external_identity = external_preflight(root, matched['external']['contract'])
                verify_completed(directory, external_identity)
                path = directory/'predictions.csv'; ext = pd.read_csv(path)
                hashes[str(path)] = sha(path); external.append(ext)
                scores.append({'domain':'CPTAC_source_fold','condition':condition,'fold':fold,
                               **extended_metrics(aggregate_patients(ext))})
        if len(source) == len(contract['folds']):
            data = pd.concat(source, ignore_index=True)
            if data['slide_id'].duplicated().any() or (data.groupby('case_id')['fold'].nunique() > 1).any():
                raise ValueError('OOF patient/slide reuse')
            frames[condition] = data
            scores.append({'domain':'TCGA_pooled_OOF','condition':condition,
                           **extended_metrics(aggregate_patients(data))})
        if len(external) == len(contract['folds']):
            ordered = [matched_pair(external[0], d)[1] for d in external]
            data = ordered[0].copy(); columns = ['probability_0','probability_1']
            data[columns] = np.mean([d[columns].to_numpy() for d in ordered], axis=0)
            data['prediction'] = data[columns].to_numpy().argmax(1); ensembles[condition] = data
            scores.append({'domain':'CPTAC_equal_weight_ensemble','condition':condition,
                           **extended_metrics(aggregate_patients(data))})
    for domain, data in [('TCGA_pooled_OOF',frames), ('CPTAC_equal_weight_ensemble',ensembles)]:
        for reference in ('student_ce','kd_visual','kd_tme_only_shuffled'):
            if reference in data and 'kd_tme_only' in data:
                a, b = matched_pair(data[reference], data['kd_tme_only'])
                table = paired_bootstrap_differences(a, b, replicates=args.bootstrap, seed=1)
                table['domain'] = domain; table['reference_condition'] = reference
                table['candidate_condition'] = 'kd_tme_only'; intervals.append(table)
    args.output.mkdir(parents=True)
    pd.DataFrame(status).to_csv(args.output/'completion.csv', index=False)
    pd.DataFrame(scores).to_csv(args.output/'metrics.csv', index=False)
    if intervals:
        pd.concat(intervals, ignore_index=True).to_csv(args.output/'paired_patient_bootstrap.csv', index=False)
    (args.output/'provenance.json').write_text(json.dumps({'source_sha256': hashes, 'reporter_sha256': sha(__file__),
        'note': 'Patient-level metrics. Fold means and pooled OOF scores are distinct. Only complete matched conditions compared. External folds share patients. Unadjusted exploratory bootstrap conditional on fitted models. Both test cohorts previously inspected.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
