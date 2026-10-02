#!/usr/bin/env python3
"""Combine completed feature scores while retaining variant/panel/coverage identity."""
import argparse,csv,json,shutil
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from score_core import aggregate_ranks,write_csv,dump,sha
from render_scores import render_report

def main():
    p=argparse.ArgumentParser();p.add_argument('--linear',type=Path,required=True);p.add_argument('--zero',type=Path)
    p.add_argument('--neural',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=[];sources={};coverage=[]
    for directory in [a.linear,*([a.zero] if a.zero else []),*a.neural]:
        path=directory/'fold_feature_scores.csv'
        # A partial neural directory with only a plan must not enter the summary.
        if directory in a.neural:
            record=json.loads((directory/'attribution.json').read_text())
            if record['status']!='completed' or sha(directory/'scores.npz')!=record['scores_sha256']:raise ValueError('Incomplete or changed neural score output')
            coverage.append({k:record[k] for k in ['variant','cohort','method','encoder','fold','patients','slides','expected_fold_patients','expected_fold_slides','full_test_patients']})
        with path.open(newline='') as f:part=list(csv.DictReader(f))
        for r in part:
            for k in ['shots','fold','class_index','patients','slides']:r[k]=int(r[k])
            for k in ['mean_absolute_pp','mean_signed_pp','positive_patient_fraction','negative_patient_fraction']:r[k]=float(r[k])
            r['rank_in_fold']=int(r['rank_in_fold']) if r['rank_in_fold'] else None
            r['full_test_patients']=bool(record['full_test_patients']) if directory in a.neural else r.get('full_test_patients','True')=='True'
            r.pop('classes',None)
            rows.append(r)
        sources[str(path)]=sha(path)
    # Reorder dictionary fields consistently before CSV writing.
    columns=list(rows[0]);rows=[{k:r[k] for k in columns} for r in rows]
    ranked=aggregate_ranks(rows);write_csv(a.output/'fold_feature_scores.csv',rows);write_csv(a.output/'feature_rankings.csv',ranked)
    for name in ['lr_coefficients.csv','lr_coefficient_stability.csv']:
        shutil.copyfile(a.linear/name,a.output/name)
    summary=json.loads((a.linear/'summary.json').read_text())
    conditions={(r['variant'],r['cohort'],r['method'],r['encoder'],r['fold']) for r in rows}
    summary.update(recorded_at_utc=datetime.now(timezone.utc).isoformat(),completed_variant_folds=dict(Counter(r[0] for r in conditions)),
                   neural_adapter_scores_included=bool(a.neural),neural_coverage=coverage,rank_rows=len(ranked),source_score_sha256=sources)
    summary['artifact_sha256']={p.name:sha(p) for p in a.output.glob('*.csv')}
    dump(a.output/'summary.json',summary);dump(a.output/'neural_coverage.json',coverage)
    render_report(a.output)
    print(json.dumps({'completed_variant_folds':summary['completed_variant_folds'],'neural_coverage':coverage},indent=2))

if __name__=='__main__':main()
