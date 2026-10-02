#!/usr/bin/env python3
"""Export exact LR/fusion TME feature scores from completed saved fits on CPU."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from pathotme.shared_panel import common_spec, transform_rows as common_transform
from pathotme.brca_features import panel_spec, transform_rows as breast_transform
from score_core import VERSION, sha, identity, dump, write_csv, sigmoid, linear_scores, patient_scores, rank_fold, aggregate_ranks, make_units


def load(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError(f'Input hash mismatch: {path}')


def join_prediction(path, expected, label_dict):
    rows = csv_rows(path)
    if len(rows) != len(expected) or len({r['slide_id'] for r in rows}) != len(rows):
        raise ValueError('Prediction membership length/duplicates')
    by_id = {r['slide_id']: r for r in rows}
    ordered = []
    for e in expected:
        r = by_id[e['slide_id']]
        label = r.get('label_id', r.get('label'))
        if r['case_id'] != e['case_id'] or int(label_dict.get(label, label)) != e['label']:
            raise ValueError('Prediction label/patient mismatch')
        ordered.append(r)
    return ordered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); start = time.monotonic()
    out = a.output.expanduser().resolve()
    if any(out == r or r in out.parents for r in [ROOT, ROOT.parent/'PGVL-Gym']):
        raise ValueError('Keep patient-level attribution outside source repositories')
    out.mkdir(parents=True, exist_ok=False)
    m = load(a.manifest); base = a.manifest.parent
    if identity({k:v for k,v in m.items() if k!='identity'}) != m['identity']:
        raise ValueError('Baseline manifest identity mismatch')
    naming_path=ROOT/'docs/data/variant-names.json'
    names={k:v['display_name'] for k,v in load(naming_path)['variants'].items()}
    # Read-only source audit. No model/checkpoint construction, refit, or training.
    verify(m['source_sha256'])
    provenance={'manifest':str(a.manifest), 'manifest_sha256':sha(a.manifest), 'manifest_identity':m['identity'],
                'source_sha256':{str(f):sha(f) for f in Path(__file__).parent.glob('*.py')},
                'naming_sha256':sha(naming_path), 'score_version':VERSION,
                'reference':'zero after saved training-fold standardization; transformed training mean',
                'selection':'held-out features only; all features/groups retained; no fitting or test selection',
                'aggregation':'average signed slide scores within patient, absolute patient scores within fold, equal fold means',
                'uncertainty':'sample SD of fold means, not a confidence interval',
                'fusion_rule':'exact TME feature score = fixed saved alpha times LR score; native predictions held fixed',
                'spatial':'slide-level TME feature scores do not localize measurements in a thumbnail'}
    dump(out/'provenance.json',provenance)
    ranks=[]; coeffs=[]; cache={}; coverage=[]; expected_test_patients={}
    for task in m['tme_tasks']:
        folder=base/'tme_only'/task['id']; result=load(folder/'metrics.json')
        if result['status']!='completed' or result['identity']!=task['id']:
            raise ValueError('Incomplete/mismatched LR result')
        verify(result['artifact_sha256']); verify(task['input_sha256'])
        model=load(folder/'model.json')
        if model['task']!=task or model['classes']!=[0,1] or model['feature_names']!=task['feature_names']:
            raise ValueError('Stored LR model/task binding mismatch')
        spec=panel_spec(task['panel']) if task['panel'].startswith('brca_') else common_spec()
        feature_names=list(spec['feature_names'])
        if feature_names!=task['feature_names']: raise ValueError('Panel feature order mismatch')
        groups=[{'name':scale+'/'+name,'indices':[feature_names.index(c) for c in columns]}
                for scale,key in [('low','low_tokens'),('high','high_tokens')]
                for name,columns in spec[key].items()]
        units=make_units(feature_names,groups)
        test=task['memberships']['test']
        classes=[name for name,_ in sorted(task['label_dict'].items(), key=lambda x:x[1])]
        # Frozen split bytes are verified above; compare explicit memberships too.
        for phase in ['train','val','test']:
            rows=csv_rows(task['splits'][phase]); members=[]
            for row in rows:
                label=row.get('label_id',row.get('label'))
                members.append({'slide_id':row['slide_id'],'case_id':row['case_id'],'label':int(task['label_dict'].get(label,label))})
            if sorted(members,key=lambda r:r['slide_id'])!=task['memberships'][phase]:raise ValueError('Frozen split membership changed')
        for phase in ['train','val']:
            if {r['case_id'] for r in test}&{r['case_id'] for r in task['memberships'][phase]}:raise ValueError('Patient leakage')
        for case in {r['case_id'] for r in test}:
            key=(task['cohort'],task['panel'],case)
            if expected_test_patients.setdefault(key,task['fold'])!=task['fold']:raise ValueError('Patient appears in multiple test folds')
        source=csv_rows(task['tme_csv']); by_id={r['slide_id']:r for r in source}
        if len(by_id)!=len(source):raise ValueError('Duplicate TME slide')
        selected=[by_id[r['slide_id']] for r in test]
        values=np.asarray(breast_transform(selected,task['panel']) if task['panel'].startswith('brca_') else common_transform(selected),float)
        s=model['scaler']; z=(np.where(np.isnan(values),s['median'],values)-np.asarray(s['mean']))/np.asarray(s['scale'])
        probability,scores,log_contributions=linear_scores(z,model['coef'],model['intercept'],units)
        predictions=join_prediction(folder/'test_predictions.csv',test,task['label_dict'])
        reference=np.asarray([float(r['probability_1']) for r in predictions])
        error=float(np.max(np.abs(probability-reference)))
        if not np.allclose(probability,reference,atol=1e-10,rtol=1e-10):raise ValueError('LR prediction replay mismatch')
        meta={'variant':names['tme_only'],'cohort':task['cohort'],'panel':task['panel'],'method':'none','encoder':'none','shots':16,'fold':task['fold'],'classes':classes}
        task_out=out/'lr'/task['id'];task_out.mkdir(parents=True)
        cases,patient_labels,patient_values=patient_scores(test,scores)
        np.savez_compressed(task_out/'scores.npz',slide_id=np.asarray([r['slide_id'] for r in test]),case_id=np.asarray([r['case_id'] for r in test]),
                            label=np.asarray([r['label'] for r in test]),score_pp=scores,patient_id=np.asarray(cases),patient_label=patient_labels,
                            patient_score_pp=patient_values,feature_log_odds_contribution=log_contributions,probability_1=probability)
        record={**meta,'status':'completed','units':units,'classes':classes,'model_sha256':sha(folder/'model.json'),
                'lr_id':task['id'],'scores_sha256':sha(task_out/'scores.npz'),'prediction_replay_max_error':error,
                'slides':len(test),'patients':len(cases),'standardized_feature_values_exported':False}
        dump(task_out/'attribution.json',record)
        rr=rank_fold(meta,test,scores,units);ranks.extend(rr);write_csv(task_out/'ranking.csv',rr)
        for name,coef in zip(feature_names,model['coef']):
            coeffs.append({'cohort':task['cohort'],'panel':task['panel'],'fold':task['fold'],'feature_name':name,
                           'positive_class':classes[1],'standardized_coefficient':float(coef)})
        cache[task['id']]={'rows':test,'scores':scores,'probability':probability,'units':units,'meta':meta,'folder':task_out,'record':record}
        coverage.append({'variant':names['tme_only'],'cohort':task['cohort'],'method':'none','encoder':'none','fold':task['fold'],'status':'completed','slides':len(test)})
        print(json.dumps({'LR':task['cohort'],'fold':task['fold'],'slides':len(test),'max_replay_error':error}),flush=True)
    for task in m['fusion_tasks']:
        folder=base/'fusion'/task['name']; path=folder/'metrics.json'
        if not path.exists():
            coverage.append({'variant':'fusion_pair','cohort':task['cohort'],'method':task['method'],'encoder':task['encoder'],'fold':task['fold'],'status':'waiting_saved_fusion','slides':0});continue
        result=load(path)
        if result['status']!='completed' or result['task_id']!=task['id'] or result['tme_id']!=task['tme_id']:raise ValueError('Fusion task binding mismatch')
        verify(result['artifact_sha256'])
        lr=cache[task['tme_id']]; rows=join_prediction(folder/'predictions.csv',lr['rows'],task['label_dict'])
        alpha=float(result['selected_alpha'])
        if alpha not in [0,.25,.5,.75,1]:raise ValueError('Unexpected frozen fusion alpha')
        native=np.asarray([float(r['native_probability_1']) for r in rows])
        if not np.isfinite(native).all() or np.any((native<0)|(native>1)):raise ValueError('Native probabilities invalid')
        records=[]
        for condition,weight in [('native',0.),('late_fusion_fixed_half',.5),('late_fusion_validation',alpha)]:
            expected=(1-weight)*native+weight*lr['probability']
            actual=np.asarray([float(r[condition+'_probability_1']) for r in rows])
            if not np.allclose(expected,actual,atol=1e-10,rtol=1e-10):raise ValueError('Saved fused probabilities mismatch')
            meta={**lr['meta'],'variant':names[condition],'method':task['method'],'encoder':task['encoder']}
            ranks.extend(rank_fold(meta,lr['rows'],weight*lr['scores'],lr['units']))
            records.append({**meta,'alpha':weight,'score_reference':str(lr['folder']/'scores.npz'),'score_multiplier':weight,
                            'score_reference_sha256':lr['record']['scores_sha256'],
                            'definition':'Multiply LR score_pp and patient_score_pp arrays by this fixed alpha; do not rescale LR log-odds contributions.'})
            coverage.append({'variant':names[condition],'cohort':task['cohort'],'method':task['method'],'encoder':task['encoder'],'fold':task['fold'],'status':'completed','slides':len(lr['rows'])})
        dump(out/'fusion'/task['name']/'attribution.json',{'status':'completed','task_id':task['id'],'saved_fusion_metrics_sha256':sha(path),'variants':records})
    aggregate=aggregate_ranks(ranks)
    write_csv(out/'fold_feature_scores.csv',ranks);write_csv(out/'feature_rankings.csv',aggregate);write_csv(out/'lr_coefficients.csv',coeffs)
    from collections import defaultdict,Counter
    coefficient_groups=defaultdict(list)
    for r in coeffs:coefficient_groups[(r['cohort'],r['panel'],r['feature_name'],r['positive_class'])].append(r['standardized_coefficient'])
    write_csv(out/'lr_coefficient_stability.csv',[{'cohort':k[0],'panel':k[1],'feature_name':k[2],'positive_class':k[3],
              'mean_standardized_coefficient':float(np.mean(v)),'fold_sd_coefficient':float(np.std(v,ddof=1)) if len(v)>1 else None,
              'positive_coefficient_folds':sum(x>0 for x in v),'negative_coefficient_folds':sum(x<0 for x in v),'folds':len(v)} for k,v in coefficient_groups.items()])
    dump(out/'coverage.json',coverage)
    summary={'status':'completed','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
             'completed_variant_folds':dict(Counter(r['variant'] for r in coverage if r['status']=='completed')),
             'waiting_fusion_folds':sum(r['status']=='waiting_saved_fusion' for r in coverage),
             'rank_rows':len(aggregate),'neural_adapter_scores_included':False,
             'all_LR_features_retained':True,'training':False,'GPU_jobs':0,
             'artifact_sha256':{p.name:sha(p) for p in out.glob('*.csv')}}
    dump(out/'summary.json',summary)
    from render_scores import render_report
    render_report(out)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
