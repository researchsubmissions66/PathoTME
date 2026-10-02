"""Compare joint group sensitivity with actual held-out LR removal/refit effects."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(os.environ.get('PATHOTME_ROOT',str(Path(__file__).resolve().parents[2])))
sys.path[:0]=[str(ROOT/'campaigns/feature_ablation_allshots_20260915'),str(ROOT/'campaigns/feature_attribution_20260915')]
from allshot_design import base,load,sha,check_identity,verify_files,result_root
from score_core import linear_scores,patient_scores

def analyze(manifest_path,output):
    m=load(manifest_path);check_identity(m);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    verify_files(m['source_sha256']);rows=[];evidence={str(manifest_path):sha(manifest_path)};max_error=0
    for task in m['tasks']:
        folder=result_root(m,task);frame=base.read_splits(task)['test'];names=task['feature_names']
        if task['shots']==16:
            model=load(folder/'full/model.json');record=load(folder/'full/metrics.json');verify_files(record['artifact_sha256'])
            scaler=model['scaler'];coef=np.array(model['coef']);intercept=model['intercept']
            results={name:load(folder/name/'metrics.json') for name in ['full',*['leave_group_out_'+g for g in base.groups_for(task)]]}
            saved=base.checked_predictions(folder/'full/test_predictions.csv',frame).probability_1.to_numpy()
            paths=[folder/'full/model.json',*[folder/name/'metrics.json' for name in results]]
        else:
            record=load(folder/'results.json');verify_files(record['artifact_sha256'])
            design=load(folder/'design.json');selection=load(folder/'selection.json')
            model=next(x for x in selection['models'] if x['condition']['name']=='full')
            scaler=design['scaler'];coef=np.array(model['coef']);intercept=model['intercept']
            results={x['condition']['name']:x for x in record['conditions']}
            with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                saved=z['test_probability_1'][np.flatnonzero(z['condition_name']=='full')[0]]
            paths=[folder/'results.json',folder/'design.json',folder/'selection.json',folder/'predictions.npz']
        evidence.update({str(p):sha(p) for p in paths});verify_files(task['input_sha256']);evidence.update(task['input_sha256'])
        z=base.scale_values(base.feature_matrix(task,frame),scaler)
        units=[{'name':g,'indices':[names.index(n) for n in ns]} for g,ns in base.groups_for(task).items()]
        p,scores,_=linear_scores(z,coef,intercept,units);error=float(np.max(np.abs(p-saved)));max_error=max(max_error,error)
        if error>1e-12:raise ValueError('Saved full-model prediction replay mismatch')
        patients,labels,ps=patient_scores(frame.to_dict('records'),scores)
        full=results['full']['metrics']['test']['patient_metrics']['auroc_ovr']
        for i,unit in enumerate(units):
            without=results['leave_group_out_'+unit['name']]['metrics']['test']['patient_metrics']['auroc_ovr']
            rows.append({'cohort':task['cohort'],'shots':task['shots'],'fold':task['fold'],'group':unit['name'],
                         'features':len(unit['indices']),'patients':len(patients),
                         'attribution_magnitude_pp':float(np.abs(ps[:,i,1]).mean()),
                         'mean_signed_positive_class_pp':float(ps[:,i,1].mean()),
                         'full_auroc':full,'without_group_auroc':without,'auroc_drop_after_removal_pp':100*(full-without)})
    frame=pd.DataFrame(rows);frame.to_csv(root/'fold_group_comparison.csv',index=False)
    agg=frame.groupby(['cohort','shots','group'],as_index=False).agg(attribution_magnitude_pp=('attribution_magnitude_pp','mean'),
         auroc_drop_after_removal_pp=('auroc_drop_after_removal_pp','mean'),folds=('fold','nunique'),features=('features','first'))
    agg.to_csv(root/'group_comparison.csv',index=False);corr=[]
    for (cohort,shots),g in agg.groupby(['cohort','shots']):
        if len(g)!=4 or not (g.folds==5).all():raise ValueError('Incomplete matched group comparison')
        rho=float(spearmanr(g.attribution_magnitude_pp,g.auroc_drop_after_removal_pp).statistic)
        corr.append({'cohort':cohort,'shots':int(shots),'groups':4,'spearman_rho':rho})
    pd.DataFrame(corr).to_csv(root/'correlation_by_cohort_shot.csv',index=False)
    notes=['# Attribution magnitude versus downstream performance','',
      'PathoTME-LR only. The original attribution exports used finer semantic groups; this analysis recomputes the same joint train-mean replacement score for the exact four broad biological groups used by the completed refit ablations. It does not sum individual feature scores or the finer group scores.', '',
      'For every held-out patient, joint group attribution is the signed probability change after replacing all group features at the saved training-standardized mean. We take mean absolute patient magnitude, then the mean across five folds. The downstream effect is full-model AUROC minus AUROC after removing the group and freshly refitting/tuning LR with the original training and validation data. Positive means removal hurts. The latter allows other features to compensate, so it measures something different from sensitivity of the existing model.', '',
      '| Cohort | Shots | Spearman correlation |','|---|---:|---:|']
    for r in corr:notes.append(f"| {r['cohort']} | {r['shots']} | {r['spearman_rho']:+.2f} |")
    notes+=['', 'Each correlation has only four group-level points and is descriptive, not a significance claim. Groups differ in size and redundancy. Folds, shots and cohorts are not independent confirmations; the same test patients recur across shots. This posthoc comparison uses test results and cannot select a final feature panel or establish generalization. No individual-feature removal/refit study or neural-adapter attribution/performance correlation is claimed. High attribution can reflect helpful or harmful dependence, and a low drop after refitting may reflect redundancy.', '',
      'No model was trained for this analysis: all reduced-feature performance results were already completed. Joint LR scores require only saved coefficients and matrix operations. No GPU/Slurm changes.']
    (root/'REPORT.md').write_text('\n'.join(notes)+'\n')
    record={'status':'completed','scope':'LR broad-group attribution versus group-removal refit performance',
            'manifest_identity':m['identity'],'rows':len(rows),'cohort_shot_correlations':corr,
            'maximum_saved_probability_replay_error':max_error,'input_sha256':evidence,
            'source_sha256':{str(Path(__file__)):sha(__file__),str(ROOT/'campaigns/feature_attribution_20260915/score_core.py'):sha(ROOT/'campaigns/feature_attribution_20260915/score_core.py')},
            'additional_model_fits':0,'GPU_jobs':0,'Slurm_mutations':0,'groups_per_correlation':4,'inference':'descriptive only'}
    (root/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['source_sha256','input_sha256']},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.manifest,a.output)
