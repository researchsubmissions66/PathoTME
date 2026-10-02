"""Bind fixed TME covariates to exact saved out-of-fold predictions; CPU only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('PATHOTME_ROOT',str(Path(__file__).resolve().parents[2])))
sys.path.insert(0,str(ROOT/'campaigns/feature_ablation_allshots_20260915'))
from allshot_design import base,load,sha,identity,verify_files,check_identity,result_root,COHORTS

POLICY={
 'version':'tme_prediction_maps_20260915_v1',
 'unit':'one patient; mean of available fixed-transformed slide covariates',
 'projection_preprocessing':'cohort-patient median imputation, mean and population SD; display only',
 'projection_scope':'posthoc descriptive transductive; all cohort covariates, no labels or predictions',
 'umap':{'n_neighbors':15,'min_dist':0.3,'metric':'euclidean','n_components':2,
         'random_state':20260915,'transform_seed':20260915,'n_jobs':1,'n_epochs':500,'init':'spectral'},
 'pca':{'n_components':2,'svd_solver':'full'},
 'coordinates':'one fixed fit per cohort/projector, shared across all shots and prediction overlays',
 'predictions':'saved held-out slide probabilities averaged per patient within that patient\'s unique test fold',
 'decision_rule':'probability_1 >= 0.5; fixed illustrative threshold, no test tuning',
 'metrics':'mean of paired fold patient metrics; fold SD is not a confidence interval; no pooled AUROC',
 'metric_precision':'float64 saved-CSV patient averaging for all plotted variants; legacy float32 AUROC differences require exact float32 replay and are recorded',
 'delta':'100*(TME true-class probability - Native true-class probability)',
 'main_comparison':'ViLa / CLIP-RN50 / 16-shot / PathoTME-Adapter; chosen before map generation, not by outcome',
 'limitations':['UMAP separation does not establish prediction benefit, causal biology or generalization.',
                'Cohort-specific maps are not aligned; distances and cluster areas are not comparable across cohorts.',
                'Historical BRCA remains morph64; NSCLC, CRC and BLCA use core62.',
                'No original feature panel, split, classifier scaler, model or threshold is changed.',
                'Partial model coverage is disclosed; null and negative effects are retained.'],
 'neural_training':False,'gpu_jobs':0}

DISPLAY={'nsclc':'TCGA-NSCLC','brca':'TCGA-BRCA','crc':'TCGA-CRC','blca':'TCGA-BLCA'}
CLASSES={'nsclc':['LUAD','LUSC'],'brca':['IDC','ILC'],
         'crc':['Adenocarcinoma NOS','Mucinous adenocarcinoma'],'blca':['Non-papillary','Papillary']}
METHODS={'vila_mil':'ViLa','mgpath':'MGPATH','focus':'FOCUS','muse':'MUSE',
         'hive_mil':'HiVE-MIL','dyko':'DyKo','mscpt':'MSCPT'}
ARMS={'actual':'PathoTME-Adapter','zero':'PathoTME-Adapter-Zero','shuffled':'PathoTME-Adapter-Shuffled'}


def prepare(allshot_path,output):
    out=base.private_output(output)
    if out.exists():raise FileExistsError('Use a new snapshot directory; never overwrite a prepared map.')
    out.mkdir(parents=True)
    evidence={}; source={}; discrepancies=[]; precision_replays=[]; legacy_encoders=[]
    def bind(path,expected=None):
        path=str(Path(path));digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('Changed input: '+path)
        if path in evidence and evidence[path]!=digest:raise ValueError('Input changed during snapshot')
        evidence[path]=digest
        return Path(path)
    def read(path):return load(bind(path))
    def verify_consumed(path,hashes):
        key=str(Path(path))
        if key not in hashes:raise ValueError('Prediction absent from completion hash bindings: '+key)
        return bind(path,hashes[key])
    m=read(allshot_path);check_identity(m);verify_files(m['source_sha256']);source.update(m['source_sha256'])
    parent=read(m['parent_manifest']);check_identity(parent)
    baseline=read(parent['parent_manifest']);check_identity(baseline)
    source.update(baseline['source_sha256'])
    task_index={(t['cohort'],t['shots'],t['fold']):t for t in m['tasks']}
    frames={}; cohorts={}; positions={}; private=[]
    for c in COHORTS:
        tasks=sorted((t for t in m['tasks'] if t['cohort']==c and t['shots']==16),key=lambda t:t['fold'])
        for task in tasks:
            for path,digest in task['input_sha256'].items():bind(path,digest)
        parts=[pd.DataFrame(t['memberships']['test']).assign(fold=t['fold']) for t in tasks]
        frame=pd.concat(parts,ignore_index=True).sort_values('slide_id').reset_index(drop=True)
        if frame.slide_id.duplicated().any() or (frame.groupby('case_id').fold.nunique()!=1).any():
            raise ValueError('Overlapping test folds')
        if (frame.groupby('case_id').label.nunique()!=1).any():raise ValueError('Conflicting patient labels')
        raw=base.feature_matrix(tasks[0],frame)
        values=pd.DataFrame(raw,columns=tasks[0]['feature_names']).assign(case_id=frame.case_id).groupby('case_id',sort=True).mean()
        scaler=base.fit_scaler(values.to_numpy());x=base.scale_values(values.to_numpy(),scaler)
        patient=frame.groupby('case_id',sort=True).agg(label=('label','first'),fold=('fold','first'),slides=('slide_id','size'))
        if list(patient.index)!=list(values.index):raise ValueError('Patient covariate order mismatch')
        positions[c]={case:i for i,case in enumerate(patient.index)}
        folder=out/c;folder.mkdir()
        # Deliberately no label, case ID or prediction is supplied to the embedding process.
        np.savez_compressed(folder/'embedding_input.npz',x=x)
        (folder/'display_preprocessing.json').write_text(json.dumps({'policy':POLICY,'scaler':scaler,
                   'feature_names':tasks[0]['feature_names'],'patient_missing_values':int(np.isnan(values.to_numpy()).sum())},indent=2)+'\n')
        cohorts[c]={'label':DISPLAY[c],'classes':CLASSES[c], 'panel':'morph64' if c=='brca' else 'core62',
                    'patients':len(patient),'slides':len(frame),'label_id':patient.label.astype(int).tolist(),
                    'fold':patient.fold.astype(int).tolist()}
        private.extend({'cohort':c,'point_index':i,'case_id':case,'fold':int(patient.loc[case,'fold'])} for case,i in positions[c].items())
    groups=defaultdict(list); missing=[]; metrics_rows=[]
    def expected(task):
        key=(task['cohort'],task['shots'],task['fold'])
        if key not in frames:
            for p,h in task['input_sha256'].items():bind(p,h)
            frames[key]=base.read_splits(task)
            if any(f.to_dict('records')!=task['memberships'][p] for p,f in frames[key].items()):
                raise ValueError('Changed frozen phase membership')
        return frames[key]['test']
    def add(task,method,encoder,variant,p,native=None,saved_auroc=None):
        frame=expected(task);p=np.asarray(p,float)
        metric=base.metrics(frame,p)['patient_metrics']
        if saved_auroc is not None:
            difference=metric['auroc_ovr']-saved_auroc
            discrepancies.append(abs(difference))
            if abs(difference)>1e-12:
                from sklearn.metrics import roc_auc_score
                old=frame.assign(p=p.astype(np.float32)).groupby('case_id',sort=True).agg(label=('label','first'),p=('p','mean'))
                replay=float(roc_auc_score(old.label,old.p))
                if abs(replay-saved_auroc)>1e-12:
                    raise ValueError(f"Unexplained saved/prediction AUROC mismatch: {task['cohort']} {task['shots']} {task['fold']} {method} {encoder} {variant}: {saved_auroc} vs {metric['auroc_ovr']}; float32={replay}")
                precision_replays.append({'cohort':task['cohort'],'shots':task['shots'],'fold':task['fold'],'method':method,
                                         'encoder':encoder,'variant':variant,'saved_AUROC':saved_auroc,'float32_replay':replay,
                                         'plotted_float64_AUROC':metric['auroc_ovr'],'difference_pp':100*difference})
        f=frame.assign(p=p)
        if native is not None:f['native']=np.asarray(native,float)
        grouped=f.groupby('case_id',sort=True)
        a=grouped.agg(label=('label','first'),p=('p','mean'))
        if native is not None:a['native']=grouped.native.mean()
        c=task['cohort'];ix=[positions[c][case] for case in a.index]
        if a.label.tolist()!=[cohorts[c]['label_id'][i] for i in ix] or any(cohorts[c]['fold'][i]!=task['fold'] for i in ix):
            raise ValueError('Wrong patient/fold overlay')
        row={'cohort':c,'shots':task['shots'],'method':method,'encoder':encoder,'variant':variant,
             'fold':task['fold'],'point_index':ix,'probability':a.p.tolist(),'metrics':metric}
        if native is not None:
            row['native_probability']=a.native.tolist();row['native_metrics']=base.metrics(frame,native)['patient_metrics']
        groups[(c,task['shots'],method,encoder,variant)].append(row)
        metrics_rows.append({k:row[k] for k in ['cohort','shots','method','encoder','variant','fold']}|
                            {'patients':len(a)}|metric|({'native_auroc':row['native_metrics']['auroc_ovr']} if native is not None else {}))
    # Every available shot has a complete full-panel LR control; no refitting.
    for task in m['tasks']:
        folder=result_root(m,task);frame=expected(task)
        if task['shots']==16:
            r=read(folder/'full/metrics.json')
            if r['status']!='completed' or r['manifest_identity']!=m['parent_identity']:raise ValueError('Invalid reused LR')
            path=verify_consumed(folder/'full/test_predictions.csv',r['artifact_sha256'])
            p=base.checked_predictions(path,frame).probability_1.to_numpy()
            saved=r['metrics']['test']['patient_metrics']['auroc_ovr']
        else:
            r=read(folder/'results.json')
            if r['status']!='completed' or r['task_id']!=task['id'] or r['manifest_identity']!=m['identity']:raise ValueError('Invalid LR')
            path=verify_consumed(folder/'predictions.npz',r['artifact_sha256'])
            with np.load(path,allow_pickle=False) as z:
                actual=pd.DataFrame({k:z['test_'+k] for k in ['slide_id','case_id','label']})
                if not actual.equals(frame):raise ValueError('Compact prediction memberships differ')
                idx=np.flatnonzero(z['condition_name']=='full')
                if len(idx)!=1:raise ValueError('Missing/duplicate full-panel LR')
                p=z['test_probability_1'][idx[0]]
            saved=next(q['metrics']['test']['patient_metrics']['auroc_ovr'] for q in r['conditions'] if q['condition']['name']=='full')
        add(task,'TME only','none','PathoTME-LR',p,saved_auroc=saved)
    plans=[]
    for task in baseline['fusion_tasks']:
        plans.append({'task':task,'plan':task['plan'],'identity':task['launch_identity'],'shots':16})
    low=read(Path(baseline['output']).parent/'tcga_4_8shot_20260912_v1/campaign.json')
    for group in low['groups']:
        launch=read(group['launch'])
        if launch['identity']!=group['identity']:raise ValueError('Wrong low-shot launch')
        for plan in launch['plans']:plans.append({'plan':plan,'identity':launch['identity'],'shots':group['shots']})
    for entry in plans:
        plan=entry['plan'];c,fold=plan['cohort'],plan['fold'];shot=entry['shots']
        task=task_index[c,shot,fold];frame=expected(task)
        if shot==16:
            # Some original MGPATH plans omit encoder. The audited baseline task
            # explicitly binds the original recipe; never guess RN50 for those.
            binding=entry['task'];encoder_key=binding['encoder']
            if binding['method']!=plan['method']:raise ValueError('Method binding mismatch')
            if 'encoder' in plan and plan['encoder']!=encoder_key:raise ValueError('Encoder binding mismatch')
            if 'encoder' not in plan:legacy_encoders.append({'task':binding['name'],'encoder':encoder_key,'basis':'audited baseline manifest task'})
        else:encoder_key=plan['encoder']
        method=METHODS[plan['method']];encoder={'plip':'PLIP','clip-rn50':'CLIP-RN50'}[encoder_key]
        marker=Path(plan['output'])/'fold_complete.json'
        if not marker.exists():
            missing.append({'cohort':c,'shots':shot,'method':method,'encoder':encoder,'fold':fold,'reason':'no completed fold marker'});continue
        completion=read(marker)
        if completion['status']!='completed' or completion['launch_identity']!=entry['identity'] or completion['plan']!=plan:
            raise ValueError('Completion plan/identity mismatch')
        native_path=verify_consumed(Path(plan['native_dir'])/f'fold{fold}_predictions.csv',completion['native_sha256'])
        native=base.checked_predictions(native_path,frame).probability_1.to_numpy()
        for arm,variant in ARMS.items():
            r=completion['arms'][arm]
            if r['status']!='completed' or r['launch_identity']!=entry['identity']:raise ValueError('Invalid arm completion')
            path=verify_consumed(Path(plan['output'])/arm/'predictions.csv',r['artifact_sha256'])
            p=base.checked_predictions(path,frame).probability_1.to_numpy()
            add(task,method,encoder,variant,p,native,saved_auroc=r['metrics']['patient_metrics']['auroc_ovr'])
        if shot==16:
            ft=entry['task'];folder=Path(baseline['output'])/'fusion'/ft['name'];fpath=folder/'metrics.json'
            if not fpath.exists():continue
            r=read(fpath)
            if r['status']!='completed' or r['task_id']!=ft['id']:raise ValueError('Wrong fusion result')
            path=verify_consumed(folder/'predictions.csv',r['artifact_sha256']);p=pd.read_csv(path).sort_values('slide_id').reset_index(drop=True)
            if not p[['slide_id','case_id','label']].equals(frame):raise ValueError('Fusion membership mismatch')
            if not np.allclose(p.native_probability_1,native,rtol=0,atol=1e-12):raise ValueError('Fusion native differs')
            for name,variant in [('late_fusion_fixed_half','PathoTME-Fusion50'),('late_fusion_validation','PathoTME-FusionVal')]:
                add(task,method,encoder,variant,p[name+'_probability_1'],native,saved_auroc=r['conditions'][name]['patient_metrics']['auroc_ovr'])
    records=[]
    for key,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['fold']);folds=[r['fold'] for r in rows]
        indices=sum([r['point_index'] for r in rows],[])
        if len(set(indices))!=len(indices) or len(set(folds))!=len(folds):
            raise ValueError(f'Repeated patient/fold: {key}, folds={folds}, patients={len(indices)}, unique={len(set(indices))}')
        record=dict(zip(['cohort','shots','method','encoder','variant'],key))
        record.update(id='|'.join(map(str,key)),folds=folds,complete=folds==list(range(5)),patients=len(indices),
                      point_index=indices,probability=sum([r['probability'] for r in rows],[]),
                      auroc=100*float(np.mean([r['metrics']['auroc_ovr'] for r in rows])),
                      auroc_sd=100*float(np.std([r['metrics']['auroc_ovr'] for r in rows],ddof=1)) if len(rows)>1 else None)
        if 'native_probability' in rows[0]:
            record['native_probability']=sum([r['native_probability'] for r in rows],[])
            record['native_auroc']=100*float(np.mean([r['native_metrics']['auroc_ovr'] for r in rows]))
            record['delta_auroc']=record['auroc']-record['native_auroc']
            y=np.array(cohorts[key[0]]['label_id'])[indices];p=np.array(record['probability']);n=np.array(record['native_probability'])
            before=(n>=.5)==y;after=(p>=.5)==y
            record['outcomes']={'Corrected':int((~before&after).sum()),'Worsened':int((before&~after).sum()),
                                'Both correct':int((before&after).sum()),'Both wrong':int((~before&~after).sum())}
            record['mean_delta_true_class_pp']=float(np.mean(100*(p-n)*(2*y-1)))
        records.append(record)
    pd.DataFrame(private).to_csv(out/'private_patient_index.csv',index=False)
    pd.DataFrame(metrics_rows).to_csv(out/'fold_metrics.csv',index=False)
    dataset={'policy':POLICY,'cohorts':cohorts,'comparisons':records}
    (out/'overlays.json').write_text(json.dumps(dataset,allow_nan=False,separators=(',',':'))+'\n')
    source.update({str(p):sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in ['.py','.html']})
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'policy':POLICY,'allshot_identity':m['identity'],
              'source_sha256':source,'input_sha256':evidence,'output':str(out),'missing_neural_folds':missing,
              'cohorts':{c:{k:v for k,v in x.items() if k not in ['label_id','fold']} for c,x in cohorts.items()},
              'counts':{'LR_cohort_shot_groups':sum(r['variant']=='PathoTME-LR' for r in records),
                        'paired_groups':sum('native_probability' in r for r in records),
                        'complete_paired_groups':sum('native_probability' in r and r['complete'] for r in records),
                        'neural_completed_folds':len(plans)-len(missing),'neural_expected_folds':len(plans)},
              'maximum_recomputed_saved_fold_AUROC_difference':max(discrepancies),
              'legacy_float32_precision_replays':precision_replays,
              'legacy_encoder_bindings':legacy_encoders,
              'prepared_artifact_sha256':{str(p):sha(p) for p in out.rglob('*') if p.is_file()}}
    manifest['identity']=identity(manifest);base.atomic_json(out/'manifest.json',manifest)
    print(json.dumps({'identity':manifest['identity'],'counts':manifest['counts'],'maximum_AUROC_difference':max(discrepancies)},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--allshots',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare(args.allshots,args.output)
