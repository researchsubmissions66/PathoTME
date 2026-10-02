"""Descriptive TME feature redundancy, with exact existing patient covariates."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,leaves_list,fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata,spearmanr
from sklearn.decomposition import PCA

ROOT=Path(os.environ.get('PATHOTME_ROOT',str(Path(__file__).resolve().parents[2])))
sys.path.insert(0,str(ROOT/'campaigns/feature_ablation_allshots_20260915'))
from allshot_design import base,load,sha,identity,verify_files,check_identity

COHORTS=['nsclc','brca','crc','blca']
POLICY={
 'version':'tme_redundancy_20260915_v1',
 'scope':'descriptive covariance structure of existing panels; no classifier, feature selection or performance evaluation',
 'unit':'one patient; arithmetic mean of available fixed-transformed slide covariates',
 'preprocessing':'exact cohort-median-imputed and cohort-standardized patient matrix used for the fixed UMAP/PCA display',
 'primary_correlation':'Spearman; average ranks for ties, Pearson correlation of rank columns',
 'clustering':'complete linkage on 1-abs(Spearman rho), optimal leaf order; same order on both heatmap axes',
 'primary_absolute_rho_threshold':0.85,
 'descriptive_threshold_sensitivity':[0.8,0.85,0.9],
 'cut_interpretation':'Within every complete-link cluster, every pair has abs(rho) >= threshold; singleton clusters retained.',
 'imputation_sensitivity':'pairwise-observed patient-mean Spearman correlations and exact pairwise patient counts; no p-values',
 'pca':'full SVD of the same imputed standardized patient feature matrix; covariance spectrum, not rank correlation spectrum',
 'effective_rank':'exp(-sum(p_i*log(p_i))), with p_i = covariance eigenvalue / sum eigenvalues',
 'participation_ratio':'1/sum(p_i**2)',
 'variance_thresholds':[0.8,0.9,0.95],
 'constant_column_rule':'population SD <= 1e-12 after saved display preprocessing; exclude and report, never fabricate correlations',
 'cohorts':'separate cohort analyses; historical BRCA morph64; other cohorts core62',
 'shots':'not a shot-dependent calculation; same cohort feature matrix at all shot counts',
 'limits':['Effective linear dimension, correlation clusters and PCA variance are not counts of independent biological signals.',
           'PCA variance is not prediction performance; no test-selected replacement panel is proposed.',
           'Cohort-marginal correlation can reflect subtype mixture, shared denominators and compositional constraints.',
           'Strong correlation alone does not establish that a feature is biologically unnecessary or safely removable.',
           'All cohort covariates enter this descriptive analysis; this is not held-out feature selection.'],
 'model_refits':0,'gpu_jobs':0,'slurm_mutations':0}


def matrix_analysis(x,threshold=.85):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or len(x)<3 or not np.isfinite(x).all():raise ValueError('Invalid feature matrix')
    active=np.flatnonzero(x.std(axis=0,ddof=0)>1e-12)
    if len(active)<2:raise ValueError('Need at least two variable features')
    values=x[:,active]
    corr=np.corrcoef(rankdata(values,axis=0,method='average'),rowvar=False)
    corr=(corr+corr.T)/2;corr=np.clip(corr,-1,1);np.fill_diagonal(corr,1)
    if not np.isfinite(corr).all():raise ValueError('Undefined correlation')
    distances=1-np.abs(corr);np.fill_diagonal(distances,0)
    tree=linkage(squareform(distances,checks=True),method='complete',optimal_ordering=True)
    order=leaves_list(tree).astype(int)
    cuts={}
    for cut in POLICY['descriptive_threshold_sensitivity']:
        clusters=fcluster(tree,t=1-cut,criterion='distance')
        numbered={old:i+1 for i,old in enumerate(dict.fromkeys(clusters[order]))}
        clusters=np.array([numbered[k] for k in clusters],dtype=int)
        for k in np.unique(clusters):
            ix=np.flatnonzero(clusters==k)
            if len(ix)>1 and np.min(np.abs(corr[np.ix_(ix,ix)]))<cut-1e-12:raise ValueError('Complete-link threshold violated')
        cuts[cut]=clusters
    pca=PCA(n_components=min(values.shape),svd_solver='full').fit(values)
    eigen=pca.explained_variance_;ratio=pca.explained_variance_ratio_;cumulative=np.cumsum(ratio)
    if not np.isclose(ratio.sum(),1,atol=1e-12) or np.any(eigen < 0):raise ValueError('Invalid PCA spectrum')
    positive=ratio[ratio>0]
    stats={'features_total':x.shape[1],'variable_features':len(active),'constant_features':x.shape[1]-len(active),
           'effective_rank':float(np.exp(-np.sum(positive*np.log(positive)))),
           'participation_ratio':float(1/np.sum(ratio**2)),
           'numerical_rank':int(np.linalg.matrix_rank(values-values.mean(0))),
           'pc1_variance_percent':float(100*ratio[0]),'pc1_2_variance_percent':float(100*ratio[:2].sum()),
           **{f'pcs_{int(t*100)}':int(np.searchsorted(cumulative,t)+1) for t in POLICY['variance_thresholds']}}
    pairs=np.abs(corr[np.triu_indices(len(corr),1)])
    stats.update(pair_count=len(pairs),strong_pairs_085=int((pairs>=threshold).sum()),
                 median_absolute_rho=float(np.median(pairs)))
    return active,corr,tree,order,cuts,pca,stats


def numerical_check():
    # Known rank ties, an exact sign-reversed pair, an independent column, and a constant.
    x=np.array([[0,0,0,1],[0,0,1,1],[1,-1,0,1],[1,-1,1,1],[2,-2,0,1],[2,-2,1,1]],float)
    x[:,:3]=(x[:,:3]-x[:,:3].mean(0))/x[:,:3].std(0)
    active,r,tree,order,cuts,pca,stats=matrix_analysis(x)
    expected=np.asarray(spearmanr(x[:,:3],axis=0).statistic)
    assert np.allclose(r,expected,rtol=0,atol=1e-14)
    assert active.tolist()==[0,1,2] and np.isclose(r[0,1],-1) and np.allclose(r[:2,2],0)
    assert cuts[.85][0]==cuts[.85][1] and cuts[.85][0]!=cuts[.85][2]
    assert stats['constant_features']==1 and np.allclose(pca.explained_variance_ratio_[:2],[2/3,1/3])
    return {'status':'passed','scope':'average-rank Spearman, negative redundancy, constant exclusion, complete-link cut, known PCA spectrum'}


def compute(maps_manifest,allshot_manifest,output):
    start=time.monotonic();out=base.private_output(output)
    if out.exists():raise FileExistsError('Use a new output directory; never overwrite an existing analysis')
    numerical=numerical_check();maps=load(maps_manifest);allshots=load(allshot_manifest)
    check_identity(maps);check_identity(allshots)
    if maps['allshot_identity']!=allshots['identity']:raise ValueError('Mismatched cohort sources')
    evidence={str(Path(p).resolve()):sha(p) for p in [maps_manifest,allshot_manifest]}
    source=dict(allshots['source_sha256']);verify_files(source)
    source.update({str(p):sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in ['.py','.html']})
    maps_root=Path(maps['output']);report_path=maps_root/'embedding_report.json';embedding=load(report_path)
    if embedding['manifest_identity']!=maps['identity']:raise ValueError('Wrong PCA reference')
    evidence[str(report_path)]=sha(report_path)
    out.mkdir(parents=True);cohorts={};summaries=[];pair_rows=[];feature_rows=[];cluster_rows=[];sensitivity=[];audits=[]
    for c in COHORTS:
        folder=out/c;folder.mkdir()
        inp=maps_root/c/'embedding_input.npz';meta=maps_root/c/'display_preprocessing.json'
        for p in [inp,meta]:
            if sha(p)!=maps['prepared_artifact_sha256'][str(p)]:raise ValueError('Changed UMAP covariate input')
            evidence[str(p)]=sha(p)
        d=load(meta);names=d['feature_names'];task_list=[t for t in allshots['tasks'] if t['cohort']==c and t['shots']==16]
        if sorted(t['fold'] for t in task_list)!=list(range(5)):raise ValueError('Expected five frozen outer folds')
        task=task_list[0]
        if names!=task['feature_names']:raise ValueError('Changed panel order')
        table=Path(task['tme_csv'])
        if sha(table)!=task['input_sha256'][str(table)]:raise ValueError('Changed original feature table')
        evidence[str(table)]=sha(table)
        rows=[];owners={}
        for t in task_list:
            for row in t['memberships']['test']:
                if owners.setdefault(row['case_id'],t['fold'])!=t['fold']:raise ValueError('Duplicate outer-test patient')
                rows.append({k:row[k] for k in ['slide_id','case_id']})
        frame=pd.DataFrame(rows).sort_values('slide_id').reset_index(drop=True)
        if frame.slide_id.duplicated().any():raise ValueError('Duplicate slide')
        raw_slide=base.feature_matrix(task,frame)
        raw=pd.DataFrame(raw_slide,columns=names).assign(case_id=frame.case_id).groupby('case_id',sort=True).mean().to_numpy()
        with np.load(inp,allow_pickle=False) as z:
            if z.files!=['x']:raise ValueError('Embedding input must contain only covariates')
            x=z['x']
        reconstructed=base.scale_values(raw,d['scaler'])
        if x.shape!=raw.shape or not np.allclose(x,reconstructed,rtol=0,atol=1e-12):raise ValueError('UMAP patient covariates failed to reproduce')
        if len(x)!=maps['cohorts'][c]['patients']:raise ValueError('Wrong cohort size')
        active,corr,tree,order,cuts,pca,stats=matrix_analysis(x)
        reference=np.array(embedding['cohorts'][c]['pca_explained_variance_ratio'])
        delta=float(np.max(np.abs(reference-pca.explained_variance_ratio_[:2])))
        if delta>1e-12:raise ValueError('Existing PCA reference mismatch')
        feature_names=[names[i] for i in active];observed=raw[:,active]
        obs_corr=pd.DataFrame(observed).corr(method='spearman',min_periods=3).to_numpy()
        present=np.isfinite(observed).astype(np.int64);pair_n=present.T@present
        upper=np.triu_indices(len(active),1);valid=np.isfinite(obs_corr[upper])
        differences=np.abs(obs_corr[upper][valid]-corr[upper][valid])
        stats.update(cohort=c,label=maps['cohorts'][c]['label'],panel=maps['cohorts'][c]['panel'],patients=len(x),slides=len(frame),
            missing_patient_measurements=int(np.isnan(raw).sum()),missing_fraction=float(np.isnan(raw).mean()),
            observed_pair_count_min=int(pair_n[upper].min()),observed_pair_count_max=int(pair_n[upper].max()),
            imputation_max_abs_rho_change=float(differences.max()),imputation_median_abs_rho_change=float(np.median(differences)),
            pairs_changing_085_status=int(np.sum((np.abs(obs_corr[upper][valid])>=.85)!=(np.abs(corr[upper][valid])>=.85))))
        broad=base.groups_for(task);groups={n:g for g,ns in broad.items() for n in ns}
        labels=cuts[.85];stats['cluster_units_085']=len(np.unique(labels))
        stats['multi_feature_clusters_085']=int(sum(np.sum(labels==k)>1 for k in np.unique(labels)))
        stats['features_in_multi_feature_clusters_085']=int(sum(np.sum(labels==k) for k in np.unique(labels) if np.sum(labels==k)>1))
        for cut,ids in cuts.items():
            counts=np.bincount(ids)[1:]
            sensitivity.append({'cohort':c,'absolute_rho_threshold':cut,'clusters_including_singletons':len(counts),
                                'multi_feature_clusters':int((counts>1).sum()),'largest_cluster':int(counts.max()),
                                'strong_pairs':int((np.abs(corr[upper])>=cut).sum())})
        ranks={int(i):j+1 for j,i in enumerate(order)}
        for j,n in enumerate(names):
            found=np.flatnonzero(active==j);i=int(found[0]) if len(found) else None
            row={'cohort':c,'panel':stats['panel'],'feature_id':f'F{j+1:02d}','feature_name':n,'biological_group':groups[n],
                 'missing_patients':int(np.isnan(raw[:,j]).sum()),'constant':i is None,
                 'clustered_position':ranks[i] if i is not None else None,'redundancy_cluster_085':int(labels[i]) if i is not None else None}
            feature_rows.append(row)
        for k in np.unique(labels):
            ii=np.flatnonzero(labels==k);within=np.abs(corr[np.ix_(ii,ii)])[np.triu_indices(len(ii),1)]
            cluster_rows.append({'cohort':c,'cluster':int(k),'features':len(ii),'minimum_absolute_rho':float(within.min()) if len(within) else None,
                'feature_ids':';'.join(f'F{active[i]+1:02d}' for i in ii),
                'feature_names':';'.join(feature_names[i] for i in ii),'biological_groups':';'.join(sorted(set(groups[feature_names[i]] for i in ii)))})
        for i,j in zip(*upper):
            pair_rows.append({'cohort':c,'feature_a':feature_names[i],'feature_b':feature_names[j],
                'feature_id_a':f'F{active[i]+1:02d}','feature_id_b':f'F{active[j]+1:02d}',
                'group_a':groups[feature_names[i]],'group_b':groups[feature_names[j]],'rho':float(corr[i,j]),
                'abs_rho':float(abs(corr[i,j])),'observed_rho':float(obs_corr[i,j]) if np.isfinite(obs_corr[i,j]) else None,
                'pairwise_observed_patients':int(pair_n[i,j]),'strong_085':bool(abs(corr[i,j])>=.85)})
        for name,matrix in [('spearman',corr),('observed_spearman',obs_corr),('observed_pair_counts',pair_n)]:
            pd.DataFrame(matrix,index=feature_names,columns=feature_names).to_csv(folder/(name+'.csv'),index_label='feature')
        pd.DataFrame({'component':np.arange(1,len(pca.explained_variance_)+1),'eigenvalue':pca.explained_variance_,
                      'variance_ratio':pca.explained_variance_ratio_,'cumulative_variance_ratio':np.cumsum(pca.explained_variance_ratio_)}).to_csv(folder/'pca_spectrum.csv',index=False)
        pd.DataFrame(pca.components_.T,index=feature_names,columns=[f'PC{i+1}' for i in range(len(pca.components_))]).to_csv(folder/'pca_loadings.csv',index_label='feature')
        # Aggregate-only renderer inputs: no patient rows, IDs or feature measurements.
        record={'summary':stats,'feature_names':feature_names,'feature_ids':[f'F{i+1:02d}' for i in active],
                'groups':[groups[n] for n in feature_names],'rho':corr.tolist(),'observed_rho':[[float(v) if np.isfinite(v) else None for v in row] for row in obs_corr],
                'pair_n':pair_n.tolist(),'linkage':tree.tolist(),'order':order.tolist(),'clusters_085':labels.tolist(),
                'variance_ratio':pca.explained_variance_ratio_.tolist(),'eigenvalues':pca.explained_variance_.tolist()}
        (folder/'analysis.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        cohorts[c]=record;summaries.append(stats)
        audits.append({'cohort':c,'patient_matrix_max_replay_error':float(np.max(np.abs(x-reconstructed))),
                       'PCA_first_two_max_difference':delta,'correlation_symmetric':bool(np.allclose(corr,corr.T)),
                       'all_cluster_minimum_correlations_verified':True,'source_missing_count_matched':int(np.isnan(raw).sum())==d['patient_missing_values']})
        if not audits[-1]['source_missing_count_matched']:raise ValueError('Missingness count mismatch')
        print(c,json.dumps(stats),flush=True)
    for name,rows in [('summary',summaries),('feature_dictionary',feature_rows),('pairwise_correlations',pair_rows),
                      ('redundancy_clusters',cluster_rows),('threshold_sensitivity',sensitivity)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    strong=[r for r in pair_rows if r['strong_085']]
    pd.DataFrame(strong,columns=list(pair_rows[0])).sort_values(['cohort','abs_rho'],ascending=[True,False]).to_csv(out/'strong_pairs.csv',index=False)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'policy':POLICY,'maps_manifest_identity':maps['identity'],
              'allshot_identity':allshots['identity'],'output':str(out),'input_sha256':evidence,'source_sha256':source,
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
              'numerical_check':numerical,'audits':audits,'cohorts':summaries,'wall_seconds':time.monotonic()-start,
              'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    manifest['identity']=identity(manifest);base.atomic_json(out/'manifest.json',manifest)
    print(json.dumps({'status':'completed','identity':manifest['identity'],'wall_seconds':manifest['wall_seconds'],'pairs':len(pair_rows),'strong_pairs':len(strong)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--maps-manifest',type=Path,required=True)
    p.add_argument('--allshot-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compute(a.maps_manifest,a.allshot_manifest,a.output)
