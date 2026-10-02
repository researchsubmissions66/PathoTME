#!/usr/bin/env python3
"""Private aggregate figures for completed within-panel LR ablations."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_ablation import load, identity, sha, verify_files, atomic_json, csv_write

LABELS = {'nsclc':'TCGA-NSCLC', 'brca':'TCGA-BRCA', 'crc':'TCGA-CRC', 'blca':'TCGA-BLCA'}
GROUPS = {'tissue_architecture':'Tissue architecture', 'cell_composition':'Cell composition',
          'spatial_interactions':'Spatial interactions', 'tls':'TLS', 'tissue_geometry':'Tissue geometry'}
METRICS = ['auroc_ovr','nll','brier','accuracy','balanced_accuracy','macro_f1','ece']


def report(root):
    root = Path(root)
    m = load(root/'manifest.json')
    if identity({k:v for k,v in m.items() if k!='identity'}) != m['identity']:
        raise ValueError('Invalid experiment manifest')
    rows, selected, missing = [], [], []
    for task in m['tasks']:
        folder = root/'folds'/f"{task['cohort']}_f{task['fold']}"
        design_path = folder/'design.json'
        if not design_path.exists():
            missing.append({'cohort':task['cohort'], 'fold':task['fold'], 'missing':'fold'}); continue
        design = load(design_path)
        if identity({k:v for k,v in design.items() if k!='identity'}) != design['identity']:
            raise ValueError('Invalid design identity')
        for cond in design['conditions']:
            path = folder/cond['name']/'metrics.json'
            if not path.exists():
                missing.append({'cohort':task['cohort'], 'fold':task['fold'], 'missing':cond['name']}); continue
            r = load(path)
            if r['status']!='completed' or r['manifest_identity']!=m['identity'] or r['design_identity']!=design['identity'] or r['condition']!=cond:
                raise ValueError('Invalid condition binding')
            verify_files(r['artifact_sha256'])
            score = r['metrics']['test']['patient_metrics']
            rows.append({'cohort':r['cohort'], 'panel':r['panel'], 'fold':r['fold'], 'variant':r['variant'],
                **{k:cond[k] for k in ['name','kind','k','group','repeat']}, 'selected_C':r['selected_C'],
                **{k:score[k] for k in METRICS}, 'patients':r['metrics']['test']['patients']})
            if cond['kind']=='rfe':
                selected.extend({'cohort':r['cohort'], 'panel':r['panel'], 'fold':r['fold'], 'k':cond['k'],
                                 'feature':n} for n in cond['feature_names'])
    if not rows:
        raise ValueError('No completed fits to report')
    frame = pd.DataFrame(rows)
    full = frame[frame.kind=='full'].set_index(['cohort','fold'])
    for metric in METRICS:
        frame['delta_'+metric] = [r[metric]-float(full.loc[(r['cohort'],r['fold']),metric]) for r in rows]
    csv_write(root/'fold_metrics.csv', frame)
    aggregate=[]
    keys=['cohort','panel','variant','name','kind','k','group','repeat']
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        if group.fold.duplicated().any(): raise ValueError('Duplicate fold')
        row={**dict(zip(keys,key)), 'folds':len(group), 'complete_five_fold':set(group.fold)==set(range(5))}
        for metric in METRICS:
            for name in [metric,'delta_'+metric]:
                row[name+'_mean']=float(group[name].mean())
                row[name+'_fold_sd']=float(group[name].std(ddof=1)) if len(group)>1 else None
        aggregate.append(row)
    agg=pd.DataFrame(aggregate)
    csv_write(root/'condition_summary.csv',agg)
    selection=[]
    for (cohort,panel,k), group in pd.DataFrame(selected).groupby(['cohort','panel','k']):
        names=next(t['feature_names'] for t in m['tasks'] if t['cohort']==cohort)
        for n in names:
            selection.append({'cohort':cohort,'panel':panel,'k':int(k),'feature':n,
                              'selected_folds':int((group.feature==n).sum()),'evaluated_folds':int(group.fold.nunique()),
                              'selection_frequency':float((group.feature==n).sum()/group.fold.nunique())})
    csv_write(root/'selection_frequency.csv',pd.DataFrame(selection))
    curves=[]
    for cohort in LABELS:
        a=agg[(agg.cohort==cohort)&agg.complete_five_fold]
        for k in m['policy']['subset_sizes']:
            learned=a[(a.kind=='rfe')&(a.k==k)]; random=a[(a.kind=='random')&(a.k==k)]
            if len(learned)!=1 or len(random)!=m['policy']['random_repeats']: continue
            v=random.auroc_ovr_mean.to_numpy()
            curves.append({'cohort':cohort,'k':k,'rfe_auroc':float(learned.iloc[0].auroc_ovr_mean),
                           'random_mean_auroc':float(v.mean()), 'random_min_auroc':float(v.min()),
                           'random_max_auroc':float(v.max()),'random_panels':len(v),
                           'random_range_is_not_CI':True})
    csv_write(root/'subset_curve.csv',pd.DataFrame(curves))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig, axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,(cohort,label) in zip(axes.ravel(),LABELS.items()):
        a=agg[(agg.cohort==cohort)&agg.complete_five_fold]
        p=next(t['panel'] for t in m['tasks'] if t['cohort']==cohort)
        ax.set_title(label+' · '+('morph64' if cohort=='brca' else 'core62'))
        full_a=a[a.kind=='full']
        if len(full_a)!=1: ax.text(.5,.5,'Incomplete folds',ha='center',transform=ax.transAxes);continue
        f=full_a.iloc[0]
        for repeat in range(m['policy']['random_repeats']):
            r=a[(a.kind=='random')&(a.repeat==repeat)].sort_values('k')
            ax.plot([*r.k,f.k],100*np.array([*r.auroc_ovr_mean,f.auroc_ovr_mean]),color='#aaa',alpha=.45,lw=.8)
        r=a[a.kind=='rfe'].sort_values('k')
        ax.errorbar([*r.k,f.k],100*np.array([*r.auroc_ovr_mean,f.auroc_ovr_mean]),
                    yerr=100*np.array([*r.auroc_ovr_fold_sd,f.auroc_ovr_fold_sd]),fmt='o-',color='#225ea8',
                    capsize=3,label='Training-selected; fold SD')
        c=[x for x in curves if x['cohort']==cohort]
        ax.plot([x['k'] for x in c],100*np.array([x['random_mean_auroc'] for x in c]),'s--',color='#666',label='Mean of 10 random panels')
        ax.set_xticks([8,16,32,int(f.k)]);ax.set_xlabel('Retained features');ax.set_ylabel('Patient AUROC (%)')
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('PathoTME-LR · subset-size curves · 16-shot / five folds\nGray lines: individual random panels; error bars: fold SD, not confidence intervals',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(root/f'subset_size_curves.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(cohort,label) in zip(axes.ravel(),LABELS.items()):
        a=agg[(agg.cohort==cohort)&agg.complete_five_fold]
        groups=m['groups'][cohort];y=np.arange(len(groups));height=.32
        ax.set_title(label+' · '+('morph64' if cohort=='brca' else 'core62'))
        for kind,offset,color,title in [('leave_group_out',-.18,'#d95f0e','Remove group'),('group_only',.18,'#3182bd','Group alone')]:
            r=a[a.kind==kind].set_index('group')
            if set(r.index)!=set(groups):continue
            v=[r.loc[g] for g in groups]
            ax.barh(y+offset,[100*z.delta_auroc_ovr_mean for z in v],height=height,color=color,label=title,
                    xerr=[100*z.delta_auroc_ovr_fold_sd for z in v],capsize=2)
        ax.set_yticks(y,[GROUPS[g]+f' ({len(v)})' for g,v in groups.items()]);ax.invert_yaxis()
        ax.axvline(0,color='black',lw=.8);ax.set_xlabel('AUROC change from full panel (pp)');ax.grid(axis='x',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('PathoTME-LR · biological-group refits\nPositive = higher AUROC than full panel; error bars = SD of paired fold differences',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(root/f'biological_group_ablations.{ext}')
    plt.close(fig)
    lines=['# PathoTME within-panel feature ablations','',
        'Fresh PathoTME-LR fits on the existing 16-shot/five-fold cohorts. The neural adapters have not been retrained by this workflow.', '',
        'Training-only recursive elimination at fixed C=1 creates nested 8/16/32-feature subsets. Each retained panel receives its own validation-selected C using the parent five-value grid. Ten nested random panels provide descriptive controls. All fixed and selected subsets are retained; no best test subset is selected.', '',
        'Core62 and historical BRCA morph64 remain separate. Biological groups partition each panel: core62 has tissue architecture (14), cell composition (28), spatial interactions (16), TLS (4); BRCA has tissue architecture (16), cell composition (28), spatial interactions (12), tissue geometry (8). These broad families aggregate the existing finer tokenizer groups.', '',
        '## Subset-size curve','', '| Cohort | Full panel AUROC | 8 features | 16 features | 32 features |','|---|---:|---:|---:|---:|']
    for cohort,label in LABELS.items():
        a=agg[(agg.cohort==cohort)&agg.complete_five_fold]
        values=[]
        for name in ['full','rfe_k8','rfe_k16','rfe_k32']:
            r=a[a.name==name]
            values.append(f'{100*r.iloc[0].auroc_ovr_mean:.2f} ± {100*r.iloc[0].auroc_ovr_fold_sd:.2f}' if len(r)==1 else 'pending')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', 'Values are patient AUROC percentages, mean ± fold SD. Fold SD is not a confidence interval. Gray random-panel curves are individual five-fold means, not independent patient replications.', '',
        '## Biological-group ablations','', '| Cohort | Group | Features | Remove: ΔAUROC (pp) | Alone: ΔAUROC (pp) |','|---|---|---:|---:|---:|']
    for cohort,label in LABELS.items():
        a=agg[(agg.cohort==cohort)&agg.complete_five_fold]
        for g,features in m['groups'][cohort].items():
            values=[]
            for kind in ['leave_group_out','group_only']:
                r=a[(a.kind==kind)&(a.group==g)]
                values.append(f'{100*r.iloc[0].delta_auroc_ovr_mean:+.2f}' if len(r)==1 else 'pending')
            lines.append('| '+label+' | '+GROUPS[g]+' | '+str(len(features))+' | '+' | '.join(values)+' |')
    lines += ['', 'Changes use the matching full-panel LR fit. Predictive performance after refitting measures sufficiency and redundancy within this panel. It does not prove biological necessity, causal relevance, original-panel optimality, or a neural-adapter effect. Correlated features may substitute for one another. Selection frequencies describe overlap across five fitted folds and are not posterior probabilities.', '',
        'AUROC, NLL, Brier, accuracy, balanced accuracy, macro-F1 and ECE are retained in `condition_summary.csv`; patient-level outputs remain private. This experiment is exploratory after prior results were inspected.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    summary={'status':'completed' if not missing else 'partial','manifest_identity':m['identity'],
             'condition_folds_completed':len(rows),'expected_condition_folds':m['counts']['condition_folds'],
             'complete_five_fold_conditions':int(agg.complete_five_fold.sum()), 'missing':missing,
             'neural_adapter_refits':0,'gpu_jobs':0,
             'artifact_sha256':{p.name:sha(p) for p in root.iterdir() if p.suffix in ['.csv','.png','.pdf','.svg']}}
    atomic_json(root/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['artifact_sha256','missing']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    report(p.parse_args().output)
