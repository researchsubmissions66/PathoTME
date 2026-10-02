#!/usr/bin/env python3
"""Aggregate all sample counts, retaining unsupported and reused settings."""
import argparse
from pathlib import Path
import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from allshot_design import base,np,load,sha,atomic_json,verify_files,check_identity,result_root,COHORTS,REQUESTED
pd=base.pd
LABELS={'nsclc':'TCGA-NSCLC','brca':'TCGA-BRCA','crc':'TCGA-CRC','blca':'TCGA-BLCA'}
GROUPS={'tissue_architecture':'Tissue architecture','cell_composition':'Cell composition',
        'spatial_interactions':'Spatial interactions','tls':'TLS','tissue_geometry':'Tissue geometry'}
METRICS=['auroc_ovr','nll','brier','accuracy','balanced_accuracy','macro_f1','ece']


def read_rows(m):
    rows=[];selected=[];missing=[]
    for task in m['tasks']:
        folder=result_root(m,task)
        if task['shots']==16:
            design=load(folder/'design.json');results=[]
            for cond in design['conditions']:
                r=load(folder/cond['name']/'metrics.json')
                if r['manifest_identity']!=m['parent_identity'] or r['design_identity']!=design['identity']:
                    raise ValueError('Invalid reused result')
                verify_files(r['artifact_sha256']);results.append(r)
        else:
            p=folder/'results.json'
            if not p.exists():missing.append([task['cohort'],task['shots'],task['fold']]);continue
            result=load(p)
            if result['status']!='completed' or result['task_id']!=task['id'] or result['manifest_identity']!=m['identity']:
                raise ValueError('Invalid new result')
            verify_files(result['artifact_sha256']);results=result['conditions']
        if len(results)!=42:raise ValueError('Incomplete feature conditions')
        for r in results:
            cond=r['condition'];p=r['metrics']['test']['patient_metrics']
            rows.append({'cohort':task['cohort'],'panel':task['panel'],'shots':task['shots'],'fold':task['fold'],
                         'variant':'PathoTME-LR','reused_16shot':task['shots']==16,
                         **{k:cond[k] for k in ['name','kind','k','group','repeat']},
                         'selected_C':r['selected_C'],**{k:p[k] for k in METRICS},
                         'patients':r['metrics']['test']['patients']})
            if cond['kind']=='rfe':
                selected.extend({'cohort':task['cohort'],'panel':task['panel'],'shots':task['shots'],'fold':task['fold'],
                                 'k':cond['k'],'feature':n} for n in cond['feature_names'])
    return rows,selected,missing


def save_figure(root,name,fig):
    for ext in ['png','pdf','svg']:fig.savefig(root/f'{name}.{ext}')
    plt.close(fig)


def make_figures(root,m,agg):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    a=agg[agg.complete_five_fold]
    fig,axes=plt.subplots(4,5,figsize=(17,12),constrained_layout=True)
    for i,cohort in enumerate(COHORTS):
        for j,shot in enumerate(REQUESTED):
            ax=axes[i,j];g=a[(a.cohort==cohort)&(a.shots==shot)]
            ax.set_title(LABELS[cohort]+f' · {shot}-shot',fontsize=10)
            full=g[g.kind=='full']
            if len(full)!=1:
                ax.text(.5,.5,'Unsupported patient count' if shot not in m['support'][cohort]['supported_shots'] else 'Pending',ha='center',va='center',transform=ax.transAxes,fontsize=9)
                ax.set_axis_off();continue
            f=full.iloc[0]
            for repeat in range(10):
                r=g[(g.kind=='random')&(g['repeat']==repeat)].sort_values('k')
                ax.plot([*r.k,int(f.k)],100*np.array([*r.auroc_ovr_mean,f.auroc_ovr_mean]),color='#aaa',alpha=.45,lw=.7)
            r=g[g.kind=='rfe'].sort_values('k')
            ax.errorbar([*r.k,int(f.k)],100*np.array([*r.auroc_ovr_mean,f.auroc_ovr_mean]),
                        yerr=100*np.array([*r.auroc_ovr_fold_sd,f.auroc_ovr_fold_sd]),fmt='o-',color='#225ea8',capsize=2,markersize=3)
            random=g[g.kind=='random'].groupby('k').auroc_ovr_mean.mean()
            ax.plot(random.index,100*random.values,'s--',color='#666',markersize=3)
            ax.set_xticks([8,16,32,int(f.k)]);ax.grid(axis='y',alpha=.2)
            if j==0:ax.set_ylabel('Patient AUROC (%)')
            if i==3:ax.set_xlabel('Retained TME features')
    fig.suptitle('PathoTME-LR · subset-size curves at every supported shot\nBlue: training RFE; gray: ten random panels; dashed: random mean; error bars: fold SD, not CI\nFull panel: core62, except historical BRCA morph64',fontsize=12)
    save_figure(root,'subset_size_by_shot',fig)
    colors={'rfe_k8':'#1b9e77','rfe_k16':'#d95f02','rfe_k32':'#7570b3','full':'#222'}
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,cohort in zip(axes.ravel(),COHORTS):
        g=a[a.cohort==cohort]
        for name,color in colors.items():
            r=g[g.name==name].sort_values('shots')
            ax.errorbar(r.shots,100*r.auroc_ovr_mean,yerr=100*r.auroc_ovr_fold_sd,fmt='o-',capsize=2,color=color,
                        label='Full panel' if name=='full' else name.replace('rfe_k','')+' selected features',markersize=4)
        ax.set_title(LABELS[cohort]+' · '+('morph64' if cohort=='brca' else 'core62'))
        ax.set_xscale('log',base=2);ax.set_xticks(m['support'][cohort]['supported_shots'],m['support'][cohort]['supported_shots'])
        ax.set_xlabel('Training examples per class (shots)');ax.set_ylabel('Patient AUROC (%)');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('PathoTME-LR · sample-efficiency curves\nFive-fold means; error bars are fold SD, not confidence intervals',fontsize=12)
    save_figure(root,'shot_curves',fig)
    palette=['#225ea8','#d95f0e','#238b45','#88419d']
    for kind in ['leave_group_out','group_only']:
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for ax,cohort in zip(axes.ravel(),COHORTS):
            g=a[a.cohort==cohort];groups=list(base.groups_for(next(t for t in m['tasks'] if t['cohort']==cohort)))
            for group,color in zip(groups,palette):
                r=g[(g.kind==kind)&(g.group==group)].sort_values('shots')
                metric='delta_auroc_ovr' if kind=='leave_group_out' else 'auroc_ovr'
                ax.errorbar(r.shots,100*r[metric+'_mean'],yerr=100*r[metric+'_fold_sd'],fmt='o-',capsize=2,color=color,label=GROUPS[group],markersize=4)
            if kind=='leave_group_out':ax.axhline(0,color='#222',lw=.8)
            else:
                r=g[g.kind=='full'].sort_values('shots');ax.plot(r.shots,100*r.auroc_ovr_mean,'k--',label='Full panel')
            ax.set_title(LABELS[cohort]+' · '+('morph64' if cohort=='brca' else 'core62'))
            ax.set_xscale('log',base=2);ax.set_xticks(m['support'][cohort]['supported_shots'],m['support'][cohort]['supported_shots'])
            ax.set_xlabel('Training examples per class (shots)');ax.grid(axis='y',alpha=.2)
            ax.set_ylabel('AUROC change from full panel (pp)' if kind=='leave_group_out' else 'Patient AUROC (%)')
            ax.legend(fontsize=7,loc='best')
        fig.suptitle('PathoTME-LR · '+('remove each biological group' if kind=='leave_group_out' else 'each biological group alone')+'\nError bars: '+('SD of paired fold differences' if kind=='leave_group_out' else 'fold SD')+', not confidence intervals',fontsize=12)
        save_figure(root,'group_removal_by_shot' if kind=='leave_group_out' else 'group_only_by_shot',fig)


def report(root):
    root=Path(root);m=load(root/'manifest.json');check_identity(m)
    verify_files(m['reused_result_sha256'])
    rows,selected,missing=read_rows(m)
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('No completed results')
    if frame.duplicated(['cohort','shots','fold','name']).any():raise ValueError('Duplicate condition/fold')
    full=frame[frame.kind=='full'].set_index(['cohort','shots','fold'])
    for metric in METRICS:
        frame['delta_'+metric]=[r[metric]-float(full.loc[(r['cohort'],r['shots'],r['fold']),metric]) for r in rows]
    base.csv_write(root/'fold_metrics.csv',frame)
    keys=['cohort','panel','shots','variant','name','kind','k','group','repeat'];summary=[]
    for key,g in frame.groupby(keys,dropna=False,sort=True):
        row={**dict(zip(keys,key)),'folds':len(g),'complete_five_fold':set(g.fold)==set(range(5))}
        for metric in METRICS:
            for n in [metric,'delta_'+metric]:
                row[n+'_mean']=float(g[n].mean());row[n+'_fold_sd']=float(g[n].std(ddof=1)) if len(g)>1 else None
        summary.append(row)
    agg=pd.DataFrame(summary);base.csv_write(root/'condition_summary.csv',agg)
    frequencies=[]
    if selected:
        for (cohort,panel,shots,k),g in pd.DataFrame(selected).groupby(['cohort','panel','shots','k']):
            names=next(t['feature_names'] for t in m['tasks'] if t['cohort']==cohort)
            for n in names:frequencies.append({'cohort':cohort,'panel':panel,'shots':int(shots),'k':int(k),'feature':n,
                'selected_folds':int((g.feature==n).sum()),'evaluated_folds':int(g.fold.nunique()),
                'selection_frequency':float((g.feature==n).sum()/g.fold.nunique())})
    base.csv_write(root/'selection_frequency.csv',pd.DataFrame(frequencies))
    random_summary=[]
    for (cohort,shots,k),g in agg[(agg.kind=='random')&agg.complete_five_fold].groupby(['cohort','shots','k']):
        random_summary.append({'cohort':cohort,'shots':int(shots),'k':int(k),'random_panels':len(g),
                               'mean_auroc':float(g.auroc_ovr_mean.mean()),'min_auroc':float(g.auroc_ovr_mean.min()),
                               'max_auroc':float(g.auroc_ovr_mean.max()),'range_is_not_CI':True})
    base.csv_write(root/'random_panel_summary.csv',pd.DataFrame(random_summary))
    make_figures(root,m,agg)
    lines=['# PathoTME feature ablations across cohorts and shots','',
      'PathoTME-LR only. Shots count training examples per class; validation uses the same count per class. Feature subset sizes are a separate axis: 8, 16, 32, and the full 62/64 measurements.', '',
      'The original 16-shot results are reused by verified reference. New conditions inherit the exact RFE/C-grid/preprocessing/metric code. All feature masks and C choices are saved before test features are evaluated. Ten random panels per size remain identical across matching panels and shots.', '',
      '## Coverage','', '| Cohort | Completed supported shots | Unsupported shots |','|---|---|---|']
    for c in COHORTS:
        support=m['support'][c]
        lines.append('| '+LABELS[c]+' | '+', '.join(map(str,support['supported_shots']))+' | '+(', '.join(support['excluded']) or 'None')+' |')
    lines+=['','CRC has only 58–59 minority-class development patients per fold, so 32/64 shots cannot supply equally sized disjoint training and validation sets. BLCA has 98–99, excluding 64 shots. No patients are reused and validation is not reduced.', '',
      'All outer test memberships remain unchanged. NSCLC/BRCA replay the original max64 sampling permutation and reproduce existing 4/8/16 memberships. CRC/BLCA preserve their max16 phase anchors; BLCA32 adds unused patients to each phase. This is a separately recorded extension, not a replacement of prior splits.', '',
      '## Subset-size results','', '| Cohort | Shots | Full panel | 8 selected | 16 selected | 32 selected |','|---|---:|---:|---:|---:|---:|']
    for c in COHORTS:
        for shot in m['support'][c]['supported_shots']:
            g=agg[(agg.cohort==c)&(agg.shots==shot)&agg.complete_five_fold];values=[]
            for name in ['full','rfe_k8','rfe_k16','rfe_k32']:
                r=g[g.name==name]
                values.append(f'{100*r.iloc[0].auroc_ovr_mean:.2f} ± {100*r.iloc[0].auroc_ovr_fold_sd:.2f}' if len(r)==1 else 'pending')
            lines.append('| '+LABELS[c]+f' | {shot} | '+' | '.join(values)+' |')
    lines+=['','Patient AUROC percentages: five-fold mean ± fold SD, not confidence intervals. Learned subsets vary by training fold. All sizes and random controls are reported; no winning test subset is used to select a final panel.', '',
      '## Biological-group results','', '| Cohort | Shots | Biological group | Remove: ΔAUROC (pp) | Alone: AUROC (%) |','|---|---:|---|---:|---:|']
    for c in COHORTS:
        for shot in m['support'][c]['supported_shots']:
            g=agg[(agg.cohort==c)&(agg.shots==shot)&agg.complete_five_fold]
            for group in base.groups_for(next(t for t in m['tasks'] if t['cohort']==c)):
                values=[]
                for kind,metric in [('leave_group_out','delta_auroc_ovr_mean'),('group_only','auroc_ovr_mean')]:
                    r=g[(g.kind==kind)&(g.group==group)]
                    values.append(f'{100*r.iloc[0][metric]:+.2f}' if len(r)==1 else 'pending')
                lines.append('| '+LABELS[c]+f' | {shot} | '+GROUPS[group]+' | '+' | '.join(values)+' |')
    lines+=['','Core62 groups: tissue architecture 14, cell composition 28, spatial interactions 16, TLS 4. Historical BRCA morph64: tissue architecture 16, cell composition 28, spatial interactions 12, tissue geometry 8.', '',
      'All patient/slide metrics, including NLL and Brier, remain in the private fold outputs; the aggregate CSV retains all patient metrics. Selection frequency and random-panel spread are descriptive, not posterior inclusion probabilities or confidence intervals. These are exploratory LR sufficiency/redundancy studies, not causal biological necessity or validation of the original panel against excluded features. No neural-adapter effects or common optimal subset are established.', '',
      'Figure files: `subset_size_by_shot`, `shot_curves`, `group_removal_by_shot`, `group_only_by_shot`, each in PDF/PNG/SVG. No results are added to the public website.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    record={'status':'completed' if not missing else 'partial','manifest_identity':m['identity'],
            'condition_folds_completed':len(rows),'condition_folds_expected':m['counts']['total_condition_folds'],
            'reused_16shot_condition_folds':sum(r['shots']==16 for r in rows),
            'new_condition_folds_completed':sum(r['shots']!=16 for r in rows),
            'complete_five_fold_conditions':int(agg.complete_five_fold.sum()),'missing':missing,
            'supported_shots':{c:s['supported_shots'] for c,s in m['support'].items()},'neural_adapter_refits':0,'gpu_jobs':0,
            'artifact_sha256':{p.name:sha(p) for p in root.iterdir() if p.suffix in ['.csv','.png','.pdf','.svg']}}
    atomic_json(root/'summary.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ['artifact_sha256','missing']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    report(p.parse_args().output)
