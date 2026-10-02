"""Publication exports and an aggregate-only interactive correlation viewer."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,to_rgba
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram

COHORTS=['nsclc','brca','crc','blca']
INK='#243348';MUTED='#687789'
GROUPS={'tissue_architecture':('Tissue architecture','#397EA8'),
        'cell_composition':('Cell composition','#D9914B'),
        'spatial_interactions':('Spatial interactions','#168F83'),
        'tls':('TLS','#B36B88'),'tissue_geometry':('Tissue geometry','#8273A1')}
COLORS=['#397EA8','#168F83','#C65A76','#8273A1']
CMAP=LinearSegmentedColormap.from_list('rose_pearl_teal',['#AC3E62','#F6F4F0','#087F78'])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def save(fig,folder,name):
    for extension in ['pdf','svg','png']:fig.savefig(folder/(name+'.'+extension),dpi=600,facecolor='white')
    plt.close(fig)

def feature_label(name):
    # Display aliases only; exact names are retained in dictionaries, CSV and HTML.
    replacements=[('AVG_MIN_DISTANCE_OF_','Distance: '),('RATIO_OF_','Ratio: '),
        ('CELL_PERCENTAGE_','Cell proportion: '),('CELL_DENSITY_','Cell density: '),
        ('RELATIVE_AREA_','Area: '),('AROUND_CARCINOMA_CELL_IN_WHOLE_TUMOR_REGION','around carcinoma, whole tumor'),
        ('WHOLE_TUMOR_REGION','whole tumor'),('INNER_INVASIVE_MARGIN','inner margin'),('OUTER_INVASIVE_MARGIN','outer margin'),
        ('CARCINOMA_CELL','carcinoma'),('LOG1P_COUNT_TLS_','TLS log1p count: '),('TLS_IMMATURE_PRESENT','TLS immature present'),
        ('TLS_MATURE_PRESENT','TLS mature present')]
    for a,b in replacements:name=name.replace(a,b)
    return name.replace('_',' ').lower()

def heatmap_block(fig,record,x,y,size,detailed=False,title=''):
    width,height=fig.get_size_inches();box=lambda a,b,w,h:[a/width,b/height,w/width,h/height]
    n=len(record['order']);order=np.array(record['order']);corr=np.array(record['rho'])
    ax=fig.add_axes(box(x,y,size,size));im=ax.imshow(corr[np.ix_(order,order)],vmin=-1,vmax=1,cmap=CMAP,interpolation='nearest',aspect='equal')
    groups=[GROUPS[record['groups'][i]][1] for i in order]
    strip=fig.add_axes(box(x,y+size+.025,size,.065));strip.imshow(np.array([to_rgba(c) for c in groups])[None,:,:],aspect='auto');strip.set_axis_off()
    tree=fig.add_axes(box(x,y+size+.13,size,.6 if detailed else .42))
    with plt.rc_context({'lines.linewidth':.65}):
        result=dendrogram(np.array(record['linkage']),ax=tree,no_labels=True,color_threshold=0,above_threshold_color='#8795A4')
    if result['leaves']!=record['order']:raise ValueError('Dendrogram/heatmap order mismatch')
    tree.axhline(.15,color='#C65A76',lw=.6,ls=':');tree.set_ylim(0,1.03);tree.set_axis_off()
    tree.set_title(title,fontsize=9 if not detailed else 10,loc='left',pad=6,color=INK)
    for spine in ax.spines.values():spine.set_color('#D7DEE6');spine.set_linewidth(.5)
    if detailed:
        ax.set_xticks(range(n),[record['feature_ids'][i] for i in order],rotation=90,fontsize=5.5)
        ax.set_yticks(range(n),[record['feature_ids'][i]+'  '+feature_label(record['feature_names'][i]) for i in order],fontsize=5.7)
        ax.tick_params(length=0,pad=3)
        ax.set_xlabel('Feature IDs in the same clustered order; full names in the feature dictionary',fontsize=8,labelpad=10)
    else:
        ticks=[0,15,31,47,n-1]
        ax.set_xticks(ticks,[i+1 for i in ticks],fontsize=7);ax.set_yticks(ticks,[i+1 for i in ticks],fontsize=7)
        ax.tick_params(length=2,width=.5,pad=3)
        ax.set_xlabel('Clustered feature position',fontsize=7,labelpad=5)
    return im

def group_legend(fig,anchor=(.06,.08),size=8):
    handles=[Line2D([],[],marker='s',color='none',markerfacecolor=c,markeredgecolor='none',markersize=6,label=label) for label,c in GROUPS.values()]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=anchor,ncol=3,frameon=False,fontsize=size,columnspacing=1.7,handletextpad=.5)

def figures(root,records):
    folder=root/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':INK,'axes.labelcolor':INK,
                         'axes.titleweight':'semibold','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(8.5,10.5));fig.text(.075,.963,'How redundant is the TME feature panel?',fontsize=15,weight='semibold')
    fig.text(.075,.936,'Patient-level Spearman correlations  ·  features clustered by |ρ|',fontsize=9,color=MUTED)
    for i,c in enumerate(COHORTS):
        s=records[c]['summary'];title=chr(65+i)+'  '+s['label']+f" · n={s['patients']} · {s['panel']}"
        im=heatmap_block(fig,records[c],x=.65 if i%2==0 else 4.85,y=6.0 if i<2 else 1.7,size=3,title=title)
    group_legend(fig,anchor=(.066,.079),size=7)
    cax=fig.add_axes([.70,.09,.22,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-1,0,1]);cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7,length=2);cb.set_label('Signed Spearman ρ',fontsize=7,labelpad=3)
    fig.text(.075,.037,'Trees use complete linkage on 1 − |ρ|; heatmaps retain correlation direction. Colors above each matrix mark biology.\nOrders are cohort-specific. Correlation clusters describe redundancy, not independent biological signals.',fontsize=7,color=MUTED,linespacing=1.5)
    save(fig,folder,'tme_redundancy_atlas')
    for c in COHORTS:
        r=records[c];s=r['summary'];fig=plt.figure(figsize=(11.7,9.6))
        fig.text(.055,.956,s['label']+' · TME feature redundancy',fontsize=15,weight='semibold')
        fig.text(.055,.919,f"{s['patients']} patients · {s['features_total']} features · {s['strong_pairs_085']} pairs with |ρ| ≥ 0.85",fontsize=10,color=MUTED)
        im=heatmap_block(fig,r,x=4.6,y=1.2,size=6.3,detailed=True,title='Complete linkage · signed correlations · '+s['panel'])
        cax=fig.add_axes([11.1/11.7,1.2/9.6,.13/11.7,6.3/9.6]);cb=fig.colorbar(im,cax=cax,ticks=[-1,-.5,0,.5,1]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=8,length=2);cb.set_label('Spearman ρ',fontsize=8)
        group_legend(fig,anchor=(.05,.063),size=8)
        fig.text(.055,.025,'Strong anticorrelation is also grouped as redundancy. Dotted tree cut: |ρ| ≥ 0.85 within each complete-link cluster.\nDisplay labels are abbreviated; exact source names and every pairwise value are preserved in CSV and the interactive viewer.',fontsize=7,color=MUTED,linespacing=1.5)
        save(fig,folder,c+'_clustered_spearman')
    fig,axes=plt.subplots(2,2,figsize=(7.7,7.0));fig.subplots_adjust(left=.10,right=.96,bottom=.14,top=.81,hspace=.65,wspace=.32)
    fig.text(.10,.945,'How much variation do fewer dimensions retain?',fontsize=13,weight='semibold')
    fig.text(.10,.902,'PCA of the fixed, standardized TME matrix · all components retained in the analysis',fontsize=8,color=MUTED)
    for i,(ax,c) in enumerate(zip(axes.ravel(),COHORTS)):
        r=records[c];s=r['summary'];v=100*np.cumsum(r['variance_ratio']);k=np.arange(1,len(v)+1)
        ax.plot(k,v,color=COLORS[i],lw=1.8);ax.set_xlim(0,65);ax.set_ylim(0,103);ax.set_xticks([0,16,32,48,64]);ax.set_yticks([0,25,50,75,100])
        ax.set_title(chr(65+i)+'  '+s['label']+' · '+s['panel'],loc='left',fontsize=9,pad=9)
        for threshold in [80,90,95]:
            pc=s['pcs_'+str(threshold)];ax.plot([0,pc,pc],[threshold,threshold,0],ls=':' if threshold!=90 else '--',color='#B7C0CB',lw=.6)
            ax.scatter(pc,v[pc-1],s=15 if threshold==90 else 8,color=COLORS[i],zorder=3)
        ax.text(.96,.15,f"80 / 90 / 95%: {s['pcs_80']} / {s['pcs_90']} / {s['pcs_95']} PCs\nEffective rank: {s['effective_rank']:.1f}",ha='right',transform=ax.transAxes,fontsize=7,color=MUTED,linespacing=1.6)
        ax.set_xlabel('Principal components',fontsize=8);ax.set_ylabel('Cumulative variance (%)',fontsize=8);ax.tick_params(labelsize=7)
    fig.text(.10,.038,'Effective rank is the entropy-based dimension of the covariance spectrum. It is descriptive, not a biological count.\nPCA variance does not establish predictive sufficiency; no replacement panel or classifier is selected here.',fontsize=7,color=MUTED,linespacing=1.5)
    save(fig,folder,'pca_variance_curves')

def render(root):
    root=Path(root);m=load(root/'manifest.json')
    for p,h in m['artifact_sha256'].items():
        if sha(root/p)!=h:raise ValueError('Changed analysis output')
    records={c:load(root/c/'analysis.json') for c in COHORTS}
    figures(root,records)
    payload={'cohorts':records,'groups':GROUPS,'threshold':.85}
    encoded=json.dumps(payload,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (root/'redundancy_atlas.html').write_text((Path(__file__).parent/'viewer.html').read_text().replace('__DATA__',encoded))
    lines=['# TME feature redundancy','',
      'Descriptive analysis of the exact patient matrices used by the TME UMAP atlas. Four cohorts are analyzed separately; historical BRCA remains morph64. No new feature, classifier, attribution/performance analysis or Slurm job is added. The full cohort covariates enter this analysis, so it is not a held-out feature-selection experiment. It is independent of shot count.', '',
      '## Summary','',
      '| Cohort | Features | PCs for 80% | PCs for 90% | PCs for 95% | Effective rank | Participation ratio | Strong pairs |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in COHORTS:
        s=records[c]['summary'];lines.append(f"| {s['label']} | {s['features_total']} | {s['pcs_80']} | {s['pcs_90']} | {s['pcs_95']} | {s['effective_rank']:.2f} | {s['participation_ratio']:.2f} | {s['strong_pairs_085']} / {s['pair_count']} |")
    lines+=['', 'Strong pairs have absolute Spearman correlation at least 0.85. Both signs count as potential redundancy, while the heatmap retains the sign. PCA variance, effective linear dimensions and correlation groups are different summaries and do not count independent biological signals or establish which features can be removed without affecting AUROC.', '',
      '## Methods','',
      'Fixed panel transforms are applied at slide level. Available slide covariates are averaged within each patient; cohort-patient medians fill remaining missing entries, and the saved cohort means/population SDs standardize features exactly as in the UMAP display. Reconstructed matrices and the earlier two-component PCA variance ratios are checked. Class labels and predictions do not enter correlations, clustering or PCA. Constant columns are reported and excluded from these calculations.', '',
      'Spearman correlation uses average ranks for ties. Complete-link hierarchical clustering uses dissimilarity 1 − |rho| with optimal leaf ordering. At the prespecified 0.85 cut, every feature pair inside a non-singleton cluster has |rho| >= 0.85. Thresholds 0.80 and 0.90 provide descriptive sensitivity checks. Singletons are retained. No threshold is chosen by model outcomes and no representative feature is promoted to a replacement panel.', '',
      'PCA uses full SVD of standardized covariates, not the Spearman matrix. Effective rank = exp(−sum(p log p)) and participation ratio = 1/sum(p²), where p is the fraction of covariance eigenvalue mass. All components, loadings and cumulative variance fractions are retained. Correlation can reflect subtype mixtures, common denominators or compositional constraints; it does not establish causal biological relationships.', '',
      '## Missing-data sensitivity','',
      'Primary correlations use the same imputed patient matrix as UMAP. A second estimate uses only observed patient values for each feature pair, with no imputation; the exact number of patients per pair is reported. This is a sensitivity check, not a significance test. It does not undo missing-value conventions already built into the fixed source panel.', '',
      '| Cohort | Missing patient measurements | Minimum observed patients/pair | Max change in rho | Pairs crossing 0.85 cutoff |','|---|---:|---:|---:|---:|']
    for c in COHORTS:
        s=records[c]['summary'];lines.append(f"| {s['label']} | {s['missing_patient_measurements']} | {s['observed_pair_count_min']} | {s['imputation_max_abs_rho_change']:.4f} | {s['pairs_changing_085_status']} |")
    lines+=['','## Outputs','',
      '- `figures/tme_redundancy_atlas`: four-cohort overview with dendrograms and biological-group color strips.',
      '- `figures/*_clustered_spearman`: detailed per-cohort matrices with IDs and readable feature labels.',
      '- `figures/pca_variance_curves`: full cumulative spectra and dimension summaries.',
      '- All six figure sets have vector PDF/SVG and 600-DPI PNG exports.',
      '- `redundancy_atlas.html`: offline viewer with feature search, full-name pair details, raw/signed correlation and imputation sensitivity.',
      '- `feature_dictionary.csv`, `strong_pairs.csv`, `pairwise_correlations.csv`, `redundancy_clusters.csv`, `threshold_sensitivity.csv`: aggregate, named results.',
      '- Each cohort directory includes full correlation matrices, observed pair counts, PCA spectrum and loadings.', '',
      'The report/viewer/export package contains aggregate statistics only; no raw patient feature vectors, patient IDs or private paths are included. Private manifests retain source/input hashes. Existing UMAP, attribution, model results and public website remain unchanged.', '',
      'References: [SciPy complete linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html), [SciPy Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=[*sorted((root/'figures').iterdir()),root/'REPORT.md',root/'redundancy_atlas.html',
           *sorted(root.glob('*.csv')),*sorted(root.glob('*/*.csv'))]
    with zipfile.ZipFile(root/'PathoTME_redundancy_publication.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(root))
    result={'status':'completed','manifest_identity':m['identity'],'renderer_sha256':sha(__file__),
            'figure_sets':6,'package_files':len(files),'aggregate_only':True,
            'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in [*files,root/'PathoTME_redundancy_publication.zip']}}
    (root/'render_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);render(p.parse_args().output)
