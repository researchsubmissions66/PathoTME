"""Publication figures and a self-contained anonymous interactive TME atlas."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.lines import Line2D

COHORTS=['nsclc','brca','crc','blca']
INK='#243348';MUTED='#687789'
CLASS=['#397EA8','#D9914B']
OUTCOME={'Both correct':'#CDD6E1','Both wrong':'#716B90','Corrected':'#168F83','Worsened':'#C65A76'}
DIVERGING=LinearSegmentedColormap.from_list('rose_pearl_teal',['#AC3E62','#F3F1EE','#087F78'])
CONFIDENCE=LinearSegmentedColormap.from_list('pearl_blue',['#EEF2F5','#87B2BC','#22526F'])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def axes_style(ax,xy,projector='UMAP'):
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal',adjustable='box')
    lo=xy.min(0);hi=xy.max(0);margin=np.maximum((hi-lo)*.08,.1)
    ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1])
    # Consistent coordinates; arrows are orientation guides, not quantitative axes.
    for end in [(.15,.04),(.04,.15)]:ax.annotate('',xy=end,xytext=(.04,.04),xycoords='axes fraction',arrowprops={'arrowstyle':'-','color':'#ABB6C3','lw':.8})
    ax.text(.16,.025,projector+' 1',transform=ax.transAxes,fontsize=6,color=MUTED)
    ax.text(.02,.16,projector+' 2',transform=ax.transAxes,fontsize=6,color=MUTED,rotation=90,va='bottom')

def scatter_classes(ax,xy,y):
    # Labels do not determine draw order; fixed patient permutation prevents class overpainting.
    order=np.random.default_rng(20260915).permutation(len(xy))
    ax.scatter(*xy[order].T,s=10,c=np.array(CLASS)[y[order]],alpha=.83,edgecolors='white',linewidths=.22)

def legend_handle(color,label,marker='o'):
    return Line2D([],[],color='none',marker=marker,markerfacecolor=color,markeredgecolor='white',markersize=5,label=label)

def save(fig,root,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(root/(name+'.'+ext),dpi=600 if ext=='png' else 300,facecolor='white')
    plt.close(fig)

def outcome_arrays(group,y):
    idx=np.asarray(group['point_index']);p=np.asarray(group['probability']);n=np.asarray(group['native_probability']);labels=y[idx]
    a=(n>=.5)==labels;b=(p>=.5)==labels
    status=np.where(a&b,'Both correct',np.where(~a&~b,'Both wrong',np.where(~a&b,'Corrected','Worsened')))
    return idx,status,100*(p-n)*(2*labels-1)

def render(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());data=json.loads((root/'overlays.json').read_text())
    er=json.loads((root/'embedding_report.json').read_text())
    if er['manifest_identity']!=m['identity']:raise ValueError('Wrong projection manifest')
    if sha(root/'overlays.json')!=m['prepared_artifact_sha256'][str(root/'overlays.json')]:raise ValueError('Changed predictions')
    maps={}
    for c in COHORTS:
        p=root/c/'embedding.npz'
        if sha(p)!=er['cohorts'][c]['embedding_sha256']:raise ValueError('Changed coordinates')
        with np.load(p,allow_pickle=False) as z:maps[c]={k:z[k] for k in ['umap','pca']}
        data['cohorts'][c]['coordinates']={k:v.tolist() for k,v in maps[c].items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':INK,'axes.labelcolor':INK,
                         'axes.titleweight':'semibold','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    figure_dir=root/'figures';figure_dir.mkdir(exist_ok=True)
    for projector in ['umap','pca']:
        fig,axes=plt.subplots(2,2,figsize=(7.2,7.8));fig.subplots_adjust(left=.04,right=.97,bottom=.15,top=.82,hspace=.62,wspace=.20)
        fig.text(.055,.965,'The tumor microenvironment, in two dimensions',fontsize=14,weight='semibold')
        fig.text(.055,.927,projector.upper()+' of fixed TME measurements  ·  one point per patient',fontsize=9,color=MUTED)
        for i,(ax,c) in enumerate(zip(axes.ravel(),COHORTS)):
            info=data['cohorts'][c];xy=maps[c][projector];y=np.array(info['label_id'])
            scatter_classes(ax,xy,y);axes_style(ax,xy,projector.upper())
            ax.set_title(chr(65+i)+'   '+info['label'],loc='left',fontsize=10,pad=19)
            ax.text(0,1.035,f"{info['patients']:,} patients  ·  {info['panel']}",transform=ax.transAxes,color=MUTED,fontsize=7)
            ax.legend(handles=[legend_handle(CLASS[j],name+f'  (n = {int((y==j).sum())})') for j,name in enumerate(info['classes'])],
                      loc='upper left',bbox_to_anchor=(-.01,-.08),frameon=False,fontsize=6.8,ncol=1,handletextpad=.3,borderaxespad=0)
        fig.text(.055,.035,'Unsupervised, descriptive projection of all cohort covariates; labels are used only for color.\nMaps are fitted separately by cohort. Visual separation alone does not demonstrate predictive improvement.',fontsize=7,color=MUTED,linespacing=1.5)
        save(fig,figure_dir,'tme_'+projector+'_atlas')
    main=[]
    for c in COHORTS:
        match=[g for g in data['comparisons'] if g['cohort']==c and g['shots']==16 and g['method']=='ViLa' and g['encoder']=='CLIP-RN50' and g['variant']=='PathoTME-Adapter']
        if match and match[0]['complete']:main.append(match[0])
    if main:
        max_delta=max(np.max(np.abs(outcome_arrays(g,np.array(data['cohorts'][g['cohort']]['label_id']))[2])) for g in main)
        bound=max(1,float(np.ceil(max_delta/5)*5));norm=TwoSlopeNorm(0,vmin=-bound,vmax=bound)
        fig,axes=plt.subplots(len(main),3,figsize=(9.2,2.75*len(main)+1.65),squeeze=False)
        fig.subplots_adjust(left=.06,right=.96,bottom=.17,top=.85,hspace=.53,wspace=.23)
        fig.text(.06,.96,'Where adding TME changes the prediction',fontsize=15,weight='semibold')
        fig.text(.06,.921,'ViLa / CLIP-RN50  ·  16-shot  ·  Native → PathoTME-Adapter',fontsize=9,color=MUTED)
        for row,g in enumerate(main):
            c=g['cohort'];info=data['cohorts'][c];xy=maps[c]['umap'];y=np.array(info['label_id']);idx,status,delta=outcome_arrays(g,y)
            for ax in axes[row]:axes_style(ax,xy)
            scatter_classes(axes[row,0],xy,y)
            for name in OUTCOME:
                select=idx[status==name];axes[row,1].scatter(*xy[select].T,s=11 if name in ['Corrected','Worsened'] else 7,
                        c=OUTCOME[name],edgecolors='white',linewidths=.2,alpha=.94 if name in ['Corrected','Worsened'] else .7)
            sc=axes[row,2].scatter(*xy[idx].T,c=delta,cmap=DIVERGING,norm=norm,s=10,edgecolors='white',linewidths=.15)
            for col,title in enumerate(['Histologic subtype','Prediction changes','True-class probability change']):
                axes[row,col].set_title(chr(65+row*3+col)+'   '+title,loc='left',fontsize=8.5,pad=19)
            axes[row,0].text(0,1.035,info['label']+f"  ·  n = {info['patients']}",transform=axes[row,0].transAxes,fontsize=7,color=MUTED)
            axes[row,0].legend(handles=[legend_handle(CLASS[j],name) for j,name in enumerate(info['classes'])],
                   loc='upper left',bbox_to_anchor=(0,-.08),frameon=False,ncol=2,fontsize=7,borderaxespad=0)
            o=g['outcomes'];axes[row,1].text(0,-.10,f"{o['Corrected']} corrected  ·  {o['Worsened']} worsened",transform=axes[row,1].transAxes,fontsize=7,color=MUTED)
            axes[row,2].text(0,-.10,f"Fold-mean AUROC: {g['native_auroc']:.1f} → {g['auroc']:.1f}%  ({g['delta_auroc']:+.1f} pp)",transform=axes[row,2].transAxes,fontsize=7,color=INK)
        fig.legend(handles=[legend_handle(v,k) for k,v in OUTCOME.items()],loc='lower left',bbox_to_anchor=(.058,.083),ncol=4,frameon=False,fontsize=7,columnspacing=1.7)
        cax=fig.add_axes([.715,.102,.22,.013]);cb=fig.colorbar(sc,cax=cax,orientation='horizontal',ticks=[-bound,0,bound]);cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6,length=2);cb.set_label('Change in true-class probability (pp)',fontsize=6,labelpad=2)
        fig.text(.06,.022,'Same UMAP coordinates in every column. All predictions are from each patient’s held-out fold.\nCorrection uses a fixed 0.5 threshold; AUROC summarizes ranking. Both gains and harms are shown.',fontsize=7,color=MUTED,linespacing=1.5)
        save(fig,figure_dir,'tme_prediction_changes_vila_rn50_16shot')
    # TME-only prediction confidence at every supported sample count, on fixed coordinates.
    for c in COHORTS:
        groups=sorted([g for g in data['comparisons'] if g['cohort']==c and g['variant']=='PathoTME-LR'],key=lambda g:g['shots'])
        fig,axes=plt.subplots(1,len(groups),figsize=(2.5*len(groups),3.5),squeeze=False)
        fig.subplots_adjust(left=.035,right=.975,bottom=.24,top=.72,wspace=.18)
        info=data['cohorts'][c];fig.text(.035,.91,info['label']+'  ·  TME-only prediction across shots',fontsize=12,weight='semibold')
        fig.text(.035,.82,'PathoTME-LR  ·  held-out probability assigned to the true subtype',fontsize=8,color=MUTED)
        for ax,g in zip(axes[0],groups):
            xy=maps[c]['umap'];idx=np.array(g['point_index']);y=np.array(info['label_id'])[idx];p=np.array(g['probability']);true=np.where(y==1,p,1-p)
            sc=ax.scatter(*xy[idx].T,c=true,cmap=CONFIDENCE,vmin=0,vmax=1,s=8,edgecolors='white',linewidths=.15)
            axes_style(ax,xy);ax.set_title(f"{g['shots']}-shot",loc='left',fontsize=9,pad=8)
            ax.text(0,-.12,f"AUROC {g['auroc']:.1f} ± {g['auroc_sd']:.1f}%",transform=ax.transAxes,fontsize=7,color=MUTED)
        cax=fig.add_axes([.72,.09,.23,.025]);cb=fig.colorbar(sc,cax=cax,orientation='horizontal',ticks=[0,.5,1]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6,length=2)
        fig.text(.035,.06,'Fixed unsupervised UMAP; coordinates never use predictions.\nAUROC: five-fold mean ± fold SD, not a confidence interval.',fontsize=7,color=MUTED,linespacing=1.5)
        save(fig,figure_dir,c+'_lr_allshot_umap')
    # The browser receives only allowlisted anonymous fields; patient IDs and measurements stay private.
    payload=json.dumps(data,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=(Path(__file__).parent/'viewer.html').read_text()
    (root/'tme_atlas.html').write_text(template.replace('__DATA__',payload))
    rows=[]
    for g in data['comparisons']:
        rows.append({k:v for k,v in g.items() if k not in ['point_index','probability','native_probability','outcomes']})
    import pandas as pd
    pd.DataFrame(rows).to_csv(root/'comparison_summary.csv',index=False)
    text=['# TME feature maps and held-out prediction overlays','',
      'Four fixed UMAP maps and four PCA controls cover every eligible study patient. One point represents a patient, using the mean of available fixed-transformed slide measurements. Display-only imputation and standardization use cohort patient covariates. This is an unsupervised, posthoc transductive visualization; its coordinates are not held-out features and were not used to fit or select any classifier. Labels/predictions never enter the projection process.', '',
      'UMAP: 15 neighbors, min_dist 0.3, Euclidean metric, 500 epochs, spectral initialization, fixed seed 20260915, one UMAP worker. Parameters and main comparison were fixed before projection; no seed search, outcome-based filtering, supervised UMAP, cluster smoothing, point relocation or test threshold tuning is performed. PCA uses the same covariates as a projection-sensitivity control.', '',
      'Saved out-of-fold slide probabilities are averaged per patient. A patient occurs once, with predictions only from their own held-out fold. The patient-level display vector does not replace the classifier input. Prediction correction means Native wrong and TME correct at the fixed probability_1 >= 0.5 threshold; worsened means the reverse. Continuous color shows 100 × (TME minus Native true-class probability). Neither correction count nor average confidence change is AUROC.', '',
      '## Coverage','', '| Cohort | Patients | Slides | Panel |','|---|---:|---:|---|']
    for c in COHORTS:
        x=data['cohorts'][c];text.append(f"| {x['label']} | {x['patients']} | {x['slides']} | {x['panel']} |")
    text+=['',json.dumps(m['counts'],indent=2),'',
      'The viewer exposes every available method/encoder/shot/variant, including Zero and Shuffled controls. Missing patients are gray and partial fold counts are explicit. Main static paired figure uses the prespecified ViLa/CLIP-RN50/16-shot/Adapter comparison only for cohorts with all five folds. New cohorts are not silently replaced with another architecture or an incomplete estimate.', '',
      '## Files','',
      '- `tme_atlas.html`: self-contained offline viewer; anonymous point numbers, class, fold and saved probabilities. No raw measurements, patient IDs, private paths or external assets.',
      '- `figures/tme_umap_atlas.{pdf,svg,png}`: four-cohort subtype atlas.',
      '- `figures/tme_pca_atlas.{pdf,svg,png}`: linear projection control.',
      '- `figures/tme_prediction_changes_vila_rn50_16shot.{pdf,svg,png}`: matched correction and signed probability overlays.',
      '- `figures/*_lr_allshot_umap.{pdf,svg,png}`: full-panel LR at all supported shots.',
      '- `fold_metrics.csv`, `comparison_summary.csv`, `manifest.json`, `embedding_report.json`: exact snapshot and quantitative context.', '',
      'PDF and SVG retain vector points and editable text; PNG exports are 600 DPI. Fonts: DejaVu Sans, embedded TrueType in PDF. Subtype colors are blue/amber; correction colors are teal/rose with neutral unchanged cases. Signed color limits are symmetric and do not clip plotted values. No contours imply biological clusters and no UMAP decision boundary is fabricated.', '',
      'Plot metrics consistently average the saved CSV probabilities in float64. Two legacy adapter/zero fold AUROC discrepancies are explained by exact float32 patient-mean replay; maximum difference is 0.02956 percentage points. Original results remain unchanged. The private manifest retains each discrepancy and replay.', '',
      'The display is descriptive, not evidence that UMAP clusters are biological groups or that TME universally improves prediction. Report paired fold metrics, all gains/harms and projection dependence. Fold SD is not a confidence interval. Cohort-specific maps are not aligned, and historical BRCA uses morph64. All patient arrays remain private; nothing was added to the public website.', '',
      'Sources: [UMAP paper](https://arxiv.org/abs/1802.03426), [official reproducibility guidance](https://umap-learn.readthedocs.io/en/latest/reproducibility.html), [official clustering caveats](https://umap-learn.readthedocs.io/en/latest/clustering.html).']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    report={'manifest_identity':m['identity'],'cohort_maps':4,'PCA_controls':4,'main_paired_cohorts':[g['cohort'] for g in main],
            'counts':m['counts'],'render_amendment':{'reason':'atlas title and legend spacing; no coordinate/prediction changes','renderer_sha256':sha(__file__),'parent_renderer_sha256':sha(Path(__file__).with_name('render_maps.py'))},'figures':[p.name for p in sorted(figure_dir.glob('*.pdf'))],
            'output_sha256':{str(p.relative_to(root)):sha(p) for p in [*figure_dir.iterdir(),root/'tme_atlas.html',root/'comparison_summary.csv',root/'REPORT.md']}}
    (root/'render_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='output_sha256'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);render(p.parse_args().output)
