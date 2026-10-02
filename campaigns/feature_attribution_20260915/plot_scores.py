#!/usr/bin/env python3
"""Standalone research figures from the complete LR feature rankings."""
import argparse,csv,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with (a.output/'feature_rankings.csv').open() as f:rows=list(csv.DictReader(f))
    for level in ['feature','group']:
        fig,axes=plt.subplots(2,2,figsize=(20,15),layout='constrained')
        for ax,cohort in zip(axes.flat,['nsclc','brca','crc','blca']):
            selected=sorted([r for r in rows if r['variant']=='PathoTME-LR' and r['cohort']==cohort and r['class_index']=='1' and r['level']==level],key=lambda r:-float(r['mean_absolute_pp']))[:10]
            means=np.array([float(r['mean_absolute_pp']) for r in selected]);sd=np.array([float(r['fold_sd_absolute_pp']) for r in selected]);y=np.arange(len(selected))
            ax.barh(y,means,color='#7754a2',alpha=.9)
            ax.errorbar(means,y,xerr=sd,fmt='none',ecolor='#283347',capsize=2,lw=1)
            ax.set_yticks(y,['\n'.join(textwrap.wrap(r['feature_name'].replace('_',' ').lower(),49)) for r in selected],fontsize=8)
            ax.invert_yaxis();ax.set_xlim(left=min(0,float((means-sd).min())-.05))
            ax.axvline(0,color='#aaa',lw=.5)
            ax.set_title(f"TCGA-{cohort.upper()} · {selected[0]['class_name']}\n5 folds · {selected[0]['patients']} held-out patients",fontsize=13)
            ax.set_xlabel('Mean absolute patient probability change (pp) ± fold SD',fontsize=9)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'PathoTME-LR · Top 10 {level} sensitivities\nTraining-fold mean replacement; fold SD is descriptive, not a confidence interval',fontsize=16)
        for suffix in ['png','pdf','svg']:fig.savefig(a.output/f'lr_{level}_attribution.{suffix}',dpi=150)
        plt.close(fig)
    print('Saved feature and biological-group figures as PNG, PDF and SVG.')

if __name__=='__main__':main()
