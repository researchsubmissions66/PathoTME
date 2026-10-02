"""Immutable paired-hierarchy checks and patient-level MSCPT adapter exports."""
import json,os
from pathlib import Path
from pathotme.locked_tcga import rows
from pathotme.mscpt_contract import validate_config
from pathotme.focus_contract import validate_donors


def selected(launch,cohort,encoder,fold):
    matches = [p for p in launch['plans'] if (p['cohort'],p['encoder'],p['fold'])==(cohort,encoder,fold)]
    if len(matches)!=1: raise ValueError('expected one bound MSCPT fold')
    return matches[0]


def check_features(plan,cfg):
    validate_config(cfg)
    audit = json.loads(Path(plan['feature_inventory']).read_text())
    phases = {phase:rows(Path(cfg['split_dir'])/f"fold{plan['fold']}/{phase}.csv")
              for phase in ('train','val','test')}
    validate_donors(phases,json.loads(Path(plan['donor_maps']).read_text()))
    for values in phases.values():
        for row in values:
            item = audit['slides'][row['slide_id']]
            for key in ('feature_path_column_l','feature_path_column_s'):
                path = Path(os.path.expandvars(row[cfg[key]]))
                bound = item['files'][str(path)];stat = path.stat()
                if (stat.st_size,stat.st_mtime_ns)!=(bound['size'],bound['mtime_ns']):
                    raise ValueError(f'feature changed after hierarchy audit: {path}')


def evaluate(loader,bridge,model,metric_fn):
    import numpy as np
    import pandas as pd
    import torch
    from run_vila_guided import _metric_bundle
    probabilities,labels,metadata,attention=[],[],[],[]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            detail=bridge.eval_step_with_details(batch,model)
            probabilities.append(detail['probabilities'][0].cpu().numpy())
            labels.append(int(batch[-1].reshape(-1)[0]));metadata.append(detail['metadata'])
            names=model.conditioner.token_names
            row={}
            for scale,weight in detail['tme_group_attention'].items():
                weight=weight[0].cpu().numpy().reshape(2,-1,len(names)).mean(1)
                for c in range(2):
                    row.update({f'tme_attention_class{c}_{scale}_{name}':float(weight[c,i])
                                for i,name in enumerate(names)})
            attention.append(row)
    probabilities=np.asarray(probabilities);labels=np.asarray(labels)
    if probabilities.shape!=(len(labels),2) or not np.isfinite(probabilities).all():
        raise ValueError('invalid test probabilities')
    frame=pd.DataFrame(metadata);frame['label']=labels
    frame['prediction']=probabilities.argmax(-1)
    frame[['probability_0','probability_1']]=probabilities
    frame=pd.concat([frame,pd.DataFrame(attention)],axis=1)
    return frame,_metric_bundle(probabilities,labels,metadata,metric_fn)
