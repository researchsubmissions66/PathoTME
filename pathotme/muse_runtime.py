"""Single-bag runtime checks and evaluation for the MUSE extension."""
import json
import os
from pathlib import Path
from pathotme.locked_tcga import rows
from pathotme.muse_contract import validate_config
from pathotme.focus_contract import validate_donors


def selected(launch, cohort, encoder, fold):
    found = [p for p in launch['plans'] if (p['cohort'], p['encoder'], p['fold']) == (cohort, encoder, fold)]
    if len(found) != 1:
        raise ValueError('expected exactly one bound MUSE plan')
    return found[0]


def check_features(plan, cfg):
    validate_config(cfg)
    inventory = json.loads(Path(plan['feature_inventory']).read_text())
    phases = {phase: rows(Path(cfg['split_dir']) / f"fold{plan['fold']}/{phase}.csv")
              for phase in ('train', 'val', 'test')}
    validate_donors(phases, json.loads(Path(plan['donor_maps']).read_text()))
    for values in phases.values():
        for row in values:
            path = Path(os.path.expandvars(row[cfg['feature_path_column']]))
            item = inventory[str(path)]
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
                raise ValueError(f'feature changed after header audit: {path}')


def evaluate(loader, bridge, model, metric_fn):
    """Export group attention by class query, with patient-level metrics.

    These are model attention weights, not causal or token-level attribution.
    Full patch attention and original-row indices are available from the
    bridge's eval_step_with_details without storing large dense arrays here.
    """
    import numpy as np
    import pandas as pd
    import torch
    from run_vila_guided import _metric_bundle
    probabilities, labels, metadata, attention = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            detail = bridge.eval_step_with_details(batch, model)
            probabilities.append(detail['probabilities'][0].cpu().numpy())
            labels.append(int(batch[-1].reshape(-1)[0]))
            metadata.append(detail['metadata'])
            weights = detail['tme_group_attention'][0].cpu().numpy()
            names = model.conditioner.token_names
            if weights.shape != (2, len(names)):
                raise ValueError('TME class-query/group attention shape mismatch')
            row = {f'tme_attention_query{q}_{name}': float(weights[q, t])
                   for q in range(2) for t, name in enumerate(names)}
            experts = detail['expert_weights'].cpu().numpy()
            if experts.shape != (2, 8):
                raise ValueError('MUSE expert-weight shape mismatch')
            row.update({f'expert_weight_query{q}_expert{e}': float(experts[q, e])
                        for q in range(2) for e in range(8)})
            attention.append(row)
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if probabilities.shape != (len(labels), 2) or not np.isfinite(probabilities).all():
        raise ValueError('invalid test probabilities')
    frame = pd.DataFrame(metadata)
    frame['label'] = labels
    frame['prediction'] = probabilities.argmax(-1)
    frame[['probability_0', 'probability_1']] = probabilities
    frame = pd.concat([frame, pd.DataFrame(attention)], axis=1)
    return frame, _metric_bundle(probabilities, labels, metadata, metric_fn)
