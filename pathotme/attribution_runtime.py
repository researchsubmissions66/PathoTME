"""Checkpoint reconstruction for the seven standalone PathoTME architectures."""
from __future__ import annotations

import json
import os
from pathlib import Path


def make_bridge(cfg, maps, arm, device):
    """Use the exact native/cross-encoder extension without refitting anything."""
    name, cohort, encoder = cfg['method'], cfg['task'], cfg['backbone']
    if name in ('vila_mil', 'mgpath'):
        cross = (name, encoder) in (('vila_mil', 'plip'), ('mgpath', 'clip-rn50'))
        if cross:
            from pathotme.cross_encoder_models import make_method
            return make_method(cfg, cohort, name, arm, maps, device)
        if name == 'mgpath' and cohort == 'brca':
            from pathotme.brca_mgpath_index_repair import IndexedBreastGuidedMGPathMethod
            from pathotme.locked_models import permute_table
            bridge = IndexedBreastGuidedMGPathMethod(cfg, device)
            if arm == 'shuffled':
                bridge.tme = permute_table(bridge.tme, maps)
            return bridge
        from pathotme.locked_models import make_method
        return make_method(cfg, cohort, name, arm, maps, device)
    if name == 'focus':
        from pathotme.focus_tme import FocusTMEMethod as cls
    elif name == 'muse':
        from pathotme.muse_tme import MUSETMEMethod as cls
    elif name == 'hive_mil':
        from pathotme.hive_tme import HiVETMEMethod as cls
    elif name == 'mscpt':
        from pathotme.mscpt_tme import MSCPTTMEMethod as cls
    elif name == 'dyko':
        from pathotme.dyko_tme import DyKoTMEMethod as cls
    else:
        raise ValueError('Unsupported PathoTME architecture')
    return cls(cfg, maps, arm, device)


def load_model(bound, arm, device):
    import torch
    from train import build_loaders, set_seed
    cfg = {**bound['config'], 'num_workers': 0, 'pin_memory': False}
    set_seed(cfg['seed'] + cfg['_fold_index'])
    maps = json.loads(Path(bound['completion']['plan']['donor_maps']).read_text())
    bridge = make_bridge(cfg, maps, arm, device)
    model = bridge.build_model()
    saved = torch.load(bound['directory'] / 'best.pt', map_location=device, weights_only=True)
    if saved['identity'] != bound['record']['identity']:
        raise ValueError('Loaded adapter identity mismatch')
    model.load_adapter_state_dict(saved['state'])
    if not bool(model.standardizer.fitted.item()):
        raise ValueError('Saved training-fold preprocessing is not fitted')
    model.eval()
    loader = build_loaders(cfg['method'], cfg, cfg['_fold_index'])[2]
    return bridge, model, loader


def one_batch(loader, slide_id):
    frame = getattr(loader.dataset, 'frame', None)
    if frame is None:
        frame = getattr(loader.dataset, 'df', None)
    if frame is None:
        raise ValueError('Cannot identify exact test dataset rows')
    indices = [i for i, value in enumerate(frame['slide_id'].astype(str)) if value == slide_id]
    if len(indices) != 1:
        raise ValueError('Select exactly one existing held-out slide')
    return loader.collate_fn([loader.dataset[indices[0]]])


def check_selected_features(bound, row):
    """Recheck the launch-audited size/mtime for only the consumed feature files."""
    inventory = json.loads(Path(bound['completion']['plan']['feature_inventory']).read_text())
    if bound['config']['method'] in ('hive_mil','mscpt'):
        inventory = inventory['slides'][row['slide_id']]['files']
    paths = []
    for key in ('feature_path_column', 'feature_path_column_s', 'feature_path_column_l'):
        column = bound['config'].get(key)
        if not column or column not in row:
            continue
        path = Path(os.path.expandvars(row[column])).expanduser().resolve()
        record = inventory.get(str(path))
        if record is None:
            raise ValueError(f'Selected feature file is not in the launch inventory: {path}')
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
            raise ValueError(f'Feature file changed since the launch audit: {path}')
        paths.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    if not paths:
        raise ValueError('No exact audited feature paths found')
    return paths
