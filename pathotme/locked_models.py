"""New BRCA MGPATH panel bridge; existing native and PathoTME code is unchanged."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methods.mgpath.adapter import MGPathMethod
from pathotme.brca_features import panel_spec, transform_rows
from pathotme.brca_vila import BreastTMETokenizer
from pathotme.brca_vila_adapter import BreastGuidedViLaMethod
from pathotme.guided_mgpath import TMEGuidedMGPath
from pathotme.guided_mgpath_adapter import TMEGuidedMGPathMethod, one
from pathotme.guided_vila import FoldStandardizer
from pathotme.guided_vila_adapter import TMEGuidedViLaMethod


class BreastGuidedMGPath(TMEGuidedMGPath):
    """Keep native PLIP graphs/projection/Sinkhorn and use the frozen BRCA panel."""

    def __init__(self, base_model, **kwargs):
        super().__init__(base_model, **kwargs)
        self.standardizer = FoldStandardizer(feature_count=64)
        self.conditioner.tokenizer = BreastTMETokenizer(
            self.conditioner.hidden_dim, dropout=kwargs.get('dropout', 0.1))


class BreastGuidedMGPathMethod(TMEGuidedMGPathMethod):
    extension_name = 'pathotme_brca_mgpath_guided_morph64_v1'

    def __init__(self, cfg, device='cuda:0'):
        MGPathMethod.__init__(self, cfg, device)
        if self.is_encoder_extension or self.backbone_name != 'plip':
            raise ValueError('native PLIP required')
        self.names = panel_spec('brca_morph64_v1')['feature_names']
        with Path(cfg['tme_feature_csv']).open() as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ['slide_id', *self.names]:
                raise ValueError('BRCA panel column order mismatch')
            values = list(reader)
        self.tme = pd.DataFrame(transform_rows(values, 'brca_morph64_v1'),
                                index=[r['slide_id'] for r in values], columns=self.names)
        if self.tme.index.has_duplicates:
            raise ValueError('duplicate TME rows')

    def build_model(self):
        base = MGPathMethod.build_model(self)
        checkpoint = Path(self.cfg['base_checkpoint_dir']) / f"fold{self.cfg['_fold_index']}_best.pt"
        base.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True), strict=True)
        return BreastGuidedMGPath(base, tme_mode=self.cfg['tme_mode'],
            hidden_dim=self.cfg['tme_hidden_dim'], num_heads=self.cfg['tme_attention_heads'],
            dropout=self.cfg['tme_dropout'], initial_gate=self.cfg['tme_initial_gate']).to(self.device)

    def prepare_fold(self, fold, model, train_loader):
        if fold != self.cfg['_fold_index']:
            raise ValueError('fold mismatch')
        ids = train_loader.dataset.frame['slide_id'].astype(str).tolist()
        model.standardizer.fit(self.tme.loc[ids, self.names].to_numpy(dtype=np.float32))

    def inputs(self, batch):
        low, low_edges, high, high_edges, label = self._unpack(batch, self.device)
        metadata = batch[-2]
        values = self.tme.loc[one(metadata, 'slide_id'), self.names].to_numpy(dtype=np.float32)
        tme = torch.as_tensor(values, device=self.device).unsqueeze(0)
        return (low, low_edges, high, high_edges, tme), label, metadata


def permute_table(table, maps):
    """Copy complete transformed rows with a frozen, split-local donor mapping."""
    out = table.copy(deep=True)
    for mapping in maps.values():
        recipients, donors = list(mapping), list(mapping.values())
        if set(recipients) != set(donors) or len(donors) != len(set(donors)):
            raise ValueError('shuffle must be a bijection within each split')
        out.loc[recipients] = table.loc[donors].to_numpy(copy=True)
    return out


def make_method(cfg, cohort, method, arm, maps, device):
    cls = {('nsclc', 'vila_mil'): TMEGuidedViLaMethod,
           ('brca', 'vila_mil'): BreastGuidedViLaMethod,
           ('nsclc', 'mgpath'): TMEGuidedMGPathMethod,
           ('brca', 'mgpath'): BreastGuidedMGPathMethod}[cohort, method]
    bridge = cls(cfg, device=device)
    if arm == 'shuffled':
        key = '_tme' if method == 'vila_mil' else 'tme'
        setattr(bridge, key, permute_table(getattr(bridge, key), maps))
    return bridge
