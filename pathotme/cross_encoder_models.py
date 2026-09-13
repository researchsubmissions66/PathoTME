"""Task-owned cross-encoder PathoTME extensions; frozen PGVL sources are unchanged."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from methods.base import BaseMethod
from methods.mgpath.adapter import MGPathMethod
from methods.mgpath.model import MGPathModel
from methods.mgpath.prompts import load_prompt_bank, UPSTREAM_COMMIT
from common.backbones import BackboneCapability as Cap, FeatureLevel, MethodBackboneContract, SwapPolicy
from common.models.paired_encoder_extension import FeatureSpacePromptLearner, build_paired_feature_projector, project_paired_features, paired_logit_scale
from pathotme.brca_vila_adapter import BreastGuidedViLaMethod
from pathotme.guided_vila_adapter import TMEGuidedViLaMethod
from pathotme.guided_mgpath import TMEGuidedMGPath
from pathotme.guided_mgpath_adapter import TMEGuidedMGPathMethod, one
from pathotme.locked_models import BreastGuidedMGPath, BreastGuidedMGPathMethod, permute_table
from pathotme.features import FEATURE_NAMES, transform_nsclc_features, validate_feature_table
from pathotme.cross_encoder_contract import CLIP_PORT,CLIP_TEXT_POLICY,validate_clip_config
from pathotme.brca_mgpath_index_repair import IndexedBreastGuidedMGPathMethod


class CLIPMGPathMethod(MGPathMethod):
    """Explicit RN50 port: four feature contexts, native graph/centers/Sinkhorn.

    This is a new PathoTME extension, not a registered PGVL native MGPATH
    condition. The name retains the graph loader's dispatch key only.
    """
    extension_name = CLIP_PORT
    backbone_contract = MethodBackboneContract(
        method=CLIP_PORT, feature_level=FeatureLevel.DUAL_SCALE_PATCH_BAG,
        swap_policy=SwapPolicy.FIXED, default_backbone='clip-rn50',
        supported_backbones=('clip-rn50',), feature_dims={'clip-rn50': (1024,)},
        feature_boundaries={'clip-rn50': {'openai/clip-rn50@official': (1024,)}},
        required_capabilities=frozenset({Cap.TEXT_ENCODE, Cap.PAIRED_TILE_TEXT}),
        enforce_native_dimension=True, require_feature_space=True,
        rationale='PathoTME-owned RN50 shared-space MGPATH port with four feature contexts.')

    def __init__(self, cfg, device='cuda:0'):
        validate_clip_config(cfg)
        BaseMethod.__init__(self, cfg, device)
        expected = {'pathotme_encoder_extension': CLIP_PORT,
            'pathotme_text_policy': CLIP_TEXT_POLICY, 'feature_dim': 1024,
            'prompt_views': 4, 'image_centers': 64, 'type_gnn': 'gat_conv',
            'ratio_graph': 0.2, 'ot_epsilon': 0.1, 'ot_iterations': 100,
            'upstream_commit': UPSTREAM_COMMIT, 'feature_projection': 'none',
            'feature_resolutions': {'low': '5x', 'high': '10x'}}
        drift = {k:(cfg.get(k),v) for k,v in expected.items() if cfg.get(k)!=v}
        if drift: raise ValueError(f'RN50 extension contract mismatch: {drift}')
        self.is_encoder_extension = True

    def build_model(self):
        import clip
        bank=load_prompt_bank(self.cfg['text_prompt_path'],self.cfg['label_dict'],self.cfg['prompt_class_bindings'])
        if bank.file_sha256 != self.cfg['text_prompt_file_sha256']:
            raise ValueError('prompt bank changed')
        encoder=self.load_encoder(weights_path=self.cfg['backbone_weights']).freeze()
        projector,width=build_paired_feature_projector(encoder,
            feature_space_id=self.cfg['feature_space_id'],feature_dim=self.cfg['feature_dim'])
        # Preserve full bank bytes. The consumed prefix/EOT policy is explicit,
        # matching the existing native PLIP path's 77-position prefix boundary.
        tokens=clip.tokenize(list(bank.prompts),context_length=77,truncate=True).to(self.device)
        with torch.no_grad():features=F.normalize(encoder.raw_model.encode_text(tokens).float(),dim=-1)
        prompt=FeatureSpacePromptLearner(features,views=4,class_specific=False)
        result=MGPathModel(None,None,bank.prompts,n_classes=self.cfg['n_classes'],
            source_dim=1024,shared_dim=width,ratio_graph=0.2,prompt_module=prompt,
            projector=projector,logit_scale=paired_logit_scale(encoder.raw_model)).to(self.device)
        result.register_buffer('pathotme_consumed_prompt_tokens',tokens.detach().clone())
        return result


class ProjectedViLaInputs:
    """Apply the checkpoint's exact PLIP visual projection before TME guidance."""
    def _inputs(self,batch,model):
        if self.cfg['backbone']!='plip' or not self.is_encoder_extension:
            raise ValueError('this added pairing requires the registered PLIP ViLa port')
        if len(batch)!=4 or not isinstance(batch[2],dict):raise ValueError('invalid ViLa batch')
        low,high=(self._slide_bag(x.to(self.device)) for x in batch[:2])
        low,high=(project_paired_features(model.base,x) for x in (low,high))
        inputs=(low,torch.zeros(len(low),2,device=self.device),high,
                torch.zeros(len(high),2,device=self.device),batch[-1].to(self.device),self._tme_values(batch[2]))
        return inputs,batch[2]


class PLIPGuidedViLa(ProjectedViLaInputs,TMEGuidedViLaMethod):pass
class PLIPBreastGuidedViLa(ProjectedViLaInputs,BreastGuidedViLaMethod):pass


class CLIPGuidedMGPath(TMEGuidedMGPathMethod):
    """Use the same core62 conditioner over the declared frozen RN50 port."""
    def __init__(self,cfg,device='cuda:0'):
        # The task-owned RN50 class validates the encoder contract explicitly.
        checked=CLIPMGPathMethod(cfg,device)
        self.cfg,self.device,self.backbone_name=checked.cfg,checked.device,checked.backbone_name
        self.is_encoder_extension=True
        table=validate_feature_table(pd.read_csv(cfg['tme_feature_csv']))
        transformed=transform_nsclc_features(table)
        transformed.index=table['slide_id'].astype(str)
        self.tme=transformed
        if tuple(self.tme.columns)!=FEATURE_NAMES or self.tme.index.has_duplicates:
            raise ValueError('core62 schema mismatch')

    def build_model(self):
        base=CLIPMGPathMethod(self.cfg,self.device).build_model()
        path=Path(self.cfg['base_checkpoint_dir'])/f"fold{self.cfg['_fold_index']}_best.pt"
        base.load_state_dict(torch.load(path,map_location=self.device,weights_only=True),strict=True)
        wrapper=BreastGuidedMGPath if self.cfg['task']=='brca' else TMEGuidedMGPath
        return wrapper(base,tme_mode=self.cfg['tme_mode'],hidden_dim=self.cfg['tme_hidden_dim'],
            num_heads=self.cfg['tme_attention_heads'],dropout=self.cfg['tme_dropout'],
            initial_gate=self.cfg['tme_initial_gate']).to(self.device)


class CLIPBreastGuidedMGPath(CLIPGuidedMGPath):
    def __init__(self,cfg,device='cuda:0'):
        import csv
        from pathotme.brca_features import panel_spec,transform_rows
        checked=CLIPMGPathMethod(cfg,device)
        self.cfg,self.device,self.backbone_name=checked.cfg,checked.device,checked.backbone_name
        self.is_encoder_extension=True;self.names=panel_spec('brca_morph64_v1')['feature_names']
        with Path(cfg['tme_feature_csv']).open() as h:
            reader=csv.DictReader(h)
            if reader.fieldnames!=['slide_id',*self.names]:raise ValueError('BRCA panel order mismatch')
            rows=list(reader)
        self.tme=pd.DataFrame(transform_rows(rows,'brca_morph64_v1'),
            columns=self.names,index=[r['slide_id'] for r in rows])
        if self.tme.index.has_duplicates:raise ValueError('duplicate TME slides')
    prepare_fold=BreastGuidedMGPathMethod.prepare_fold
    inputs=IndexedBreastGuidedMGPathMethod.inputs


def make_method(cfg,cohort,method,arm,maps,device):
    cls={('nsclc','vila_mil'):PLIPGuidedViLa,('brca','vila_mil'):PLIPBreastGuidedViLa,
         ('nsclc','mgpath'):CLIPGuidedMGPath,('brca','mgpath'):CLIPBreastGuidedMGPath}[cohort,method]
    bridge=cls(cfg,device=device)
    if arm=='shuffled':
        key='_tme' if method=='vila_mil' else 'tme'
        setattr(bridge,key,permute_table(getattr(bridge,key),maps))
    return bridge
