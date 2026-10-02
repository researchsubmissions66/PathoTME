"""MSCPT multiscale semantic conditioning with the unchanged PGVL graph forward."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from methods.base import BaseMethod
from methods.mscpt.adapter import MSCPTMethod
from methods.mscpt.mscpt_model.mscpt import CustomCLIP, GcnPromptLearner
from methods.mscpt.selection import build_selection_text_features
from common.backbones import SwapPolicy, BackboneCapability as Cap
from common.models.paired_encoder_extension import build_paired_feature_projector
from pathotme.focus_tme import TMEQueryConditioner
from pathotme.guided_vila import FoldStandardizer
from pathotme.muse_data import load_tme
from pathotme.mscpt_contract import PORT, validate_config


class SemanticMSCPTCore(CustomCLIP):
    """Expose frozen scale descriptions through an identity boundary.

    The inherited forward keeps native selection, affinity, GCN, branch
    arithmetic, normalization and top-k, including upstream graph quirks.
    The property has no persistent tensor intervention or in-place mutation.
    """
    @property
    def text_features_zs(self):
        bank = self.frozen_text_bank
        return self.frozen_semantics(bank) if 'frozen_semantics' in self._modules else bank

    @text_features_zs.setter
    def text_features_zs(self, value):
        self.register_buffer('frozen_text_bank', value.detach().clone())

    def finish_boundaries(self):
        self.frozen_semantics = nn.Identity()
        value = self.text_features_ft
        del self.text_features_ft
        self.register_buffer('text_features_ft', value.detach().clone())


class PLIPMSCPTCore(SemanticMSCPTCore):
    """Native PLIP constructor with explicit device placement for frozen text IDs."""
    def __init__(self,raw,tokenizer,bank,cfg):
        from methods.mscpt.mscpt_model.mscpt import (MYPromptLearner,TextEncoder,
            TextEncoderZS,VisionPromptLearner,VisionEncoder)
        nn.Module.__init__(self)
        raw.eval().requires_grad_(False)
        self.prompt_learner=MYPromptLearner(list(bank),raw,cfg['n_tpro'],cfg['n_high'],tokenizer)
        self.gcn_prompt_learner_big=GcnPromptLearner(True)
        self.gcn_prompt_learner_small=GcnPromptLearner(True)
        # Preserve native initialization order/RNG, then freeze the unused RGB branch.
        self.vision_prompt_learner=VisionPromptLearner(raw,cfg['n_vpro']).requires_grad_(False)
        self.image_encoder=VisionEncoder(raw,cfg['n_vpro']).requires_grad_(False)
        self.text_encoder=TextEncoder(raw,cfg['n_tpro'],cfg['n_high'])
        self.text_encoder_zs=TextEncoderZS(raw)
        self.logit_scale=raw.logit_scale;self.model=raw;self.n_class=len(bank)
        self.high_patch_topk=cfg.get('high_patch_topk',5);self.low_patch_topk=cfg['num_k']
        self.selection_topk_per_class=cfg['selection_topk_per_class']
        device=next(raw.parameters()).device
        def tokenize(texts):
            tokens=tokenizer.batch_encode_plus(texts,max_length=64,add_special_tokens=True,
                return_token_type_ids=False,truncation=True,padding='max_length',return_attention_mask=True)
            return {k:torch.as_tensor(v,device=device) for k,v in tokens.items()}
        representations,features=[],[]
        with torch.no_grad():
            for branch in bank.values():
                tokens=tokenize(branch['small_mag'])
                final,layers=self.text_encoder_zs(tokens['input_ids'],tokens['attention_mask'])
                representations.append(F.normalize(final,dim=-1));features.append(F.normalize(layers,dim=-1))
        self.text_features_zs=torch.stack(representations);self.text_features_ft=torch.stack(features)
        self.text_prompts=bank;self.clip_model_proj=raw.visual_projection
        if cfg.get('selection_prompt_path'):
            with torch.no_grad():
                selection=build_selection_text_features(cfg['selection_prompt_path'],list(bank),
                    lambda texts:raw.get_text_features(**tokenize(texts)))
        else:selection=torch.empty(0,512,device=device)
        self.register_buffer('selection_text_features',selection)
        self.finish_boundaries()


class RN50MSCPTCore(SemanticMSCPTCore):
    def __init__(self, raw, bank, cfg):
        from torch_geometric.nn import GCNConv
        from pathotme.mscpt_text import RN50PromptLearner, RN50TextTower, frozen_rn50
        nn.Module.__init__(self)
        raw.float().eval().requires_grad_(False)
        self.prompt_learner = RN50PromptLearner(raw,bank,cfg['n_tpro'],cfg['n_high'])
        self.text_encoder = RN50TextTower(raw,cfg['n_tpro'],cfg['n_high'])
        # Reuse the same graph class and forward, changing only its shared width.
        self.gcn_prompt_learner_big = GcnPromptLearner(True)
        self.gcn_prompt_learner_small = GcnPromptLearner(True)
        for graph in (self.gcn_prompt_learner_big,self.gcn_prompt_learner_small):
            graph.conv1=GCNConv(1024,1024);graph.conv2=GCNConv(1024,1024)
        representations,features=[],[]
        for branch in bank.values():
            final,layers=frozen_rn50(raw,branch['small_mag'])
            representations.append(final);features.append(layers)
        self.text_features_zs=torch.stack(representations)
        self.text_features_ft=torch.stack(features)
        self.text_prompts=bank
        self.n_class=len(bank)
        self.high_patch_topk=cfg.get('high_patch_topk',5)
        self.low_patch_topk=cfg['num_k']
        self.selection_topk_per_class=cfg['selection_topk_per_class']
        self.register_buffer('logit_scale',raw.logit_scale.detach().float().clone())
        # The inherited forward consults dimensions; input is already shared.
        self.clip_model_proj=nn.Identity()
        self.clip_model_proj.in_features=self.clip_model_proj.out_features=1024
        if cfg.get('selection_prompt_path'):
            selection=build_selection_text_features(cfg['selection_prompt_path'],list(bank),
                lambda texts:frozen_rn50(raw,texts)[0])
        else: selection=torch.empty(0,1024)
        self.register_buffer('selection_text_features',selection)
        self.finish_boundaries()


class PairedMSCPT(nn.Module):
    def __init__(self,core,projector,input_dim,width):
        super().__init__()
        self.Custom_model=core
        self.paired_feature_projector=projector.requires_grad_(False)
        self.input_dim,self.model_dim=input_dim,width

    def train(self,mode=True):
        super().train(mode)
        self.paired_feature_projector.eval()
        self.Custom_model.text_encoder.eval()
        if hasattr(self.Custom_model,'model'): self.Custom_model.model.eval()
        return self

    def forward(self,data,train=True):
        low,high=(x.squeeze(0) if x.ndim==3 else x for x in data)
        if any(x.ndim!=2 or x.shape[1]!=self.input_dim for x in (low,high)):
            raise ValueError('MSCPT requires correctly sized 5x and 20x bags')
        low=self.paired_feature_projector(low.float());high=self.paired_feature_projector(high.float())
        result=self.Custom_model(low,high,train=train)
        if train: return tuple(x.unsqueeze(0) for x in result)
        return result[0].unsqueeze(0),result[1]


class MSCPTNativeMethod(MSCPTMethod):
    backbone_contract=replace(MSCPTMethod.backbone_contract,
        swap_policy=SwapPolicy.ALLOWLIST,supported_backbones=('plip','clip-rn50'),
        default_backbone='plip',feature_dims={'plip':(768,), 'clip-rn50':(1024,)},
        feature_boundaries={'plip':{'hf:vinid/plip#vision-preprojection':(768,)},
                            'clip-rn50':{'openai/clip-rn50@official':(1024,)}},
        required_capabilities=frozenset({Cap.PAIRED_TILE_TEXT}),
        rationale='Task-owned cached-feature MSCPT; RN50 deep text port, no active deep visual prompting.')

    def __init__(self,cfg,device='cuda:0'):
        validate_config(cfg)
        BaseMethod.__init__(self,cfg,device)

    def build_model(self):
        from pathotme.locked_tcga import sha
        path=Path(self.cfg['description_prompt_path'])
        if sha(path)!=self.cfg['description_prompt_sha256']: raise ValueError('MSCPT description bank changed')
        if self.cfg.get('selection_prompt_path') and sha(self.cfg['selection_prompt_path'])!=self.cfg['selection_prompt_sha256']:
            raise ValueError('MSCPT selector bank changed')
        source=json.loads(path.read_text());bank={c:source[c] for c in self.cfg['label_dict']}
        if any(len(b['small_mag'])!=10 for b in bank.values()): raise ValueError('ten frozen descriptions per class required')
        encoder=self.load_encoder(weights_path=self.cfg['backbone_weights']).freeze()
        raw=encoder.raw_model.float().eval().requires_grad_(False)
        projector,width=build_paired_feature_projector(encoder,feature_space_id=self.cfg['feature_space_id'],feature_dim=self.cfg['feature_dim'])
        if self.cfg['backbone']=='plip':
            core=PLIPMSCPTCore(raw,encoder.raw_tokenizer,bank,self.cfg)
        else: core=RN50MSCPTCore(raw,bank,self.cfg)
        model=PairedMSCPT(core,projector,self.cfg['feature_dim'],width).to(self.device)
        # Every trainable native parameter must belong to a prompt/graph learner.
        if any('prompt_learner' not in n for n,p in model.named_parameters() if p.requires_grad):
            raise ValueError('unexpected native MSCPT trainable parameters')
        return model


class TMEGuidedMSCPT(nn.Module):
    def __init__(self,base,cohort,tme_mode='actual',**kwargs):
        super().__init__()
        if tme_mode not in ('zero','actual'): raise ValueError('unknown TME mode')
        self.base=base.eval().requires_grad_(False)
        self.standardizer=FoldStandardizer(62 if cohort=='nsclc' else 64)
        self.conditioner=TMEQueryConditioner(base.model_dim,cohort,**kwargs)
        self.tme_mode=tme_mode

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def adapter_state_dict(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith('base.')}

    def load_adapter_state_dict(self,state):
        if set(state)!=set(self.adapter_state_dict()): raise ValueError('adapter checkpoint keys differ')
        self.load_state_dict({**self.state_dict(),**state},strict=True)

    def forward(self,high,low,label,tme,*,return_details=False):
        if tme.shape!=(1,self.standardizer.feature_count): raise ValueError('one TME row required')
        values=self.standardizer(tme)
        if self.tme_mode=='zero':values=torch.zeros_like(values)
        calls={}
        def hook(name):
            def condition(_module,_inputs,features):
                if name in calls: raise ValueError('MSCPT semantic boundary repeated')
                shape=features.shape
                conditioned,attention=self.conditioner(features.reshape(-1,shape[-1]),values)
                calls[name]=attention
                return conditioned.reshape(shape)
            return condition
        core=self.base.Custom_model
        handles=[core.frozen_semantics.register_forward_hook(hook('low')),
                 core.text_encoder.register_forward_hook(hook('high'))]
        try: logits=self.base((low,high),train=False)[0]
        finally:
            for handle in handles:handle.remove()
        if set(calls)!={'low','high'}: raise ValueError('both MSCPT semantic branches must be consumed')
        loss=F.cross_entropy(logits,label);probabilities=logits.softmax(1);prediction=probabilities.argmax(1,keepdim=True)
        if not return_details:return probabilities,prediction,loss
        return dict(logits=logits,probabilities=probabilities,prediction=prediction,loss=loss,
                    tme_group_attention=calls)


class MSCPTTMEMethod(MSCPTNativeMethod):
    extension_name=PORT
    def __init__(self,cfg,maps,arm,device='cuda:0'):
        if arm not in ('zero','actual','shuffled') or cfg.get('tme_mode')!=('zero' if arm=='zero' else 'actual'):
            raise ValueError('TME arm/mode mismatch')
        super().__init__(cfg,device)
        self.tme=load_tme(cfg['tme_feature_csv'],cfg['task'])
        if arm=='shuffled':
            original=self.tme.copy(deep=True)
            for mapping in maps.values():
                self.tme.loc[list(mapping)]=original.loc[list(mapping.values())].to_numpy(copy=True)

    def build_model(self):
        base=super().build_model()
        checkpoint=Path(self.cfg['base_checkpoint_dir'])/f"fold{self.cfg['_fold_index']}_best.pt"
        base.load_state_dict(torch.load(checkpoint,map_location=self.device,weights_only=True),strict=True)
        from train import set_seed
        set_seed(self.cfg['seed']+self.cfg['_fold_index'])
        return TMEGuidedMSCPT(base,self.cfg['task'],tme_mode=self.cfg['tme_mode'],
            hidden_dim=self.cfg['tme_hidden_dim'],num_heads=self.cfg['tme_attention_heads'],
            dropout=self.cfg['tme_dropout'],initial_gate=self.cfg['tme_initial_gate']).to(self.device)

    def prepare_fold(self,fold,model,train_loader):
        if fold!=self.cfg['_fold_index']: raise ValueError('fold mismatch')
        ids=train_loader.dataset.df.slide_id.astype(str).tolist()
        model.standardizer.fit(self.tme.loc[ids].to_numpy(dtype=np.float32))

    def inputs(self,batch,model):
        high,low,label=batch[0].to(self.device),batch[1].to(self.device),batch[-1].to(self.device)
        metadata={k:str(batch[-2][k] if isinstance(batch[-2][k],str) else batch[-2][k][0]) for k in ('slide_id','case_id')}
        tme=torch.as_tensor(self.tme.loc[metadata['slide_id']].to_numpy(dtype=np.float32),device=self.device).unsqueeze(0)
        return (high,low,label,tme),metadata

    def train_step(self,batch,model,optimizer,loss_fn=None):
        inputs,_=self.inputs(batch,model);optimizer.zero_grad(set_to_none=True)
        detail=model(*inputs,return_details=True);detail['loss'].backward();optimizer.step()
        return dict(loss=detail['loss'].item(),logits=detail['logits'].detach(),label=inputs[2])

    @torch.no_grad()
    def eval_step(self,batch,model,loss_fn=None):
        inputs,_=self.inputs(batch,model);detail=model(*inputs,return_details=True)
        return dict(loss=detail['loss'].item(),logits=detail['logits'],label=inputs[2])

    @torch.no_grad()
    def eval_step_with_details(self,batch,model):
        inputs,metadata=self.inputs(batch,model)
        return {**model(*inputs,return_details=True),'metadata':metadata}
