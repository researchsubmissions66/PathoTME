"""Paired HiVE-MIL ports and TME residuals before filtering and graph aggregation."""
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from methods.base import BaseMethod
from methods.hive_mil.adapter import HiVEMILMethod
from methods.hive_mil.model import CustomCLIP, HierHeteroGNN
from methods.hive_mil.prompts import load_hierarchical_prompts
from common.backbones import SwapPolicy
from common.models.paired_encoder_extension import build_paired_feature_projector
from pathotme.focus_tme import TMEQueryConditioner
from pathotme.guided_vila import FoldStandardizer
from pathotme.muse_data import load_tme
from pathotme.hive_contract import PORT, validate_config
from pathotme.hive_text import build_text_port


class PairedHiVECore(CustomCLIP):
    """Use the unmodified vendored forward, graph, filtering and HTCL methods."""
    def __init__(self, prompt_learner, text_encoder, width, logit_scale):
        nn.Module.__init__(self)
        self.num_classes = 2
        self.filter_alpha = 0.5
        self.dtype = torch.float32
        self.dim = width
        self.graph_prompt_learner = HierHeteroGNN(width,width,width)
        self.prompt_learner, self.text_encoder = prompt_learner,text_encoder
        self.register_buffer('logit_scale', logit_scale.detach().float().clone())
        self.mag_ratio, self.num_low_mag_texts, self.num_high_mag_subtexts = 16,4,3


class PairedHiVE(nn.Module):
    def __init__(self, core, projector, input_dim):
        super().__init__()
        self.custom_vlm = core
        self.paired_feature_projector = projector.requires_grad_(False)
        self.input_dim, self.model_dim = input_dim,core.dim
        self.contrastive_lambda = 0.5
        # Native HiVE trains the token learner and graph prompt learner only.
        for name,p in self.custom_vlm.named_parameters():
            p.requires_grad_('prompt_learner' in name)
        self.custom_vlm.text_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.paired_feature_projector.eval()
        self.custom_vlm.text_encoder.eval()
        return self

    def forward(self, low, high, label=None):
        if (low.ndim != 2 or low.shape[1] != self.input_dim
                or high.shape != (len(low),16,self.input_dim)):
            raise ValueError('paired HiVE requires low [N,D] and high [N,16,D]')
        low = self.paired_feature_projector(low.float())
        high = self.paired_feature_projector(high.float())
        logits,contrastive = self.custom_vlm(low,high)
        logits = logits.unsqueeze(0)
        if label is None:
            return logits
        return logits,F.cross_entropy(logits,label)+self.contrastive_lambda*contrastive


class HiVENativeMethod(HiVEMILMethod):
    """PathoTME-owned PLIP/RN50 ports, without changing the PGVL allowlist."""
    backbone_contract = replace(HiVEMILMethod.backbone_contract,
        swap_policy=SwapPolicy.ALLOWLIST, supported_backbones=('plip','clip-rn50'),
        default_backbone='plip', feature_dims={'plip':(768,), 'clip-rn50':(1024,)},
        feature_boundaries={'plip':{'hf:vinid/plip#vision-preprojection':(768,)},
                            'clip-rn50':{'openai/clip-rn50@official':(1024,)}})

    def __init__(self,cfg,device='cuda:0'):
        validate_config(cfg)
        BaseMethod.__init__(self,cfg,device)

    def build_model(self):
        from pathotme.locked_tcga import sha
        bank = load_hierarchical_prompts(self.cfg['text_prompt_path'],self.cfg['label_dict'])
        if sha(self.cfg['text_prompt_path']) != self.cfg['text_prompt_file_sha256']:
            raise ValueError('original HiVE prompt hash mismatch')
        encoder = self.load_encoder(weights_path=self.cfg['backbone_weights']).freeze()
        learner,tower = build_text_port(encoder,bank,self.cfg.get('hive_token_ids_sha256'))
        projector,width = build_paired_feature_projector(encoder,
            feature_space_id=self.cfg['feature_space_id'],feature_dim=self.cfg['feature_dim'])
        core = PairedHiVECore(learner,tower,width,encoder.raw_model.logit_scale)
        return PairedHiVE(core,projector,self.cfg['feature_dim']).to(self.device)


class TMEGuidedHiVE(nn.Module):
    """Condition all 32 hierarchical text nodes; retain the frozen graph."""
    def __init__(self,base,cohort,tme_mode='actual',**kwargs):
        super().__init__()
        if tme_mode not in ('actual','zero'):
            raise ValueError('unknown TME mode')
        self.base = base.eval().requires_grad_(False)
        self.standardizer = FoldStandardizer(62 if cohort=='nsclc' else 64)
        self.conditioner = TMEQueryConditioner(base.model_dim,cohort,**kwargs)
        self.tme_mode = tme_mode

    def train(self,mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def adapter_state_dict(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith('base.')}

    def load_adapter_state_dict(self,state):
        if set(state) != set(self.adapter_state_dict()):
            raise ValueError('adapter checkpoint keys differ')
        self.load_state_dict({**self.state_dict(),**state},strict=True)

    def forward(self,low,high,label,tme,*,return_details=False):
        if tme.shape != (1,self.standardizer.feature_count):
            raise ValueError('one correctly sized TME row required')
        values = self.standardizer(tme)
        if self.tme_mode=='zero': values = torch.zeros_like(values)
        calls=[]
        def condition(_module,_inputs,features):
            if len(features)!=2 or any(v.shape != (16,self.base.model_dim) for v in features.values()):
                raise ValueError('native class-ordered 4+12 description hierarchy required')
            queries = torch.cat(list(features.values()),dim=0)
            conditioned,attention = self.conditioner(queries,values)
            calls.append(attention)
            return OrderedDict((name,conditioned[i*16:(i+1)*16])
                               for i,name in enumerate(features))
        # Each process owns its fold model. Restore the boundary on any error;
        # the output graph still retains conditioner gradients after removal.
        handle = self.base.custom_vlm.text_encoder.register_forward_hook(condition)
        try:
            logits = self.base(low,high)
        finally:
            handle.remove()
        if len(calls)!=1:
            raise ValueError('expected exactly one hierarchical text-encoder call')
        loss = F.cross_entropy(logits,label)
        probability = logits.softmax(1)
        prediction = probability.topk(1,dim=1)[1]
        if not return_details: return probability,prediction,loss
        return dict(logits=logits,probabilities=probability,prediction=prediction,loss=loss,
            tme_group_attention=calls[0], retained_parent_mask=self.base.custom_vlm.x5_mask,
            low_text_embeddings=self.base.custom_vlm.x5_text_embeddings,
            high_text_embeddings=self.base.custom_vlm.x20_text_embeddings)


class HiVETMEMethod(HiVENativeMethod):
    extension_name = PORT
    def __init__(self,cfg,maps,arm,device='cuda:0'):
        if arm not in ('zero','actual','shuffled'):
            raise ValueError('unknown adapter arm')
        if cfg.get('tme_mode') != ('zero' if arm=='zero' else 'actual'):
            raise ValueError('TME input mode differs from arm')
        super().__init__(cfg,device)
        self.tme = load_tme(cfg['tme_feature_csv'],cfg['task'])
        if arm=='shuffled':
            original = self.tme.copy(deep=True)
            for mapping in maps.values():
                self.tme.loc[list(mapping)] = original.loc[list(mapping.values())].to_numpy(copy=True)

    def build_model(self):
        base = super().build_model()
        checkpoint = Path(self.cfg['base_checkpoint_dir'])/f"fold{self.cfg['_fold_index']}_best.pt"
        base.load_state_dict(torch.load(checkpoint,map_location=self.device,weights_only=True),strict=True)
        from train import set_seed
        set_seed(self.cfg['seed']+self.cfg['_fold_index'])
        return TMEGuidedHiVE(base,self.cfg['task'],tme_mode=self.cfg['tme_mode'],
            hidden_dim=self.cfg['tme_hidden_dim'],num_heads=self.cfg['tme_attention_heads'],
            dropout=self.cfg['tme_dropout'],initial_gate=self.cfg['tme_initial_gate']).to(self.device)

    def prepare_fold(self,fold,model,train_loader):
        if fold!=self.cfg['_fold_index']: raise ValueError('fold mismatch')
        ids = train_loader.dataset.frame.slide_id.astype(str).tolist()
        model.standardizer.fit(self.tme.loc[ids].to_numpy(dtype=np.float32))

    def inputs(self,batch,model):
        low,high,label = self._unpack_batch(batch,self.device)
        metadata = {key:str(batch[-2][key] if isinstance(batch[-2][key],str) else batch[-2][key][0])
                    for key in ('slide_id','case_id')}
        tme = torch.as_tensor(self.tme.loc[metadata['slide_id']].to_numpy(dtype=np.float32),
                              device=self.device).unsqueeze(0)
        return (low,high,label,tme),metadata

    def train_step(self,batch,model,optimizer,loss_fn=None):
        inputs,_ = self.inputs(batch,model)
        optimizer.zero_grad(set_to_none=True)
        detail = model(*inputs,return_details=True)
        detail['loss'].backward();optimizer.step()
        return dict(loss=detail['loss'].item(),logits=detail['logits'].detach(),label=inputs[2])

    @torch.no_grad()
    def eval_step(self,batch,model,loss_fn=None):
        inputs,_ = self.inputs(batch,model)
        detail = model(*inputs,return_details=True)
        return dict(loss=detail['loss'].item(),logits=detail['logits'],label=inputs[2])

    @torch.no_grad()
    def eval_step_with_details(self,batch,model):
        inputs,metadata = self.inputs(batch,model)
        return {**model(*inputs,return_details=True),'metadata':metadata}
