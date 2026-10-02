"""CRC/BLCA task bindings over inherited PathoTME model equations.

No original module globals, task names or source files are changed. Native
constructors receive a local copy of their globals pointing to the new exact
configuration contract. Forward, loss and optimizer implementations are inherited.
"""
import hashlib,json,math
from pathlib import Path
from types import FunctionType
import numpy as np
import pandas as pd
import torch
from torch import nn
from methods.vila_mil.adapter import ViLaMILMethod
from methods.mgpath.adapter import MGPathMethod
from pathotme.cross_encoder_models import CLIPMGPathMethod,ProjectedViLaInputs
from pathotme.guided_vila_adapter import TMEGuidedViLaMethod
from pathotme.guided_mgpath_adapter import TMEGuidedMGPathMethod
from pathotme.guided_vila import FoldStandardizer,SemanticTMETokenizer,TMEGuidedViLaMIL
from pathotme.guided_mgpath import TMEGuidedMGPath
from pathotme.focus_tme import FocusNativeMethod,FocusTMEMethod,TMEGuidedFOCUS,TMEQueryConditioner
from pathotme.muse_tme import MUSENativeMethod,MUSETMEMethod,TMEGuidedMUSE
from pathotme.hive_tme import HiVENativeMethod,HiVETMEMethod,TMEGuidedHiVE
from pathotme.dyko_tme import DyKoNativeMethod,DyKoTMEMethod,TMEGuidedDyKo
from pathotme.mscpt_tme import MSCPTNativeMethod,MSCPTTMEMethod,TMEGuidedMSCPT
from pathotme.features import FEATURE_NAMES,validate_feature_table,transform_nsclc_features
from pathotme.shared_panel import common_spec

LABELS={'crc':{'ADENO_NOS':0,'MUCINOUS':1},'blca':{'NON_PAPILLARY':0,'PAPILLARY':1}}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_config(cfg):
    """Check the full frozen native recipe or its declared adapter overlay."""
    if cfg.get('task') not in LABELS or cfg.get('label_dict')!=LABELS[cfg['task']]:
        raise ValueError('CRC/BLCA class binding mismatch')
    path=Path(cfg['cohort_contract_path'])
    if sha(path)!=cfg['cohort_contract_sha256']:raise ValueError('Cohort contract changed')
    contract=json.loads(path.read_text());expected=contract['native_config']
    mutable={'results_dir','_fold_index','k_start','k_end','cohort_contract_path','cohort_contract_sha256',
             'pathotme_arm','base_checkpoint_dir','tme_mode','tme_panel','tme_feature_csv',
             'tme_hidden_dim','tme_attention_heads','tme_dropout','tme_initial_gate',
             'optimizer','lr','weight_decay','lr_scheduler'}
    if set(cfg)-set(expected)-mutable:raise ValueError('Unexpected configuration fields')
    drift={k:(cfg.get(k),v) for k,v in expected.items() if k not in mutable and cfg.get(k)!=v}
    if drift:raise ValueError(f'Frozen new-cohort recipe drift: {drift}')
    arm=cfg.get('pathotme_arm','native')
    if arm not in ['native','zero','actual','shuffled']:raise ValueError('Invalid arm')
    fold=cfg.get('_fold_index',cfg.get('k_start'))
    if type(fold) is not int or fold not in range(5) or cfg.get('k_start')!=fold or cfg.get('k_end')!=fold+1:raise ValueError('Invalid fold range')
    optimization=expected if arm=='native' else {**contract['adapter'],'lr_scheduler':None}
    if any(cfg.get(k)!=optimization.get(k) for k in ['optimizer','lr','weight_decay','lr_scheduler']):raise ValueError('Optimizer recipe mismatch')
    if any(type(cfg[k]) is not float for k in ['lr','weight_decay']):raise TypeError('Numeric optimizer values required')
    if arm!='native':
        if cfg.get('tme_mode')!=('zero' if arm=='zero' else 'actual') or cfg.get('tme_panel')!='shared_core62_v1':raise ValueError('TME arm/panel mismatch')
        for key in ['hidden_dim','attention_heads','dropout','initial_gate']:
            if cfg.get('tme_'+key)!=contract['adapter'][key]:raise ValueError('Conditioner recipe drift')
        if cfg.get('tme_feature_csv')!=contract['tme_csv']:raise ValueError('Wrong cohort TME table')
    if contract['feature_schema_sha256']!=common_spec()['feature_schema_sha256']:raise ValueError('Common panel identity changed')
    for path,digest in contract['asset_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Changed cohort asset: {path}')

def rebound(function,**bindings):
    """Reuse constructor bytecode with task-local validator bindings only."""
    result=FunctionType(function.__code__,{**function.__globals__,**bindings},function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result

ORIGINAL_NATIVE={'vila_mil':ViLaMILMethod,'mgpath':MGPathMethod,'focus':FocusNativeMethod,
                 'muse':MUSENativeMethod,'hive_mil':HiVENativeMethod,'dyko':DyKoNativeMethod,'mscpt':MSCPTNativeMethod}
ORIGINAL_TME={'vila_mil':TMEGuidedViLaMethod,'mgpath':TMEGuidedMGPathMethod,
              'focus':FocusTMEMethod,'muse':MUSETMEMethod,'hive_mil':HiVETMEMethod,
              'dyko':DyKoTMEMethod,'mscpt':MSCPTTMEMethod}
_NATIVE={};_ADAPTER={}

def native_type(method,encoder):
    key=method,encoder
    if key not in _NATIVE:
        original=CLIPMGPathMethod if key==('mgpath','clip-rn50') else ORIGINAL_NATIVE[method]
        constructor=rebound(original.__init__,validate_config=validate_config,validate_clip_config=validate_config)
        def initialize(self,cfg,device='cuda:0'):
            validate_config(cfg)
            constructor(self,cfg,device)
        overrides={'__init__':initialize}
        if key==('mscpt','plip'):
            def build_exact_class_keys(self):
                model=original.build_model(self)
                learner=model.Custom_model.prompt_learner
                keys=list(model.Custom_model.text_prompts)
                if learner.classnames!=[k.replace('_',' ') for k in keys]:raise ValueError('Unexpected native MSCPT class normalization')
                # The names are dictionary lookup keys, never prompt wording.
                # Preserve canonical bank bytes and class order exactly.
                learner.classnames=keys
                return model
            overrides['build_model']=build_exact_class_keys
        _NATIVE[key]=type('Cohort'+original.__name__,(original,),overrides)
    return _NATIVE[key]

class SharedQueryConditioner(TMEQueryConditioner):
    def __init__(self,model_dim,cohort,hidden_dim=128,num_heads=4,dropout=0.1,initial_gate=0.1):
        nn.Module.__init__(self)
        if cohort not in LABELS or hidden_dim%num_heads or not 0<initial_gate<1:raise ValueError('Invalid common-panel conditioner')
        self.tokenizer=SemanticTMETokenizer(hidden_dim,dropout=dropout)
        self.token_names=tuple('low_'+n for n in self.tokenizer.LOW_TOKEN_NAMES)+tuple('high_'+n for n in self.tokenizer.HIGH_TOKEN_NAMES)
        self.query_projection=nn.Linear(model_dim,hidden_dim,bias=False)
        self.cross_attention=nn.MultiheadAttention(hidden_dim,num_heads,dropout=dropout)
        self.output_projection=nn.Linear(hidden_dim,model_dim,bias=False)
        self.gate_logit=nn.Parameter(torch.tensor(math.log(initial_gate/(1-initial_gate))))

GUIDED={'focus':TMEGuidedFOCUS,'muse':TMEGuidedMUSE,'hive_mil':TMEGuidedHiVE,'dyko':TMEGuidedDyKo,'mscpt':TMEGuidedMSCPT}

def guided_model(method,base,cfg):
    options=dict(tme_mode=cfg['tme_mode'],hidden_dim=cfg['tme_hidden_dim'],num_heads=cfg['tme_attention_heads'],dropout=cfg['tme_dropout'],initial_gate=cfg['tme_initial_gate'])
    if method=='vila_mil':return TMEGuidedViLaMIL(base,**options)
    if method=='mgpath':return TMEGuidedMGPath(base,**options)
    def initialize(self,base,cohort,**kwargs):
        nn.Module.__init__(self)
        self.base=base.eval().requires_grad_(False)
        self.standardizer=FoldStandardizer(62)
        width=(base.D if method=='focus' else base.class_text_features.shape[-1] if method=='muse'
               else base.classifier.in_features if method=='dyko' else base.model_dim)
        self.tme_mode=kwargs.pop('tme_mode')
        self.conditioner=SharedQueryConditioner(width,cohort,**kwargs)
    wrapper=type('Common62'+GUIDED[method].__name__,(GUIDED[method],),{'__init__':initialize})
    return wrapper(base,cfg['task'],**options)

def adapter_type(method,encoder):
    key=method,encoder
    if key in _ADAPTER:return _ADAPTER[key]
    native=native_type(method,encoder);original=ORIGINAL_TME[method]
    def initialize(self,cfg,maps,arm,device='cuda:0'):
        if cfg.get('pathotme_arm')!=arm or arm=='native':raise ValueError('Adapter arm mismatch')
        native.__init__(self,cfg,device)
        self.fold=cfg['_fold_index']
        table=validate_feature_table(pd.read_csv(cfg['tme_feature_csv']))
        transformed=transform_nsclc_features(table);transformed.index=table.slide_id.astype(str)
        if transformed.index.has_duplicates:raise ValueError('Duplicate TME identities')
        if arm=='shuffled':
            before=transformed.copy(deep=True)
            for mapping in maps.values():transformed.loc[list(mapping)]=before.loc[list(mapping.values())].to_numpy(copy=True)
        setattr(self,'_tme' if method=='vila_mil' else 'tme',transformed)
    def build(self):
        base=native.build_model(self)
        checkpoint=Path(self.cfg['base_checkpoint_dir'])/f"fold{self.cfg['_fold_index']}_best.pt"
        base.load_state_dict(torch.load(checkpoint,map_location=self.device,weights_only=True),strict=True)
        from train import set_seed
        set_seed(self.cfg['seed']+self.cfg['_fold_index'])
        return guided_model(method,base,self.cfg).to(self.device)
    overrides={'__init__':initialize,'build_model':build,'backbone_contract':native.backbone_contract,
               'extension_name':'pathotme_crc_blca_shared_core62_v1'}
    if key==('vila_mil','plip'):overrides['_inputs']=ProjectedViLaInputs._inputs
    _ADAPTER[key]=type('Common62'+original.__name__,(original,),overrides)
    return _ADAPTER[key]
