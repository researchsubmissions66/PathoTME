"""Locked PathoTME MSCPT feature-only comparison, PLIP and explicit RN50 port."""
from copy import deepcopy
PORT = 'pathotme_mscpt_multiscale_text_tme_v1'
TEXT_POLICY = 'mscpt_native_64_prefix12_layer_context_v1'


def build_config(base, paired, study, cohort, encoder):
    cfg = deepcopy(base)
    plip = encoder == 'plip'; key = encoder.replace('-', '_')
    for name in ('backbone', 'backbone_weights', 'encoder'):
        cfg[name] = deepcopy(paired[name])
    cfg.update(method='mscpt', task=cohort, pathotme_extension=PORT, benchmark=study,
        experiment=f'{study}_{cohort}_mscpt_{key}', feature_dim=768 if plip else 1024,
        model_feature_dim=512 if plip else 1024, feature_space_id=paired['feature_space_id'],
        prompt_feature_space_id=paired['encoder']['feature_space_id'],
        feature_sources={'low':f'{key}_5x','high':f'{key}_20x'},
        feature_resolutions={'low':'5x','high':'20x'},
        feature_path_column_l=f'feature__{key}_5x',feature_path_column_s=f'feature__{key}_20x',
        feat_data_dir='',selected_5x_dir='',num_workers=0,optimizer='adam',
        max_batch_failure_rate=0.0,mscpt_text_policy=TEXT_POLICY,
        input_mode='precomputed_shared_features',encoder_provenance='adapted',
        implementation_provenance=PORT, feature_projection='native_visual_projection' if plip else 'none',
        pathotme_encoder_extension='native_plip_deep_text' if plip else 'rn50_deep_text_1024_graph_v1',
        fidelity_note='Feature-only MSCPT on paired 5x/20x bags. PLIP retains the vendored deep text learner; '
          'RN50 ports its 64-position 2+10 context and layer-description prompting to the native text transformer '
          'and uses 1024D graph layers. Native graph arithmetic, selector ensemble, top-k aggregation and '
          'three-branch CE are retained. Deep visual prompting is inactive. No PGVL allowlist expansion.')
    cfg.pop('coverage_extension',None)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    encoder=cfg.get('backbone'); plip=encoder=='plip'; key=encoder.replace('-','_') if encoder else ''
    space='hf:vinid/plip' if plip else 'openai/clip-rn50@official'
    expected=dict(pathotme_extension=PORT,method='mscpt',shots=16,n_classes=2,batch_size=1,
        feature_dim=768 if plip else 1024,model_feature_dim=512 if plip else 1024,
        feature_space_id=space+('#vision-preprojection' if plip else ''),prompt_feature_space_id=space,
        input_mode='precomputed_shared_features',feature_resolutions={'low':'5x','high':'20x'},
        feature_path_column_l=f'feature__{key}_5x',feature_path_column_s=f'feature__{key}_20x',
        include_metadata=True,n_tpro=2,n_high=10,num_k=100,selection_topk_per_class=30,
        epochs=50,es_patience=10,es_stop_epoch=0,checkpoint_monitor='mscpt_val_best_score',
        mscpt_text_policy=TEXT_POLICY,upstream_fidelity='partial')
    drift={k:(cfg.get(k),v) for k,v in expected.items() if cfg.get(k)!=v}
    if encoder not in ('plip','clip-rn50') or drift: raise ValueError(f'MSCPT contract drift: {drift}')
    labels={'nsclc':{'LUAD':0,'LUSC':1},'brca':{'IDC':0,'ILC':1}}.get(cfg.get('task'))
    if cfg.get('label_dict')!=labels or labels is None: raise ValueError('MSCPT class binding mismatch')
    paired=cfg.get('encoder',{})
    if (paired.get('name')!=encoder or paired.get('weights')!=cfg['backbone_weights']
            or paired.get('feature_space_id')!=space or paired.get('feature_dim')!=expected['model_feature_dim']):
        raise ValueError('MSCPT paired checkpoint mismatch')
    if cfg.get('max_patches') is not None: raise ValueError('unregistered patch truncation')
    for key in ('lr','weight_decay'):
        if type(cfg.get(key)) is not float: raise TypeError(f'{key} must be numeric')
