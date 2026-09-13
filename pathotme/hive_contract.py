"""Fixed paired-encoder boundary for the fifth PathoTME architecture."""
from copy import deepcopy

PORT = 'pathotme_hive_paired_token16_tme_v1'
UPSTREAM = 'fa5ccec1a99db510e9add85b318e6241acb1aecd'
TEXT_POLICY = 'prefix16_native_eot_truncate_77_v1'


def build_config(base, paired, study, cohort, encoder):
    cfg = deepcopy(base)
    if base['task'] != cohort or base['method'] != 'hive_mil':
        raise ValueError('native HiVE cohort source required')
    plip = encoder == 'plip'
    key = 'plip' if plip else 'clip_rn50'
    width, shared = (768, 512) if plip else (1024, 1024)
    cfg.update(backbone=encoder, backbone_weights=paired['backbone_weights'],
        encoder=deepcopy(paired['encoder']), feature_space_id=paired['feature_space_id'],
        feature_dim=width, hive_model_dim=shared,
        prompt_feature_space_id=paired['encoder']['feature_space_id'],
        pathotme_extension=PORT, benchmark=study,
        experiment=f'{study}_{cohort}_hive_{key}', num_workers=0,
        max_batch_failure_rate=0.0, optimizer='adam',
        feature_sources={'low': f'{key}_5x', 'high': f'{key}_20x'},
        feature_path_column_l=f'feature__{key}_5x', feature_path_column_s=f'feature__{key}_20x',
        encoder_provenance='adapted', implementation_provenance=PORT,
        hive_text_context_policy=TEXT_POLICY,
        fidelity_note='PathoTME-owned paired native token16 text-tower extension. '
          'PLIP uses its frozen native visual projection; RN50 uses its shared 1024D space. '
          'Native hierarchy, graph, filtering, logit arithmetic and CE+HTCL baseline loss are retained. '
          'Paired 77-position prefix/EOT consumption replaces CONCH context; no source wording changes.',
        encoder_extension=dict(status='extended', owner='PathoTME', base_method='hive_mil',
          feature_encoder=encoder, prompt_encoder=encoder, strategy='paired_frozen_native_token16',
          feature_checkpoint=paired['backbone_weights'], prompt_checkpoint=paired['backbone_weights'],
          alignment_mode='native_visual_projection' if plip else 'native_shared'))
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    encoder = cfg.get('backbone')
    if encoder not in ('plip', 'clip-rn50'):
        raise ValueError('HiVE study requires PLIP or CLIP-RN50')
    plip = encoder == 'plip'; key = 'plip' if plip else 'clip_rn50'
    space = 'hf:vinid/plip' if plip else 'openai/clip-rn50@official'
    raw, shared = (768, 512) if plip else (1024, 1024)
    expected = dict(pathotme_extension=PORT, method='hive_mil', n_classes=2, shots=16, k=5,
        batch_size=1, num_workers=0, max_batch_failure_rate=0.0,
        feature_dim=raw, hive_model_dim=shared,
        feature_space_id=space+('#vision-preprojection' if plip else ''),
        prompt_feature_space_id=space, feature_resolutions={'low':'5x','high':'20x'},
        feature_input_kinds={'low':'patch_bag','high':'patch_bag'},
        feature_sources={'low':f'{key}_5x','high':f'{key}_20x'},
        feature_path_column_l=f'feature__{key}_5x', feature_path_column_s=f'feature__{key}_20x',
        include_metadata=True, encoder_provenance='adapted', upstream_fidelity='partial',
        upstream_commit=UPSTREAM, hive_text_context_policy=TEXT_POLICY,
        n_ctx=16, class_specific_token=False, class_token_position='end',
        num_low_mag_texts=4, num_high_mag_subtexts=3, max_children=16,
        hierarchy_geometry='per_slide_hdf5', low_mag='5x', high_mag='20x',
        filter_alpha=0.5, contrastive_lambda=0.5,
        epochs=50, es_patience=10, es_stop_epoch=0, early_stopping=True,
        checkpoint_monitor='val_error', optimizer='adam', lr=0.0001,
        weight_decay=0.00001, lr_scheduler=None,
        prompt_source='hive_mil_upstream_gpt4o_hierarchical_bank', prompt_provenance='derived')
    drift = {k:(cfg.get(k),v) for k,v in expected.items() if cfg.get(k)!=v}
    if drift:
        raise ValueError(f'HiVE study contract drift: {drift}')
    labels = {'nsclc':{'LUAD':0,'LUSC':1}, 'brca':{'IDC':0,'ILC':1}}.get(cfg.get('task'))
    if labels is None or cfg.get('label_dict') != labels:
        raise ValueError('cohort/class binding mismatch')
    paired = cfg.get('encoder', {})
    if (paired.get('name') != encoder or paired.get('feature_dim') != shared
            or paired.get('feature_space_id') != space or paired.get('weights') != cfg['backbone_weights']):
        raise ValueError('paired checkpoint/feature space mismatch')
    extension = cfg.get('encoder_extension', {})
    if (extension.get('owner') != 'PathoTME' or extension.get('base_method') != 'hive_mil'
            or extension.get('feature_encoder') != encoder or extension.get('prompt_encoder') != encoder
            or extension.get('strategy') != 'paired_frozen_native_token16'):
        raise ValueError('explicit paired HiVE port provenance required')
    for name in ['lr','weight_decay']:
        if type(cfg.get(name)) is not float:
            raise TypeError(f'{name} must remain a float')
    if cfg.get('max_patches') is not None:
        raise ValueError('no new patch truncation is allowed')
