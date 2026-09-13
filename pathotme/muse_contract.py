"""Fixed scientific boundary for PathoTME's MUSE addition."""
from copy import deepcopy

PORT = 'pathotme_muse_semantic_tme_v1'
PANELS = {'nsclc': 'core62', 'brca': 'brca_morph64_v1'}


def build_config(base, study, cohort, encoder):
    """Retain the registered paired MUSE recipe and original prompt sources."""
    cfg = deepcopy(base)
    if cfg['task'] != cohort or cfg['backbone'] != encoder:
        raise ValueError('source configuration does not match requested group')
    cfg.update(pathotme_extension=PORT, benchmark=study,
               experiment=f'{study}_{cohort}_muse_{encoder.replace("-", "_")}',
               num_workers=0, max_batch_failure_rate=0.0)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    encoder = cfg.get('backbone')
    if encoder not in ('plip', 'clip-rn50'):
        raise ValueError('MUSE study requires PLIP or CLIP-RN50')
    plip = encoder == 'plip'
    key = 'plip_10x' if plip else 'clip_rn50_10x'
    space = 'hf:vinid/plip' if plip else 'openai/clip-rn50@official'
    width, text_width = (768, 512) if plip else (1024, 1024)
    expected = dict(pathotme_extension=PORT, method='muse', n_classes=2,
        shots=16, k=5, batch_size=1, feature_dim=width, embed_dim=text_width,
        feature_space_id=space + ('#vision-preprojection' if plip else ''),
        prompt_feature_space_id=space, feature_resolutions={'bag': '10x'},
        feature_sources={'bag': key}, feature_input_kinds={'bag': 'patch_bag'},
        feature_path_column=f'feature__{key}', include_metadata=True,
        encoder_provenance='adapted', upstream_fidelity='partial',
        muse_prompt_learning='feature_space_context_fallback',
        muse_recipe='cvpr_2026_paper_algorithm_1',
        muse_runtime='sfse_smmo_full_queue_feature_space_context_v1',
        muse_upstream_commit='9f2ec37bad5a10bb79616900ec017830b0bdfa0a',
        muse_text_context_policy='native_eot_truncate_77_v1',
        muse_semantic_updates_per_slide=20, muse_logit_fusion='paper_mean',
        muse_base_patch_filter='paper_top_20_percent', num_experts=8,
        num_selected=2, num_heads=8, retrieval_k=20, top_patch_ratio=0.2,
        n_ctx=16, dropout=0.25, epochs=200, es_patience=20, es_stop_epoch=80,
        early_stopping=True, checkpoint_monitor='val_error', optimizer='adam',
        lr=0.0001, weight_decay=0.00001, lr_scheduler=None,
        prompt_provenance='upstream', prompt_source='muse_upstream_description_csvs')
    drift = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
    if drift:
        raise ValueError(f'MUSE study contract drift: {drift}')
    labels = {'nsclc': {'LUAD': 0, 'LUSC': 1}, 'brca': {'IDC': 0, 'ILC': 1}}.get(cfg.get('task'))
    names = {'nsclc': ['lung adenocarcinoma', 'lung squamous cell carcinoma'],
             'brca': ['invasive ductal carcinoma', 'invasive lobular carcinoma']}.get(cfg.get('task'))
    if labels is None or cfg.get('label_dict') != labels or cfg.get('classnames') != names:
        raise ValueError('cohort/class binding mismatch')
    if list(cfg.get('prompt_csvs', {})) != names:
        raise ValueError('original ordered class-description banks required')
    if cfg.get('prompt_features') is not None or cfg.get('max_patches') is not None:
        raise ValueError('no substituted text tensors or new patch truncation')
    paired = cfg.get('encoder', {})
    if (paired.get('name') != encoder or paired.get('feature_space_id') != space
            or paired.get('feature_dim') != text_width
            or paired.get('weights') != cfg.get('backbone_weights')):
        raise ValueError('paired checkpoint/space mismatch')
    patch = cfg.get('patch_encoder', {})
    if (patch.get('name') != encoder or patch.get('feature_dim') != width
            or patch.get('feature_space_id') != expected['feature_space_id']
            or patch.get('resolution') != '10x'
            or patch.get('weights') != cfg['backbone_weights']):
        raise ValueError('patch encoder/representation mismatch')
    extension = cfg.get('encoder_extension', {})
    if (not isinstance(extension, dict) or extension.get('base_method') != 'muse'
            or extension.get('prompt_encoder') != encoder
            or extension.get('feature_encoder') != encoder
            or extension.get('alignment_mode') != 'learned_projection'):
        raise ValueError('registered paired MUSE extension provenance required')
    for key in ('lr', 'weight_decay'):
        if type(cfg.get(key)) is not float:
            raise TypeError(f'{key} must remain a float')
