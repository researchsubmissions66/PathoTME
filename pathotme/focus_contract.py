"""Scientific boundary of the separately owned FOCUS TME experiment."""
from copy import deepcopy

PORT = 'pathotme_focus_query_tme_v1'
CLIP_PORT = 'pathotme_focus_clip_rn50_feature_context_v1'
STRATEGY = 'paired_feature_context_v1'
PANELS = {'nsclc': 'core62', 'brca': 'brca_morph64_v1'}


def build_config(base, clip, source, study, cohort, encoder):
    """Derive a FOCUS condition from registered recipe and feature sources."""
    cfg = deepcopy(base)
    key = 'plip_20x' if encoder == 'plip' else 'clip_rn50_20x'
    if encoder == 'clip-rn50':
        for name in ('backbone', 'backbone_weights', 'encoder'):
            cfg[name] = deepcopy(clip[name])
        cfg.update(feature_dim=1024, model_feature_dim=1024,
                   feature_space_id=source['feature_space_id'],
                   prompt_feature_space_id=source['feature_space_id'],
                   feature_projection='none', encoder_extension=None,
                   pathotme_encoder_extension=CLIP_PORT,
                   implementation_provenance='pathotme_extended_paired_feature_context',
                   fidelity_note='PathoTME-owned RN50 shared-space FOCUS extension. '
                   'Frozen paired text embeddings with one shared trainable feature context '
                   'replace CONCH token prompting. Native FOCUS compression, attention, '
                   'classifier and CE are retained. Not upstream CONCH FOCUS or token16 FOCUS.')
        cfg['prompt_encoder'] = deepcopy(cfg['encoder'])
        cfg['patch_encoder'] = {**deepcopy(cfg['encoder']), 'resolution': '20x'}
        cfg['conch_ckpt'] = cfg['backbone_weights']
    cfg.update(pathotme_extension=PORT, benchmark=study,
               experiment=f'{study}_{cohort}_focus_{encoder.replace("-", "_")}',
               feature_sources={'high': key}, feature_resolutions={'high': '20x'},
               feature_input_kinds={'high': 'patch_bag'},
               feature_path_column=f'feature__{key}', feature_path_column_l=f'feature__{key}',
               data_folder_l=source['path_template'].rsplit('/', 1)[0],
               data_folder_s=source['path_template'].rsplit('/', 1)[0],
               optimizer='adam', max_batch_failure_rate=0.0)
    cfg.pop('feature_path_column_s', None)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    encoder = cfg.get('backbone')
    if encoder not in ('plip', 'clip-rn50'):
        raise ValueError('FOCUS study requires PLIP or CLIP-RN50')
    plip = encoder == 'plip'
    key = 'plip_20x' if plip else 'clip_rn50_20x'
    space = 'hf:vinid/plip' if plip else 'openai/clip-rn50@official'
    expected = dict(pathotme_extension=PORT, method='focus', n_classes=2,
        shots=16, batch_size=1, feature_dim=768 if plip else 1024,
        model_feature_dim=512 if plip else 1024,
        feature_space_id=space + ('#vision-preprojection' if plip else ''),
        prompt_feature_space_id=space,
        feature_projection='native_visual_projection' if plip else 'none',
        encoder_extension_strategy=STRATEGY, feature_resolutions={'high': '20x'},
        feature_sources={'high': key}, feature_input_kinds={'high': 'patch_bag'},
        feature_path_column=f'feature__{key}', feature_path_column_l=f'feature__{key}',
        include_metadata=True, encoder_provenance='adapted', upstream_fidelity='partial',
        window_size=8, sim_threshold=0.8, max_context_length=8192,
        epochs=200, es_patience=20, es_stop_epoch=40, checkpoint_monitor='val_error')
    drift = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
    if drift:
        raise ValueError(f'FOCUS study contract drift: {drift}')
    labels = {'nsclc': {'LUAD': 0, 'LUSC': 1}, 'brca': {'IDC': 0, 'ILC': 1}}.get(cfg.get('task'))
    if labels is None or cfg.get('label_dict') != labels:
        raise ValueError('cohort/class binding mismatch')
    paired = cfg.get('encoder', {})
    if (paired.get('name') != encoder or paired.get('feature_space_id') != space
            or paired.get('feature_dim') != expected['model_feature_dim']
            or paired.get('weights') != cfg.get('backbone_weights')):
        raise ValueError('paired checkpoint/space mismatch')
    if not plip and (cfg.get('encoder_extension') is not None
                    or cfg.get('pathotme_encoder_extension') != CLIP_PORT):
        raise ValueError('RN50 port must have task-owned provenance')
    if plip and not isinstance(cfg.get('encoder_extension'), dict):
        raise ValueError('registered PLIP extension provenance required')
    if 'focus_token_prompt_policy' in cfg or cfg.get('max_patches') is not None:
        raise ValueError('token16 substitution or added patch truncation is not allowed')
    for k in ('lr', 'weight_decay'):
        if type(cfg.get(k)) is not float:
            raise TypeError(f'{k} must remain a float')


def validate_donors(phases, maps):
    """Require the parent's phase-local whole-row derangement by patient."""
    if set(phases) != set(maps):
        raise ValueError('donor phases differ from splits')
    for phase, values in phases.items():
        cases = {r['slide_id']: r['case_id'] for r in values}
        mapping = maps[phase]
        if set(mapping) != set(cases) or sorted(mapping.values()) != sorted(cases):
            raise ValueError('donors must be a bijection inside their phase')
        if any(cases[r] == cases[d] for r, d in mapping.items()):
            raise ValueError('donor cannot belong to the recipient patient')
