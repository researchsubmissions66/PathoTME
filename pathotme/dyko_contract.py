"""Frozen paired DyKo addition with a separately authored BRCA knowledge bank."""
from copy import deepcopy
from methods.dyko.prompts import UPSTREAM_COMMIT

PORT = 'pathotme_dyko_query_tme_v1'
CLIP_PORT = 'pathotme_dyko_clip_rn50_feature_context_concept_bridge_v1'
STRATEGY = 'paired_feature_context_concept_bridge_v1'


def build_config(base, paired, source, study, cohort, encoder, brca_assets=None):
    cfg = deepcopy(base)
    if cohort == 'brca':
        if not brca_assets:
            raise ValueError('BRCA requires a separately encoded and frozen knowledge bank')
        cfg.update(task='brca', label_dict={'IDC': 0, 'ILC': 1},
            classnames=['invasive ductal carcinoma', 'invasive lobular carcinoma'],
            text_prompt_path=brca_assets['class_prompt_path'],
            text_prompt_file_sha256=brca_assets['class_prompt_sha256'],
            prompt_file_classnames=['Invasive Ductal Carcinoma', 'Invasive Lobular Carcinoma'],
            prompt_class_bindings=['IDC', 'ILC'], prompt_provenance='generated',
            prompt_source='pathotme_reformatted_existing_focus_brca_high_scale',
            prompt_generator='existing_focus_bank_author_identity_not_reattested',
            concept_feature_path=brca_assets['tensor_path'],
            concept_feature_sha256=brca_assets['tensor_sha256'],
            concept_source='pathotme_brca_morphology64_pinned_titan_v1',
            pathotme_knowledge_bank=deepcopy(brca_assets), concept_count=64)
    elif cohort == 'nsclc':
        cfg['concept_count'] = 1000
    else:
        raise ValueError('unsupported cohort')
    if encoder == 'clip-rn50':
        for key in ('backbone', 'backbone_weights', 'encoder'):
            cfg[key] = deepcopy(paired[key])
        cfg.update(feature_dim=1024, model_feature_dim=1024,
            feature_space_id=source['feature_space_id'],
            prompt_feature_space_id=source['feature_space_id'],
            feature_projection='none', encoder_extension=None,
            pathotme_encoder_extension=CLIP_PORT,
            implementation_provenance='pathotme_extended_paired_feature_context_concept_bridge',
            fidelity_note='PathoTME-owned RN50 paired DyKo extension. Frozen paired '
            'text and one shared learned feature context replace TITAN token prompting. '
            'The cohort-specific frozen 768D TITAN concepts use a learned 768-to-1024 '
            'bridge. Native WAKI, dual attention and CE+KL training are retained. '
            'Not upstream TITAN DyKo or a registered PGVL RN50 condition.')
        cfg['prompt_encoder'] = deepcopy(cfg['encoder'])
        cfg['patch_encoder'] = {**deepcopy(cfg['encoder']), 'resolution': '20x',
            'patch_geometry': source.get('patch_geometry', '20x_224px_0px_overlap')}
    key = 'plip_20x' if encoder == 'plip' else 'clip_rn50_20x'
    cfg.update(pathotme_extension=PORT, benchmark=study,
        experiment=f'{study}_{cohort}_dyko_{encoder.replace("-", "_")}',
        feature_sources={'bag': key}, feature_resolutions={'bag': '20x'},
        feature_input_kinds={'bag': 'patch_bag'}, feature_path_column=f'feature__{key}',
        data_folder_s=source['path_template'].rsplit('/', 1)[0], max_batch_failure_rate=0.0)
    if cohort == 'brca':
        cfg['fidelity_note'] = cfg.get('fidelity_note', '') + ' BRCA is a PathoTME task extension with 64 authored morphology concepts encoded by pinned TITAN and existing FOCUS-derived class text; not a released 1000-concept bank.'
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    encoder = cfg.get('backbone')
    if encoder not in ('plip', 'clip-rn50'):
        raise ValueError('DyKo study requires PLIP or CLIP-RN50')
    plip = encoder == 'plip'
    space = 'hf:vinid/plip' if plip else 'openai/clip-rn50@official'
    key = 'plip_20x' if plip else 'clip_rn50_20x'
    cohort = cfg.get('task')
    if cohort not in ('nsclc', 'brca'):
        raise ValueError('NSCLC or BRCA required')
    expected = dict(pathotme_extension=PORT, method='dyko',
        label_dict={'LUAD': 0, 'LUSC': 1} if cohort == 'nsclc' else {'IDC': 0, 'ILC': 1}, n_classes=2, shots=16, k=5, seed=1,
        batch_size=1, include_metadata=True, feature_dim=768 if plip else 1024,
        model_feature_dim=512 if plip else 1024,
        feature_space_id=space+('#vision-preprojection' if plip else ''),
        prompt_feature_space_id=space, feature_projection='native_visual_projection' if plip else 'none',
        encoder_extension_strategy=STRATEGY, feature_sources={'bag': key},
        feature_resolutions={'bag': '20x'}, feature_input_kinds={'bag': 'patch_bag'},
        feature_path_column=f'feature__{key}', upstream_commit=UPSTREAM_COMMIT,
        n_ctx=16, visual_prototypes=10, concepts_per_prototype=10, num_heads=8,
        retrieval_temperature=0.1, structural_consistency_weight=1.0,
        kmeans_iterations=20, kmeans_seed=42, gradient_accumulation_steps=4,
        optimizer='adam', epochs=1000, early_stopping=True, es_patience=40,
        es_stop_epoch=0, checkpoint_monitor='val_loss', lr_scheduler=None,
        text_prompt_file_sha256='14bc274f5b25df020403feca0e03777408ea56dfbdfe5c7bf38b796bc77f953d',
        concept_feature_sha256='023463fb4fed7f86919b7e6884681711a66d726d525799942e46cfc8030ec477',
        prompt_file_classnames=['Lung Squamous Cell Carcinoma', 'Lung Adenocarcinoma'],
        prompt_class_bindings=['LUSC', 'LUAD'])
    if cohort == 'brca':
        assets = cfg.get('pathotme_knowledge_bank', {})
        for key in ('class_prompt_path', 'class_prompt_sha256', 'tensor_path', 'tensor_sha256',
                    'source_bank_path', 'source_bank_sha256', 'encoding_path', 'encoding_sha256'):
            if not assets.get(key):
                raise ValueError(f'missing BRCA knowledge provenance: {key}')
        expected.update(text_prompt_file_sha256=assets['class_prompt_sha256'],
            text_prompt_path=assets['class_prompt_path'],
            concept_feature_path=assets['tensor_path'], concept_feature_sha256=assets['tensor_sha256'],
            concept_count=64, prompt_file_classnames=['Invasive Ductal Carcinoma', 'Invasive Lobular Carcinoma'],
            prompt_class_bindings=['IDC', 'ILC'], concept_source='pathotme_brca_morphology64_pinned_titan_v1')
    else:
        expected['concept_count'] = 1000
    drift = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
    if drift:
        raise ValueError(f'DyKo study contract drift: {drift}')
    paired = cfg.get('encoder', {})
    if (paired.get('name') != encoder or paired.get('feature_space_id') != space
            or paired.get('feature_dim') != expected['model_feature_dim']
            or paired.get('weights') != cfg.get('backbone_weights')):
        raise ValueError('paired checkpoint/space mismatch')
    if plip and not isinstance(cfg.get('encoder_extension'), dict):
        raise ValueError('registered PLIP extension provenance required')
    if not plip and (cfg.get('encoder_extension') is not None
            or cfg.get('pathotme_encoder_extension') != CLIP_PORT):
        raise ValueError('RN50 requires PathoTME-owned provenance')
    if cfg.get('max_patches') is not None:
        raise ValueError('no additional patch truncation allowed')
    for key in ('lr', 'weight_decay'):
        if type(cfg.get(key)) is not float:
            raise TypeError(f'{key} must remain numeric')
