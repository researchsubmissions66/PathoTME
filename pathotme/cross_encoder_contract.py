"""Dependency-light scientific boundary for the explicit RN50 MGPATH port."""
CLIP_PORT='pathotme_mgpath_clip_rn50_feature_context_v1'
CLIP_TEXT_POLICY='native_clip_prefix_77_with_eot_v1'


def validate_clip_config(cfg):
    expected={'pathotme_encoder_extension':CLIP_PORT,'pathotme_text_policy':CLIP_TEXT_POLICY,
        'backbone':'clip-rn50','feature_space_id':'openai/clip-rn50@official','feature_dim':1024,
        'prompt_feature_space_id':'openai/clip-rn50@official','prompt_views':4,'image_centers':64,
        'type_gnn':'gat_conv','ratio_graph':0.2,'ot_epsilon':0.1,'ot_iterations':100,
        'feature_projection':'none','feature_resolutions':{'low':'5x','high':'10x'},
        'feature_path_column_l':'feature__clip_rn50_5x','feature_path_column_s':'feature__clip_rn50_10x',
        'mgpath_runtime':CLIP_PORT,'n_classes':2,'batch_size':1,
        'encoder_provenance':'adapted','upstream_fidelity':'partial'}
    drift={k:(cfg.get(k),v) for k,v in expected.items() if cfg.get(k)!=v}
    labels={'nsclc':{'LUAD':0,'LUSC':1},'brca':{'IDC':0,'ILC':1}}.get(cfg.get('task'))
    if labels is None or cfg.get('label_dict')!=labels:raise ValueError('cohort/class binding mismatch')
    if cfg.get('prompt_class_bindings')!=list(labels)*2:raise ValueError('prompt class/scale order mismatch')
    encoder=cfg.get('encoder',{})
    if encoder.get('name')!='clip-rn50' or encoder.get('feature_dim')!=1024 or encoder.get('feature_space_id')!='openai/clip-rn50@official':
        raise ValueError('paired encoder boundary mismatch')
    if cfg.get('encoder_extension') is not None:
        raise ValueError('RN50 MGPATH is task-owned; do not present it as a registered PGVL encoder extension')
    if drift:raise ValueError(f'RN50 port config drift: {drift}')

