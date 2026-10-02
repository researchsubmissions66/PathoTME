#!/usr/bin/env python3
"""Explain a completed PathoTME checkpoint; plan by default, never train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PGVL = Path(os.environ.get('PGVL_ROOT', ROOT.parent / 'PGVL-Gym')).resolve()
sys.path[:0] = [str(ROOT), str(PGVL)]
from pathotme.attribution import panel_layout
from pathotme.attribution_io import METHODS, SCORE_VERSION, digest, export_result, load_bound_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-providers', action='store_true')
    parser.add_argument('--completion', type=Path, help='Completed fold_complete.json')
    parser.add_argument('--launch', type=Path, help='Optional explicit bound launch.json')
    parser.add_argument('--arm', choices=('actual', 'zero', 'shuffled'), default='actual')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--slide-id', action='append', help='Exact held-out slide; repeat for selected slides')
    selection.add_argument('--all-test-slides', action='store_true', help='Explicitly explain every test slide in this fold')
    parser.add_argument('--granularity', choices=('groups', 'features', 'both'), default='both')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, help='New private artifact directory outside source repositories')
    parser.add_argument('--execute', action='store_true')
    background = parser.add_mutually_exclusive_group()
    background.add_argument('--wsi', type=Path, help='Original slide, used to generate a correctly registered thumbnail')
    background.add_argument('--thumbnail', type=Path, help='Existing full-slide, unrotated, uncropped thumbnail')
    parser.add_argument('--level0-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--spatial-reference', default='all', help='all, an exact biological group name, or an exact feature name')
    parser.add_argument('--spatial-groups', type=int, default=16, help='Maximum visual regions per scale for thumbnail attribution')
    args = parser.parse_args(argv)
    if args.list_providers:
        print(json.dumps({'methods': METHODS, 'cohorts': ['nsclc', 'brca'], 'encoders': ['plip', 'clip-rn50'],
                          'score_version': SCORE_VERSION, 'feature_counts': {'nsclc': 62, 'brca': 64},
                          'biological_groups': 16, 'spatial': 'region probability delta and its change under TME replacement'}))
        return
    if not args.completion or not args.output:
        parser.error('--completion and --output are required')
    if args.threads < 1 or not 1 <= args.spatial_groups <= 256:
        parser.error('Positive --threads and --spatial-groups between 1 and 256 are required')
    output = args.output.expanduser().resolve()
    if any(output == root or root in output.parents for root in (ROOT, PGVL)):
        parser.error('Use private results storage outside source repositories')
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    if args.thumbnail and not args.level0_size:
        parser.error('--thumbnail requires --level0-size WIDTH HEIGHT')
    bound = load_bound_run(args.completion, args.arm, args.launch)
    cfg = bound['config']; plan = bound['completion']['plan']
    names, groups = panel_layout(plan['cohort'])
    slides = [r['slide_id'] for r in bound['test_rows']] if args.all_test_slides else (args.slide_id or [bound['test_rows'][0]['slide_id']])
    if len(set(slides)) != len(slides) or any(s not in bound['predictions'] for s in slides):
        parser.error('Slide selection contains duplicates or slides outside the held-out fold')
    spatial = bool(args.thumbnail or args.wsi)
    if spatial and len(slides) != 1:
        parser.error('A thumbnail/WSI binds exactly one slide; export separate slides and aggregate their JSON files')
    if args.spatial_reference == 'all':
        indices = list(range(len(names)))
    elif args.spatial_reference in names:
        indices = [names.index(args.spatial_reference)]
    else:
        found = [g['indices'] for g in groups if g['name'] == args.spatial_reference]
        if not found:
            parser.error('Unknown spatial reference; use all, a native group name or an exact feature name')
        indices = found[0]
    count = (len(groups) if args.granularity != 'features' else 0) + (len(names) if args.granularity != 'groups' else 0)
    summary = {'mode': 'execute' if args.execute else 'plan', 'method': cfg['method'], 'cohort': cfg['task'],
               'encoder': cfg['backbone'], 'fold': cfg['_fold_index'], 'shots': cfg.get('shots', plan.get('shots', 16)),
               'arm': args.arm, 'slides': slides, 'feature_names': names, 'groups': groups,
               'TME_forward_calls': len(slides)*(count+3), 'thumbnail_overlay': spatial,
               'spatial_reference': args.spatial_reference if spatial else None,
               'spatial_max_extra_forward_calls': 2*(3+args.spatial_groups*(2 if cfg['method'] in ('vila_mil','mgpath','hive_mil','mscpt') else 1)) if spatial else 0,
               'output': str(output), 'device': args.device, 'training': False}
    print(json.dumps(summary, indent=2), flush=True)
    if not args.execute:
        return
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from common.configuration import load_dotenv
    load_dotenv(PGVL / '.env')
    import torch
    from pathotme.attribution import ablate_tme
    from pathotme.attribution_runtime import load_model, one_batch, check_selected_features
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    print('Verifying native and adapter checkpoint hashes', flush=True)
    bound = load_bound_run(args.completion, args.arm, args.launch, weights=True)
    print('Reconstructing the saved model; no standardizer refit', flush=True)
    bridge, model, loader = load_model(bound, args.arm, args.device)
    labels = sorted(cfg['label_dict'].items(), key=lambda x: x[1])
    if [v for _, v in labels] != list(range(len(labels))):
        raise ValueError('Class indices must be contiguous and match checkpoint order')
    result = {k: summary[k] for k in ('method', 'cohort', 'encoder', 'fold', 'shots', 'arm')}
    result.update(score_version=SCORE_VERSION, feature_names=names, groups=groups, classes=[n for n, _ in labels],
                  granularity=args.granularity, provenance=bound['provenance'],
                  execution={'device': args.device, 'threads': args.threads, 'num_workers': 0},
                  baseline='zero after saved training-fold standardization; transformed training mean, not biological absence',
                  interpretation='signed class-probability sensitivity in percentage points; nonadditive and noncausal',
                  source_sha256={str(path.relative_to(ROOT)): digest(path) for path in
                                 [Path(__file__), *sorted((ROOT / 'pathotme').glob('attribution*.py'))]},
                  expected_test_slides=[{'slide_id': r['slide_id'], 'case_id': r['case_id'],
                                        'label': int(cfg['label_dict'].get(r['label'], r['label']))} for r in bound['test_rows']], slides=[])
    pending_spatial = None
    for position, slide in enumerate(slides):
        row = next(r for r in bound['test_rows'] if r['slide_id'] == slide)
        features = check_selected_features(bound, row)
        batch = one_batch(loader, slide)
        def predict():
            detail = bridge.eval_step_with_details(batch, model)
            if detail['metadata']['slide_id'] != slide:
                raise ValueError('Loaded batch/slide identity mismatch')
            return detail['probabilities']
        with torch.no_grad():
            observed = predict()
            expected = observed.new_tensor([[float(bound['predictions'][slide][f'probability_{i}']) for i in range(len(labels))]])
            torch.testing.assert_close(observed, expected, atol=1e-4, rtol=1e-4)
        print(f'Explaining slide {position+1}/{len(slides)}', flush=True)
        scores = ablate_tme(model, predict, names, groups, granularity=args.granularity,
                           progress=lambda done, total: print(f'TME interventions {done}/{total}', flush=True) if done % 8 == 0 or done == total else None)
        record = dict(slide_id=slide, case_id=row['case_id'], label=int(cfg['label_dict'].get(row['label'], row['label'])),
                      saved_probability_match=True, feature_files=features, **scores)
        # Standardized measurements and their missingness flags remain private
        # JSON provenance; neither raw TME tables nor these values enter HTML.
        result['slides'].append(record)
        if spatial:
            from pathotme.attribution_spatial import spatial_tme_scores
            print('Computing spatial region effects with observed and reference TME', flush=True)
            pending_spatial = spatial_tme_scores(bridge, model, batch, row,
                                                reference_indices=indices, max_groups=args.spatial_groups)
            with torch.no_grad():
                torch.testing.assert_close(predict(), observed, atol=1e-7, rtol=0)
    result['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = time.monotonic()-started
    export_result(result, output)
    if pending_spatial is not None:
        from pathotme.attribution_spatial import write_thumbnail_overlay
        overlay = write_thumbnail_overlay(pending_spatial, output / 'spatial', classes=result['classes'],
                                           reference_name=args.spatial_reference, wsi=args.wsi,
                                           thumbnail=args.thumbnail, dimensions=args.level0_size)
        result['spatial'] = overlay
        (output / 'attribution.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        with (output / 'heatmap.html').open('a') as handle:
            handle.write('<p><a href="spatial/heatmap.html">Open the thumbnail attribution overlay</a></p>')
    print(json.dumps({'status': 'completed', 'output': str(output), 'slides': len(slides),
                      'elapsed_seconds': time.monotonic()-started, 'thumbnail_overlay': bool(pending_spatial)}, indent=2))


if __name__ == '__main__':
    main()
