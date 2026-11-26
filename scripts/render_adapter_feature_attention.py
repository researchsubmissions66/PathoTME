#!/usr/bin/env python3
"""Map top slide-level TME contributions to MUSE patch-attention changes.

The input is a completed PathoTME-Adapter run and an original WSI (or a full,
unrotated thumbnail with its level-0 dimensions). No feature is assigned a
biological location: the map shows how the saved model reallocates attention
when that feature is replaced by the training-fold reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PGVL = Path(os.environ.get('PGVL_ROOT', ROOT.parent / 'PGVL-Gym')).resolve()
sys.path[:0] = [str(ROOT), str(PGVL)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path, help='TSV with completion,slide_id and wsi OR thumbnail,level0_width,level0_height')
    source.add_argument('--completion', type=Path, help='Saved fold_complete.json for one slide')
    parser.add_argument('--slide-id', help='Exact held-out slide ID with --completion')
    background = parser.add_mutually_exclusive_group()
    background.add_argument('--wsi', type=Path, help='Original full WSI for --completion')
    background.add_argument('--thumbnail', type=Path, help='Uncropped, unrotated full-slide thumbnail')
    parser.add_argument('--level0-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.top_k < 1 or args.threads < 1:
        parser.error('--top-k and --threads must be positive')
    if args.completion and (not args.slide_id or not (args.wsi or args.thumbnail)):
        parser.error('--completion needs --slide-id and --wsi or --thumbnail')
    if args.thumbnail and (not args.level0_size or min(args.level0_size) < 1):
        parser.error('--thumbnail needs positive --level0-size WIDTH HEIGHT')
    if args.manifest and (args.slide_id or args.wsi or args.thumbnail or args.level0_size):
        parser.error('Put slide and image sources in the manifest when using --manifest')
    return args


def cases_from_args(args):
    if args.completion:
        return [dict(completion=str(args.completion), slide_id=args.slide_id,
                     wsi=str(args.wsi) if args.wsi else '',
                     thumbnail=str(args.thumbnail) if args.thumbnail else '',
                     level0_width=args.level0_size[0] if args.level0_size else '',
                     level0_height=args.level0_size[1] if args.level0_size else '')]
    with args.manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if not rows or not {'completion', 'slide_id', 'wsi', 'thumbnail', 'level0_width', 'level0_height'} <= rows[0].keys():
        raise ValueError('Manifest must have completion, slide_id, wsi, thumbnail, level0_width, level0_height columns')
    return rows


def top_feature_attention(bridge, model, batch, names, groups, predicted, k):
    import numpy as np
    import torch
    from pathotme.attribution import ablate_tme

    def details():
        return bridge.eval_step_with_details(batch, model)

    def predict():
        return details()['probabilities']

    scores = ablate_tme(model, predict, names, groups, granularity='features')
    candidates = sorted((s for s in scores['scores'] if s['score_pp'][predicted] > 0),
                        key=lambda s: (-s['score_pp'][predicted], s['name']))[:k]
    if not candidates:
        raise ValueError('This slide has no positive measured-feature contribution for its predicted class')
    with torch.no_grad():
        observed = details()
    p_observed = observed['probabilities'].detach().cpu()[0]
    actual = observed['patch_attention'].detach().float().cpu().numpy()
    if actual.ndim != 2 or actual.shape[0] != len(p_observed):
        raise ValueError('MUSE did not return one patch-attention vector per class')
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-5)
    if (actual < 0).any():
        raise ValueError('Patch attention must be nonnegative')
    result = []
    for unit in candidates:
        calls = []
        def neutralize(_module, _inputs, output):
            calls.append(True)
            changed = output.clone()
            changed[:, unit['indices']] = 0
            return changed
        hook = model.standardizer.register_forward_hook(neutralize)
        try:
            with torch.no_grad():
                reference = details()
        finally:
            hook.remove()
        if len(calls) != 1:
            raise ValueError('Feature intervention must traverse the saved standardizer once')
        p_reference = reference['probabilities'].detach().cpu()[0]
        measured_pp = 100 * float(p_observed[predicted] - p_reference[predicted])
        if abs(measured_pp - unit['score_pp'][predicted]) > 1e-4:
            raise ValueError('Attention intervention does not match the independent feature score')
        ref_attention = reference['patch_attention'].detach().float().cpu().numpy()
        if ref_attention.shape != actual.shape:
            raise ValueError('Reference attention changed the consumed patch count')
        np.testing.assert_allclose(ref_attention.sum(axis=1), 1, atol=1e-5)
        delta = actual[predicted] - ref_attention[predicted]
        result.append(dict(name=unit['name'], feature_index=unit['indices'][0], score_pp=measured_pp,
                           reference_probability=float(p_reference[predicted]),
                           attention_total_variation=float(0.5 * np.abs(delta).sum()),
                           delta_attention=delta))
    with torch.no_grad():
        torch.testing.assert_close(details()['probabilities'], observed['probabilities'], atol=1e-7, rtol=0)
    return scores, result, actual.shape[1]


def font(size, bold=False):
    from PIL import ImageFont
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def wrapped_feature(name, width=44):
    display = name.replace('_', ' ').lower()
    return textwrap.wrap(display, width=width, break_long_words=False, break_on_hyphens=False)


def draw_panel(image, coordinates, patch_size, dimensions, feature, cap, panel_width):
    import numpy as np
    from PIL import Image, ImageDraw
    scale = panel_width / image.width
    panel_height = round(image.height * scale)
    base = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS).convert('RGBA')
    layer = Image.new('RGBA', base.size)
    draw = ImageDraw.Draw(layer)
    sx, sy = panel_width / dimensions[0], panel_height / dimensions[1]
    delta = feature['delta_attention']
    if len(delta) != len(coordinates):
        raise ValueError('Attention and exact consumed patch coordinates are not aligned')
    for (x, y), value in zip(coordinates, delta):
        if value == 0:
            continue
        strength = min(1.0, abs(float(value)) / cap)
        alpha = round(205 * strength**0.72)
        if alpha < 2:
            continue
        color = (214, 45, 39, alpha) if value > 0 else (36, 95, 215, alpha)
        left, top = round(float(x) * sx), round(float(y) * sy)
        right = max(left, round((float(x) + patch_size) * sx) - 1)
        bottom = max(top, round((float(y) + patch_size) * sy) - 1)
        draw.rectangle((left, top, right, bottom), fill=color)
    return Image.alpha_composite(base, layer).convert('RGB')


def render_cohort(cohort, slides, output):
    import numpy as np
    from PIL import Image, ImageDraw
    panel_width, gap, margin = 620, 24, 35
    ncols = max(len(s['features']) for s in slides)
    row_header, panel_header, row_footer = 54, 106, 59
    panel_heights = [round(s['image'].height * panel_width / s['image'].width) for s in slides]
    width = 2 * margin + ncols * panel_width + (ncols-1) * gap
    height = 2*margin + 80 + sum(row_header + panel_header + h + row_footer for h in panel_heights) + gap*(len(slides)-1)
    canvas = Image.new('RGB', (width, height), (250, 250, 247))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 22), f'{cohort.upper()}  |  held-out MUSE / CLIP-RN50  |  top TME feature attention shifts',
              fill=(24, 34, 46), font=font(24, True))
    draw.text((margin, 57), 'Red: more attention with measured feature   Blue: more attention at training-mean reference',
              fill=(50, 59, 72), font=font(17))
    y = margin + 80
    for slide, panel_height in zip(slides, panel_heights):
        label = slide['classes'][slide['label']]
        predicted = slide['classes'][slide['predicted']]
        status = 'correct' if slide['label'] == slide['predicted'] else 'misclassified'
        draw.text((margin, y), f"{slide['slide_id']}   |   true: {label}   |   predicted: {predicted} "
                              f"(p={slide['probability']:.3f}; {status})",
                  fill=(24, 34, 46), font=font(19, True))
        y += row_header
        values = np.concatenate([np.abs(f['delta_attention']) for f in slide['features']])
        nonzero = values[values > 0]
        cap = float(np.quantile(nonzero, .99)) if len(nonzero) else 1.0
        cap = max(cap, 1e-12)
        for col, feature in enumerate(slide['features']):
            x = margin + col * (panel_width + gap)
            draw.text((x, y), f"#{col+1}  +{feature['score_pp']:.3f} pp predicted-class probability",
                      fill=(24, 34, 46), font=font(18, True))
            for line_index, line in enumerate(wrapped_feature(feature['name'])[:3]):
                draw.text((x, y + 27 + line_index*21), line, fill=(42, 52, 62), font=font(16))
            panel = draw_panel(slide['image'], slide['coordinates'], slide['patch_size'], slide['dimensions'],
                               feature, cap, panel_width)
            canvas.paste(panel, (x, y + panel_header))
            draw.rectangle((x, y + panel_header, x + panel_width-1, y + panel_header + panel_height-1),
                           outline=(120, 127, 132), width=1)
            draw.text((x, y + panel_header + panel_height + 8),
                      f"Attention mass redistributed: {100*feature['attention_total_variation']:.2f}%",
                      fill=(42, 52, 62), font=font(17))
        y += panel_header + panel_height
        draw.text((margin, y + 32),
                  f'Color saturation: +/- {100*cap:.4g} pp attention mass per patch (shared across this slide).',
                  fill=(70, 77, 84), font=font(15))
        y += row_footer + gap
    canvas.save(output, optimize=True)


def main():
    args = parse_args()
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from common.configuration import load_dotenv
    load_dotenv(PGVL / '.env')
    import numpy as np
    import torch
    from common.spatial_attribution import bind_spatial_inputs
    from common.interpretability import load_coordinate_bag
    from pathotme.attribution import panel_layout
    from pathotme.attribution_io import load_bound_run, digest
    from pathotme.attribution_runtime import check_selected_features, load_model, one_batch
    from pathotme.attribution_spatial import thumbnail_source

    torch.set_num_threads(args.threads)
    cases = cases_from_args(args)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    loaded = {}
    grouped = {}
    public_rows = []
    for index, case in enumerate(cases, 1):
        completion = Path(case['completion']).expanduser().resolve()
        if completion not in loaded:
            print(f'Loading saved checkpoint {completion.parent}', flush=True)
            bound = load_bound_run(completion, 'actual', weights=True)
            cfg = bound['config']
            if cfg['method'] != 'muse' or cfg['backbone'] != 'clip-rn50':
                raise ValueError('This renderer requires a MUSE / CLIP-RN50 adapter checkpoint')
            bridge, model, loader = load_model(bound, 'actual', args.device)
            loaded[completion] = (bound, bridge, model, loader)
        bound, bridge, model, loader = loaded[completion]
        cfg = bound['config']
        slide_id = case['slide_id']
        row = next((r for r in bound['test_rows'] if r['slide_id'] == slide_id), None)
        if row is None:
            raise ValueError(f'{slide_id} is not a held-out slide in {completion}')
        check_selected_features(bound, row)
        batch = one_batch(loader, slide_id)
        binding = bind_spatial_inputs(bridge, batch, row, slide_id)
        if len(binding) != 1 or binding[0].scale != 'patch':
            raise ValueError('Expected one exact MUSE patch-coordinate bag')
        coordinates = binding[0].coordinates.cpu().numpy()
        geometry = load_coordinate_bag(binding[0].source)
        is_wsi = bool(case.get('wsi'))
        if is_wsi and slide_id not in Path(case['wsi']).name:
            raise ValueError('The original WSI filename must contain the slide ID')
        dimensions = None if is_wsi else (int(case['level0_width']), int(case['level0_height']))
        image, thumb = thumbnail_source(wsi=case['wsi'] if is_wsi else None,
                                       thumbnail=case['thumbnail'] if not is_wsi else None,
                                       dimensions=dimensions)
        dimensions = (thumb['level0_width'], thumb['level0_height'])
        if np.any(coordinates < 0) or np.any(coordinates[:, 0] >= dimensions[0]) or np.any(coordinates[:, 1] >= dimensions[1]):
            raise ValueError('Consumed patch coordinates lie outside the WSI')
        labels = sorted(cfg['label_dict'].items(), key=lambda item: item[1])
        classes = [name for name, _ in labels]
        names, groups = panel_layout(cfg['task'])
        with torch.no_grad():
            observed = bridge.eval_step_with_details(batch, model)
        if observed['metadata']['slide_id'] != slide_id:
            raise ValueError('Model batch and selected slide differ')
        probability = observed['probabilities'].detach().cpu()[0]
        saved = torch.tensor([float(bound['predictions'][slide_id][f'probability_{i}']) for i in range(len(classes))])
        torch.testing.assert_close(probability, saved, atol=1e-4, rtol=1e-4)
        predicted = int(probability.argmax())
        print(f'[{index}/{len(cases)}] {cfg["task"]} {slide_id}: ranking {len(names)} measured features', flush=True)
        scores, features, patch_count = top_feature_attention(bridge, model, batch, names, groups,
                                                                predicted, args.top_k)
        if patch_count != len(coordinates):
            raise ValueError('MUSE attention count differs from exact consumed coordinate count')
        label = int(cfg['label_dict'].get(row['label'], row['label']))
        slide = dict(cohort=cfg['task'], slide_id=slide_id, label=label, classes=classes,
                     predicted=predicted, probability=float(probability[predicted]),
                     image=image, coordinates=coordinates, dimensions=dimensions,
                     patch_size=geometry.patch_size_level0, features=features)
        grouped.setdefault(cfg['task'], []).append(slide)
        public_rows.append(dict(cohort=cfg['task'], slide_id=slide_id, fold=cfg['_fold_index'],
                                shots=cfg.get('shots', bound['completion']['plan'].get('shots')),
                                classes=classes, true_class=classes[label], predicted_class=classes[predicted],
                                predicted_probability=float(probability[predicted]),
                                patch_count=patch_count, top_features=[{k: v for k, v in f.items() if k != 'delta_attention'}
                                                                 for f in features],
                                checkpoint_sha256=next(v for p, v in bound['provenance']['checkpoint_sha256'].items()
                                                       if p.endswith('/actual/best.pt')),
                                thumbnail_sha256=digest(case['thumbnail']) if not is_wsi else None,
                                all_tme_score_pp=scores['all_tme_score_pp'][predicted]))
        print('  ' + '; '.join(f"{f['name']} +{f['score_pp']:.3f} pp; attention TV {100*f['attention_total_variation']:.2f}%"
                            for f in features), flush=True)
    for cohort, slides in grouped.items():
        target = output / f'{cohort}_top_feature_attention.png'
        render_cohort(cohort, slides, target)
        print(f'Wrote {target}', flush=True)
    public = dict(method='PathoTME-Adapter / MUSE', backbone='CLIP-RN50', arm='actual',
                  feature_score='100*(p_pred(measured TME)-p_pred(feature set to saved training-fold transformed mean))',
                  heatmap='class-query patch attention with measured TME minus attention at one-feature training-fold reference',
                  attention_mass_redistributed='0.5*sum(abs(attention_measured-attention_reference))',
                  interpretation='Model attention redistribution, not the physical location of the measured TME feature or a causal effect.',
                  slides=public_rows)
    (output / 'summary.json').write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
