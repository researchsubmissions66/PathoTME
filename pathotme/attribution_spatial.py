"""Thumbnail overlays of visual-region effects and their dependence on TME.

All six models use the same prediction-based spatial perturbation definition.
This is not an attention map or a cell segmentation. A region's joint score is
repeated on its patches; it must not be summed over those displayed patches.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path


def spatial_tme_scores(bridge, model, batch, row, *, reference_indices, max_groups=16):
    import torch
    from common.spatial_attribution import bind_spatial_inputs, perturb_spatial
    from common.interpretability import load_coordinate_bag
    bindings = bind_spatial_inputs(bridge, batch, row, row['slide_id'])
    if not reference_indices:
        raise ValueError('A nonempty TME reference intervention is required')
    model.eval()
    def predict(value):
        detail = bridge.eval_step_with_details(value, model)
        # Some adapters return probabilities only. Log probabilities recover
        # their distribution in the shared spatial scorer without class changes.
        return detail['probabilities'].clamp_min(1e-30).log()
    _, observed = perturb_spatial(bridge, model, batch, bindings,
                                 max_groups=max_groups, predictor=predict)
    calls = []
    def replace(_module, _inputs, output):
        calls.append(True)
        result = output.clone()
        result[:, list(reference_indices)] = 0
        return result
    handle = model.standardizer.register_forward_hook(replace)
    try:
        _, reference = perturb_spatial(bridge, model, batch, bindings,
                                      max_groups=max_groups, predictor=predict)
    finally:
        handle.remove()
    if len(calls) != reference['forward_passes']:
        raise ValueError('Spatial reference did not traverse exactly one standardizer per prediction')
    maps = []
    for binding, actual, neutral in zip(bindings, observed['inputs'], reference['inputs']):
        if actual['group_ids'] != neutral['group_ids'] or actual['reference_sha256'] != neutral['reference_sha256']:
            raise ValueError('Visual interventions changed between TME conditions')
        geometry = load_coordinate_bag(binding.source)
        groups = []
        for a, b in zip(actual['groups'], neutral['groups']):
            groups.append({'group_id': a['group_id'], 'patch_count': a['patch_count'],
                           'observed_scores_pp': a['scores_pp'], 'reference_scores_pp': b['scores_pp'],
                           'difference_scores_pp': [x-y for x,y in zip(a['scores_pp'], b['scores_pp'])]})
        maps.append({'scale': binding.scale, 'coordinates': binding.coordinates.tolist(),
                     'patch_size_level0': geometry.patch_size_level0,
                     'source_patch_indices': binding.source_indices.tolist(),
                     'group_ids': actual['group_ids'], 'groups': groups,
                     'feature_path': str(binding.source), 'coordinate_frame': 'level0'})
    return {'maps': maps, 'max_groups_per_scale': max_groups,
            'forward_calls': observed['forward_passes'] + reference['forward_passes'],
            'score_definition': 'observed_scores_pp = 100*(p(x,z)-p(x_region_mean,z)); reference_scores_pp uses z with selected TME inputs set to the training mean; difference = observed - reference.',
            'visual_reference': 'fixed mean of valid consumed rows in the original bag; supplied edges, masks and coordinates remain fixed',
            'interpretation': 'Grouped model sensitivity and TME/visual-region interaction; not attention, cell localization, additive decomposition or biological causality.'}


def thumbnail_source(*, wsi=None, thumbnail=None, dimensions=None, max_size=1200):
    """Load a real thumbnail in the full, unrotated level-0 slide frame."""
    from PIL import Image
    from pathotme.attribution_io import digest
    if bool(wsi) == bool(thumbnail):
        raise ValueError('Supply exactly one source WSI or existing full-slide thumbnail')
    if wsi:
        import openslide
        with openslide.OpenSlide(str(wsi)) as slide:
            width, height = slide.dimensions
            image = slide.get_thumbnail((max_size, max_size)).convert('RGB')
        provenance = {'source': 'openslide', 'wsi_basename': Path(wsi).name}
    else:
        if not dimensions or min(dimensions) <= 0:
            raise ValueError('Existing thumbnails require --level0-size WIDTH HEIGHT for registration')
        width, height = map(int, dimensions)
        image = Image.open(thumbnail).convert('RGB')
        # Pixel rounding can change the aspect ratio slightly, but a crop or
        # rotation must never be silently stretched into the slide frame.
        expected_height = image.width * height / width
        if abs(image.height - expected_height) > 2:
            raise ValueError('Thumbnail aspect ratio does not match the level-0 dimensions')
        image.thumbnail((max_size, max_size))
        provenance = {'source': 'provided_full_slide_thumbnail', 'thumbnail_sha256': digest(thumbnail),
                      'registration': 'user-declared same unrotated, uncropped level-0 frame'}
    stream = io.BytesIO(); image.save(stream, format='PNG')
    return image, {'level0_width': width, 'level0_height': height, 'width': image.width, 'height': image.height,
                   'image': 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode(),
                   'provenance': provenance}


def write_thumbnail_overlay(spatial, output, *, classes, reference_name, wsi=None, thumbnail=None, dimensions=None):
    """Export a real thumbnail, signed PNG overlay and standalone HTML viewer."""
    from PIL import Image, ImageDraw
    image, thumb = thumbnail_source(wsi=wsi, thumbnail=thumbnail, dimensions=dimensions)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for item in spatial['maps']:
        for x, y in item['coordinates']:
            if x < 0 or y < 0 or x >= thumb['level0_width'] or y >= thumb['level0_height']:
                raise ValueError('Feature coordinates fall outside the supplied slide frame')
    image.save(output / 'thumbnail.png')
    spatial.update(thumbnail={k: v for k, v in thumb.items() if k != 'image'},
                   classes=classes, reference_name=reference_name)
    (output / 'spatial_scores.json').write_text(json.dumps(spatial, indent=2, allow_nan=False) + '\n')
    # One portable overview, class 1 and the first input scale, accompanies the
    # interactive all-class/all-scale maps. No smoothing invents tissue scores.
    item = spatial['maps'][0]
    cls = min(1, len(classes)-1)
    values = [g['difference_scores_pp'][cls] for g in item['groups']]
    limit = max(map(abs, values), default=0) or 1
    layer = Image.new('RGBA', image.size); draw = ImageDraw.Draw(layer)
    sx, sy = image.width / thumb['level0_width'], image.height / thumb['level0_height']
    for (x,y), group in zip(item['coordinates'], item['group_ids']):
        value = values[group]
        color = (215, 55, 55) if value >= 0 else (40, 105, 210)
        strength = abs(value)/limit
        if strength == 0:
            continue
        left, top = round(x*sx), round(y*sy)
        right = max(left, round((x+item['patch_size_level0'])*sx)-1)
        bottom = max(top, round((y+item['patch_size_level0'])*sy)-1)
        draw.rectangle((left, top, right, bottom), fill=(*color, round(190*strength)))
    overlay = Image.alpha_composite(image.convert('RGBA'), layer).convert('RGB')
    result = Image.new('RGB', (max(overlay.width, 600), overlay.height+65), 'white')
    result.paste(overlay, (0,0)); draw = ImageDraw.Draw(result)
    draw.text((8, overlay.height+6), f"TME-dependent region score | {classes[cls]} | {item['scale']}", fill='black')
    draw.text((8, overlay.height+24), f"Red: positive; blue: negative | display +/- {limit:.4g} pp", fill='black')
    draw.text((8, overlay.height+42), 'Joint region scores; no scores assigned to unmeasured tissue.', fill='black')
    result.save(output / 'overlay.png')
    public = {'thumbnail': thumb, 'classes': classes, 'reference_name': reference_name,
              'maps': [{k:v for k,v in m.items() if k != 'feature_path'} for m in spatial['maps']]}
    public['thumbnail'] = {k:v for k,v in thumb.items() if k != 'provenance'}
    payload = json.dumps(public, allow_nan=False).replace('<','\\u003c').replace('&','\\u0026')
    template = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PathoTME thumbnail heatmap</title>
<style>body{font:16px system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#183044;background:#f7fafb}select,input,button{padding:.4rem;margin:.3rem}canvas{max-width:100%;height:auto;border:1px solid #ccd8dd}p{line-height:1.5}</style><h1>TME attribution over the slide thumbnail</h1>
<p>Each colored region has one joint prediction-change score. Red is positive and blue is negative for the selected class. Unmeasured tissue keeps the original thumbnail. This is model sensitivity, not attention or biological causality.</p>
<label>Class <select id="cls"></select></label><label>Scale <select id="scale"></select></label><label>Map <select id="mode"><option value="difference_scores_pp">Change attributable to the selected TME intervention</option><option value="observed_scores_pp">Visual-region effect with observed TME</option><option value="reference_scores_pp">Visual-region effect with reference TME</option></select></label><label>Opacity <input id="opacity" type="range" min="0" max="1" step=".05" value=".7"></label><button id="save">Download current PNG</button><p id="legend"></p><canvas id="map"></canvas><p id="hover"></p><p id="reference"></p>
<script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),get=id=>document.getElementById(id),canvas=get('map'),ctx=canvas.getContext('2d');canvas.width=d.thumbnail.width;canvas.height=d.thumbnail.height;const sx=canvas.width/d.thumbnail.level0_width,sy=canvas.height/d.thumbnail.level0_height;
function options(id,values){values.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v;get(id).appendChild(o)})}options('cls',d.classes);options('scale',d.maps.map(m=>m.scale));get('reference').textContent=`TME replacement: ${d.reference_name}. Reference = saved training-fold mean in transformed feature space. Colors use exact feature coordinates; no tissue interpolation.`;
const img=new Image();img.onload=draw;img.src=d.thumbnail.image;
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0,canvas.width,canvas.height);const m=d.maps[+get('scale').value],scores=m.groups.map(g=>g[get('mode').value][+get('cls').value]),limit=Math.max(1e-12,...scores.map(Math.abs));m.coordinates.forEach(([x,y],i)=>{const v=scores[m.group_ids[i]],strength=Math.abs(v)/limit;if(!strength)return;ctx.fillStyle=`rgba(${v>=0?'215,55,55':'40,105,210'},${+get('opacity').value*strength})`;ctx.fillRect(x*sx,y*sy,Math.max(1,m.patch_size_level0*sx),Math.max(1,m.patch_size_level0*sy))});get('legend').textContent=`Display range: -${limit.toPrecision(4)} to +${limit.toPrecision(4)} percentage points. ${m.groups.length} spatial groups. Scores repeat across their member patches and must not be summed.`;}
canvas.onmousemove=e=>{const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*canvas.width/r.width/sx,y=(e.clientY-r.top)*canvas.height/r.height/sy,m=d.maps[+get('scale').value],i=m.coordinates.findIndex(([a,b])=>x>=a&&x<a+m.patch_size_level0&&y>=b&&y<b+m.patch_size_level0);if(i<0){get('hover').textContent='No measured patch at this location';return}const g=m.groups[m.group_ids[i]];get('hover').textContent=`Group ${g.group_id} · ${g.patch_count} patches · score ${g[get('mode').value][+get('cls').value].toFixed(5)} pp`;};['cls','scale','mode','opacity'].forEach(id=>get(id).oninput=draw);get('save').onclick=()=>{const a=document.createElement('a');a.download='pathotme-thumbnail-heatmap.png';a.href=canvas.toDataURL('image/png');a.click()};</script>'''
    (output / 'heatmap.html').write_text(template.replace('__DATA__', payload))
    return {'viewer': str(output / 'heatmap.html'), 'overlay': str(output / 'overlay.png'),
            'reference_name': reference_name, 'forward_calls': spatial['forward_calls']}
