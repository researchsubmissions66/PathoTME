"""Dependency-light artifact checks, patient aggregation and HTML export."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

SCORE_VERSION = 'tme_train_mean_ablation_probability_delta_v1'
METHODS = ('vila_mil', 'mgpath', 'focus', 'muse', 'hive_mil', 'dyko', 'mscpt')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def check_hash(path, expected):
    observed = digest(path)
    if observed != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return observed


def load_bound_run(completion_path, arm='actual', launch_path=None, *, weights=False):
    """Check the completed fold/config/predictions and optionally both checkpoints."""
    completion_path = Path(completion_path).resolve()
    completion = json.loads(completion_path.read_text())
    if completion.get('status') != 'completed' or arm not in completion.get('arms', {}):
        raise ValueError('A completed fold and saved adapter arm are required')
    plan = completion['plan']
    if plan['method'] not in METHODS:
        raise ValueError('Unsupported PathoTME architecture')
    if Path(plan['output']).resolve() != completion_path.parent:
        raise ValueError('Completion output binding mismatch')
    candidates = [Path(launch_path)] if launch_path else [p / n for p in completion_path.parents
                  for n in ('launch.sealed.json', 'launch.json') if (p / n).is_file()]
    matching = []
    for candidate in candidates:
        launch = json.loads(candidate.read_text())
        if launch.get('identity') == completion['launch_identity']:
            matching.append((candidate, launch))
    if not matching:
        raise ValueError('Matching launch was not found; pass --launch explicitly')
    launch_path, launch = matching[0]
    payload = {k: v for k, v in launch.items() if k != 'identity'}
    if hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != launch['identity']:
        raise ValueError('Launch identity mismatch')
    if plan not in launch['plans']:
        raise ValueError('Completion plan is not present in the bound launch')
    directory = completion_path.parent / arm
    config_path = directory / 'config.json'
    prediction_path = directory / 'predictions.csv'
    record = completion['arms'][arm]
    for path in (config_path, prediction_path):
        check_hash(path, record['artifact_sha256'][str(path)])
    saved_config = json.loads(config_path.read_text())
    if (record['status'] != 'completed' or saved_config['identity'] != record['identity']
            or saved_config['launch_identity'] != launch['identity'] or saved_config['arm'] != arm
            or saved_config['native_sha256'] != completion['native_sha256']):
        raise ValueError('Adapter configuration/identity binding mismatch')
    cfg = saved_config['config']
    if (cfg['task'] != plan['cohort'] or cfg['method'] != plan['method']
            or cfg['_fold_index'] != plan['fold']
            or Path(cfg['tme_feature_csv']).resolve() != Path(plan['tme_csv']).resolve()
            or cfg['tme_mode'] != ('zero' if arm == 'zero' else 'actual')
            or Path(cfg['base_checkpoint_dir']).resolve() != Path(plan['native_dir']).resolve()):
        raise ValueError('Adapter condition mismatch')
    split_path = Path(cfg['split_dir']) / f"fold{plan['fold']}/test.csv"
    for path in (split_path, Path(plan['donor_maps']), Path(plan['tme_csv']), Path(plan['feature_inventory'])):
        expected = launch['file_sha256'].get(str(path))
        if expected is None:
            raise ValueError(f'Input is missing from the launch hash contract: {path}')
        check_hash(path, expected)
    with prediction_path.open() as handle:
        predictions = list(csv.DictReader(handle))
    with split_path.open() as handle:
        test_rows = list(csv.DictReader(handle))
    def membership(rows):
        return sorted((r['slide_id'], r['case_id'], int(cfg['label_dict'].get(r['label'], r['label']))) for r in rows)
    if (membership(test_rows) != membership(predictions)
            or len({r['slide_id'] for r in test_rows}) != len(test_rows)):
        raise ValueError('Saved prediction/test membership mismatch')
    checkpoints = {str(directory / 'best.pt'): record['artifact_sha256'][str(directory / 'best.pt')],
                   str(Path(plan['native_dir']) / f"fold{plan['fold']}_best.pt"):
                       completion['native_sha256'][str(Path(plan['native_dir']) / f"fold{plan['fold']}_best.pt")]}
    if weights:
        for path, expected in checkpoints.items():
            check_hash(path, expected)
    return dict(completion=completion, config=cfg, record=record, test_rows=test_rows,
                predictions={r['slide_id']: r for r in predictions}, directory=directory,
                provenance={'completion_sha256': digest(completion_path), 'launch_identity': launch['identity'],
                            'config_sha256': digest(config_path), 'checkpoint_sha256': checkpoints,
                            'checkpoint_hashes_verified': weights, 'launch_sha256': digest(launch_path),
                            'test_csv_sha256': digest(split_path), 'tme_csv_sha256': digest(plan['tme_csv']),
                            'feature_inventory_sha256': digest(plan['feature_inventory']),
                            'donor_maps_sha256': digest(plan['donor_maps'])})


def aggregate_results(documents):
    """Rank absolute patient probability deltas, then average fold means equally.

    Only patients with every expected held-out slide are included. Cohort
    coverage is explicit; a few selected slides never become a full-cohort result.
    Separate classes and feature/group scores are never silently combined.
    """
    if not documents:
        raise ValueError('At least one attribution result is required')
    keys = ('score_version', 'cohort', 'method', 'encoder', 'shots', 'arm', 'feature_names', 'groups', 'classes', 'granularity')
    first = documents[0]
    if any(any(doc[k] != first[k] for k in keys) for doc in documents):
        raise ValueError('Aggregate one matched cohort/method/encoder/shot/arm/panel/score definition at a time')
    slides, expected, identities = {}, {}, {}
    for doc in documents:
        fold = doc['fold']
        identity = doc['provenance']['checkpoint_sha256']
        if identities.setdefault(fold, identity) != identity:
            raise ValueError('Different checkpoints for the same fold')
        expected_fold = {r['slide_id']: (r['case_id'], int(r['label'])) for r in doc['expected_test_slides']}
        if expected.setdefault(fold, expected_fold) != expected_fold:
            raise ValueError('Inconsistent test membership')
        for slide in doc['slides']:
            key = (fold, slide['slide_id'])
            if key in slides:
                raise ValueError('Duplicate slide attribution')
            if expected_fold.get(slide['slide_id']) != (slide['case_id'], slide['label']):
                raise ValueError('Attribution slide is outside its expected test partition')
            slides[key] = slide
    patients, expected_patients = defaultdict(list), defaultdict(set)
    patient_folds = {}
    for fold, rows in expected.items():
        for slide, (case, label) in rows.items():
            if patient_folds.setdefault(case, fold) != fold:
                raise ValueError('A patient appears in multiple held-out folds')
            expected_patients[(fold, case)].add(slide)
    for (fold, slide_id), slide in slides.items():
        patients[(fold, slide['case_id'])].append(slide)
    rows = []
    omitted = []
    for (fold, case), members in patients.items():
        if {s['slide_id'] for s in members} != expected_patients[(fold, case)]:
            omitted.append({'fold': fold, 'case_id': case, 'reason': 'not all patient slides explained'})
            continue
        if len({s['label'] for s in members}) != 1:
            raise ValueError('Conflicting patient labels')
        layouts = [[(u['level'], u['name'], u['indices']) for u in s['scores']] for s in members]
        if any(layout != layouts[0] for layout in layouts):
            raise ValueError('Inconsistent attribution units')
        for i, (level, name, _) in enumerate(layouts[0]):
            for cls in range(len(first['classes'])):
                score = statistics.mean(s['scores'][i]['score_pp'][cls] for s in members)
                rows.append(dict(fold=fold, case_id=case, label=members[0]['label'], level=level, name=name,
                                 class_index=cls, score_pp=score, abs_score_pp=abs(score)))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['level'], row['name'], row['class_index'])].append(row)
    rankings = []
    for (level, name, cls), values in grouped.items():
        fold_values = defaultdict(list)
        for row in values:
            fold_values[row['fold']].append(row)
        signed = [statistics.mean(r['score_pp'] for r in rs) for rs in fold_values.values()]
        absolute = [statistics.mean(r['abs_score_pp'] for r in rs) for rs in fold_values.values()]
        rankings.append(dict(level=level, name=name, class_index=cls, mean_signed_pp=statistics.mean(signed),
                             mean_absolute_pp=statistics.mean(absolute), folds=len(fold_values), patients=len(values),
                             fold_mean_absolute_pp={str(f): statistics.mean(r['abs_score_pp'] for r in rs)
                                                    for f, rs in fold_values.items()}))
    return {'condition': {k: first[k] for k in keys},
            'ranking': sorted(rankings, key=lambda x: -x['mean_absolute_pp']), 'patient_scores': rows,
            'incomplete_patients_omitted': omitted, 'explained_slides': len(slides),
            'expected_slides_in_available_folds': sum(len(v) for v in expected.values()),
            'available_folds': sorted(expected), 'full_five_fold_cohort': set(expected) == set(range(5))
            and len(slides) == sum(len(v) for v in expected.values())}


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render_html(document, path):
    """Standalone interactive signed heatmap; no external libraries or requests."""
    title = html.escape('PathoTME · TME feature attribution')
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>body{font:15px system-ui;margin:2rem auto;max-width:1250px;padding:0 1rem;color:#183044;background:#f7fafb}h1{font-size:28px}select,input{padding:.5rem;margin:.3rem}table{border-collapse:collapse;width:100%;background:white}th,td{padding:.65rem;border-bottom:1px solid #dce5e9;text-align:right}th:first-child,td:first-child{text-align:left;max-width:520px;overflow-wrap:anywhere}.scroll{overflow:auto}small{color:#475d6b}.note{background:#e9f0f5;padding:1rem;border-radius:8px}button{padding:.6rem}</style>
<h1>__TITLE__</h1><p id="scope"></p><p class="note">Score = class probability with observed TME − probability after replacing one feature or group with its training-fold reference, in percentage points. Red supports the selected class; blue opposes it. Scores are model sensitivity, are not additive, and do not establish biological causality.</p>
<label>Class <select id="cls"></select></label><label>Level <select id="level"><option value="group">Biological groups</option><option value="feature">Individual features</option></select></label><label>View <select id="view"><option value="slides">Slide heatmap</option><option value="ranking">Patient-weighted ranking</option></select></label><label>Filter <input id="filter" placeholder="Feature or group name"></label><p><small id="coverage"></small></p><div class="scroll"><table id="table"></table></div>
<p><small>Group scores come from joint replacement of all its measurements. They are not sums of individual-feature scores. Missing values use saved training-fold imputation. Feature replacement can create uncommon combinations when measurements are correlated. Raw TME measurements are not embedded. Slide identifiers are pseudonyms in this viewer.</small></p>
<script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),get=id=>document.getElementById(id);
get('scope').textContent=`${d.cohort.toUpperCase()} · ${d.method} · ${d.encoder} · ${d.shots} shots · ${d.arm} TME`;
d.classes.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.textContent=n;get('cls').appendChild(o)});
if(d.granularity==='features')get('level').value='feature';
const a=d.aggregate;get('coverage').textContent=`${a.explained_slides}/${a.expected_slides_in_available_folds} slides in ${a.available_folds.length} available fold(s). ${a.full_five_fold_cohort?'Complete five-fold cohort.':'Selected subset; not a complete cohort ranking.'} ${a.incomplete_patients_omitted.length} incomplete patient(s) excluded from rankings.`;
function cell(row,value,head=false){const c=document.createElement(head?'th':'td');c.textContent=value;row.appendChild(c);return c}
function draw(){const t=get('table');t.replaceChildren();const cl=+get('cls').value,level=get('level').value,q=get('filter').value.toLowerCase(),ranking=get('view').value==='ranking';const h=document.createElement('tr');cell(h,ranking?'Measurement':'Measurement / slide',true);
let entries;if(ranking){cell(h,'Mean |score| (pp)',true);cell(h,'Mean signed score (pp)',true);cell(h,'Patients',true);entries=a.ranking.filter(x=>x.class_index===cl&&x.level===level&&x.name.toLowerCase().includes(q));}
else{d.slides.forEach((s,i)=>cell(h,`Slide ${i+1}`,true));entries=(d.slides[0]?.scores||[]).filter(x=>x.level===level&&x.name.toLowerCase().includes(q));}t.appendChild(h);
const values=ranking?entries.map(x=>x.mean_signed_pp):entries.flatMap(x=>d.slides.map(s=>s.scores.find(u=>u.level===level&&u.name===x.name).score_pp[cl]));const max=Math.max(1e-6,...values.map(Math.abs));
entries.forEach(x=>{const r=document.createElement('tr');cell(r,x.name.replaceAll('_',' '));if(ranking){cell(r,x.mean_absolute_pp.toFixed(3));const c=cell(r,x.mean_signed_pp.toFixed(3));c.style.background=`rgba(${x.mean_signed_pp>=0?'220,70,65':'55,115,205'},${.12+.65*Math.abs(x.mean_signed_pp)/max})`;cell(r,x.patients)}else{d.slides.forEach(s=>{const u=s.scores.find(u=>u.level===level&&u.name===x.name),v=u.score_pp[cl],c=cell(r,(v>=0?'+':'')+v.toFixed(3));c.style.background=`rgba(${v>=0?'220,70,65':'55,115,205'},${.12+.65*Math.abs(v)/max})`;c.title=`Observed: ${(s.probabilities[cl]*100).toFixed(3)}%; replaced: ${(u.probability_without[cl]*100).toFixed(3)}%`;})}t.appendChild(r)})}
['cls','level','view','filter'].forEach(id=>get(id).addEventListener('input',draw));draw();</script></html>'''
    # Avoid putting patient IDs, exact feature values or private paths into HTML.
    public_view = {k: v for k, v in document.items() if k in ('cohort', 'method', 'encoder', 'shots', 'arm', 'classes', 'granularity')}
    public_view['slides'] = [{k: v for k, v in slide.items() if k in ('probabilities', 'scores')} for slide in document['slides']]
    public_view['aggregate'] = {k: v for k, v in document['aggregate'].items() if k != 'patient_scores'}
    public_view['aggregate']['incomplete_patients_omitted'] = [{} for _ in document['aggregate']['incomplete_patients_omitted']]
    payload = json.dumps(public_view, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    Path(path).write_text(template.replace('__TITLE__', title).replace('__DATA__', payload))


def export_result(document, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    document['aggregate'] = aggregate_results([document])
    (output / 'attribution.json').write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    rows = []
    for slide in document['slides']:
        for unit in slide['scores']:
            for cls, value in enumerate(unit['score_pp']):
                rows.append(dict(slide_id=slide['slide_id'], case_id=slide['case_id'], fold=document['fold'],
                                 class_name=document['classes'][cls], level=unit['level'], feature_or_group=unit['name'],
                                 score_pp=value, probability_observed=slide['probabilities'][cls],
                                 probability_replaced=unit['probability_without'][cls]))
    write_csv(output / 'slide_scores.csv', rows)
    write_csv(output / 'patient_scores.csv', document['aggregate']['patient_scores'])
    write_csv(output / 'ranking.csv', document['aggregate']['ranking'])
    render_html(document, output / 'heatmap.html')
