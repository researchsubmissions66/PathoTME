"""Anonymous, offline feature ranking viewer and a concise research report."""
import csv
import json
from pathlib import Path


def render_report(output):
    output=Path(output)
    with (output/'feature_rankings.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for key in ['mean_absolute_pp','mean_signed_pp','fold_sd_absolute_pp','mean_rank','top10_fold_fraction']:
            r[key]=float(r[key]) if r[key] else None
        for key in ['folds','patients','class_index','shots']:r[key]=int(r[key])
        r['complete_five_fold']=r['complete_five_fold']=='True'
    summary=json.loads((output/'summary.json').read_text())
    public=json.dumps({'rows':rows,'summary':{k:summary[k] for k in ['recorded_at_utc','completed_variant_folds','waiting_fusion_folds','neural_adapter_scores_included']}},allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PathoTME feature attribution</title>
<style>body{font:15px system-ui,sans-serif;max-width:1450px;margin:2rem auto;padding:0 1rem;color:#24314a;background:#faf8fd}h1{color:#622b88}label{display:inline-block;margin:.4rem}select,input{display:block;font:inherit;padding:.45rem;max-width:280px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:.6rem;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left;overflow-wrap:anywhere}thead{background:#ece4f4}p{max-width:1000px;line-height:1.6}.scroll{overflow-x:auto}#status{font-weight:600}.note{color:#526078}a{color:#622b88}</style>
<h1>PathoTME · Feature attribution</h1><p>How much does the predicted class probability change when a feature or biological group is replaced with its saved training-fold reference? Scores are in percentage points (pp). Positive values support the selected class; negative values oppose it. Features retain their exact panel names.</p>
<p class="note">Rank by mean absolute patient score, then average equally across available folds. SD describes variation between fold means; it is not a confidence interval. Group interventions are joint and nonadditive. These scores describe model sensitivity, not causal biology or validation of the panel selection.</p>
<div id="controls"></div><label>Find a feature<input id="search" type="search" placeholder="e.g. LYMPHOCYTE"></label><p id="status"></p>
<div class="scroll"><table><thead><tr><th>Feature / group</th><th>Mean |score| (pp)</th><th>Fold SD</th><th>Mean signed score (pp)</th><th>Top 10 fold fraction</th><th>Folds</th><th>Patients</th></tr></thead><tbody id="table"></tbody></table></div>
<p><a href="feature_rankings.csv" download>Download all feature rankings</a> · <a href="lr_coefficient_stability.csv" download>LR coefficient directions across folds</a> · <a href="REPORT.md">Read the report</a></p>
<p class="note">LR has no visual encoder. Fusion TME scores equal the frozen fusion weight multiplied by the LR score: this is an exact property of probability fusion, not independent evidence of a different TME explanation. A Native-only endpoint has zero TME score. Native has zero TME dependence by construction. No spatial localization is inferred from a slide-level measurement.</p>
<script>const data=__DATA__;const fields=['cohort','variant','method','encoder','class_name','level'];const labels=['Cohort','Variant','Underlying model','Visual encoder','Explained class','Granularity'];const selects={};const root=document.getElementById('controls');
fields.forEach((f,i)=>{const l=document.createElement('label');l.textContent=labels[i];const s=document.createElement('select');s.id=f;l.appendChild(s);root.appendChild(l);selects[f]=s;s.addEventListener('change',()=>{update(i+1);draw()})});
function update(from=0){for(let i=from;i<fields.length;i++){const f=fields[i],s=selects[f],old=s.value;const allowed=data.rows.filter(r=>fields.slice(0,i).every(k=>String(r[k])===selects[k].value));const options=[...new Set(allowed.map(r=>String(r[f])))].sort();s.replaceChildren();options.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o)});if(options.includes(old))s.value=old;else if(f==='variant'&&options.includes('PathoTME-LR'))s.value='PathoTME-LR';else if(f==='level'&&options.includes('feature'))s.value='feature';}}
function draw(){const query=document.getElementById('search').value.toLowerCase();const rows=data.rows.filter(r=>fields.every(f=>String(r[f])===selects[f].value)&&r.feature_name.toLowerCase().includes(query)).sort((a,b)=>b.mean_absolute_pp-a.mean_absolute_pp||a.feature_name.localeCompare(b.feature_name));const body=document.getElementById('table');body.replaceChildren();const limit=Math.max(1e-12,...rows.map(r=>Math.abs(r.mean_signed_pp)));rows.forEach(r=>{const tr=document.createElement('tr');const vals=[r.feature_name,r.mean_absolute_pp.toFixed(4),r.fold_sd_absolute_pp===null?'—':r.fold_sd_absolute_pp.toFixed(4),r.mean_signed_pp.toFixed(4),(100*r.top10_fold_fraction).toFixed(0)+'%',r.folds+'/5',r.patients];vals.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===3){td.style.background=`rgba(${r.mean_signed_pp>=0?'209,65,63':'55,107,200'},${.05+.5*Math.abs(r.mean_signed_pp)/limit})`;}tr.appendChild(td)});body.appendChild(tr)});document.getElementById('status').textContent=rows.length?`${rows.length} measurements · ${rows[0].folds}/5 folds · ${rows[0].patients} held-out patients · ${rows[0].complete_five_fold?'Complete fold coverage':'Partial fold coverage'}`:'No matching scores';}
document.getElementById('search').addEventListener('input',draw);update();draw();</script></html>'''
    (output/'feature_heatmap.html').write_text(template.replace('__DATA__',public))
    lines=['# PathoTME feature attribution','',f"Snapshot: {summary['recorded_at_utc']}",'',
           'The feature score is 100 × [class probability with observed TME − class probability after replacing the feature with its saved transformed training-fold mean]. Group scores replace the entire group jointly. Positive scores support the explained class; negative scores oppose it. Scores are nonadditive and do not establish causality.','',
           'Magnitude first averages signed slide scores within each patient, then absolute patient scores within each fold, then averages folds equally. The table shows sample SD across fold means, not a confidence interval. All held-out patients and all features are included in the completed LR fits.','',
           '## Coverage','', '| Variant | Completed condition/folds |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in summary['completed_variant_folds'].items()]
    lines += ['',f"{summary['waiting_fusion_folds']} fusion evaluations await saved results. Neural adapter coverage is recorded separately; it is not implied by this LR/fusion export.",'',
              '## Features with the largest PathoTME-LR sensitivity','',
              'Each table explains the positive class (class 1). Mean absolute sensitivity is a ranking measure; it does not mean higher feature values increase the class probability. The coefficient direction table supplies that separate LR-specific information.','']
    for cohort in sorted({r['cohort'] for r in rows}):
        selected=[r for r in rows if r['variant']=='PathoTME-LR' and r['cohort']==cohort and r['class_index']==1 and r['level']=='feature']
        selected.sort(key=lambda r:(-r['mean_absolute_pp'],r['feature_name']))
        lines += [f"### TCGA-{cohort.upper()} · {selected[0]['class_name']}",'', '| Feature | Mean absolute score ± fold SD (pp) | Top 10 fold frequency |','|---|---:|---:|']
        for r in selected[:10]:
            sd='—' if r['fold_sd_absolute_pp'] is None else f"{r['fold_sd_absolute_pp']:.3f}"
            lines.append(f"| `{r['feature_name']}` | {r['mean_absolute_pp']:.3f} ± {sd} | {r['top10_fold_fraction']*100:.0f}% |")
        lines.append('')
    lines += ['## Interpretation boundaries','',
              '- Fusion scores are exactly alpha times LR probability-change scores with the native prediction fixed. They are not independent feature discoveries.',
              '- LR coefficients describe class-1 log-odds per unit of the saved standardized feature. Per-slide log-odds contributions are saved separately and are not probability-point attributions.',
              '- Correlated features and composition constraints can make isolated mean replacement biologically unrealistic. Prefer agreement with the joint biological-group view.',
              '- Historical BRCA uses morph64; NSCLC/CRC/BLCA use core62. Keep panel and cohort identifiers attached to every result.',
              '- Feature importance is not statistical significance, a causal biomarker, an a priori feature-selection justification, or spatial localization.',
              '- Raw patient features are not embedded in HTML or copied into the source repository. Detailed score arrays remain in private storage.','',
              '[Interactive feature heatmap](feature_heatmap.html) · [All feature/group rankings](feature_rankings.csv) · [Coefficient stability](lr_coefficient_stability.csv)','']
    (output/'REPORT.md').write_text('\n'.join(lines))
