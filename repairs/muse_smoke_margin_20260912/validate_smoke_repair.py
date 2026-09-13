"""Run bounded numerical regressions and verify the saved real-bag CPU replay."""
import json
from pathlib import Path
import sys
import time
import hashlib

print('Loading repair-check dependencies', flush=True)
import torch
from muse_smoke_checks import regression_checks, require_tme_margin_effect
checks = regression_checks()
print(json.dumps(checks), flush=True)
path = Path('/tmp/pathotme_muse_diagnosis_20260912.json')
deadline = time.monotonic()+1800
while True:
    if path.exists():
        diagnosis=json.loads(path.read_text())
        if len(diagnosis['groups'])==4 and 'wall_seconds' in diagnosis:
            break
    if time.monotonic()>deadline:
        raise TimeoutError('real-bag diagnostic report incomplete')
    time.sleep(2)
results=[]
for group in diagnosis['groups']:
    assert group['tme_margin_gradient_finite'] and group['base_gradient_absent']
    assert any(v>0 for v in group['gradient_l1'].values())
    decision=require_tme_margin_effect(torch.tensor(group['logits']),
        torch.tensor(group['varied_logits']),torch.tensor([group['tme_margin_gradient_l1']]))
    results.append({'cohort':group['cohort'],'encoder':group['encoder'],'decision_check':decision})
report={'status':'passed','regressions':checks,'real_training_bag_replays':results,
        'diagnosis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(Path(__file__).parent.glob('*.py'))},
        'live_gpu_smoke':False}
Path('/tmp/pathotme_muse_smoke_repair_validation_20260912.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'passed','real_bag_groups':len(results)}),flush=True)
