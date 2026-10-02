"""Cohort identity checks and architecture-independent numerical smoke checks."""
import json, os
from pathlib import Path
from pathotme.locked_tcga import rows
from pathotme.focus_contract import validate_donors
from runtime_models import validate_config, native_type

def selected(launch,cohort,method,encoder,fold):
    found=[p for p in launch['plans'] if (p['cohort'],p['method'],p['encoder'],p['fold'])==(cohort,method,encoder,fold)]
    if len(found)!=1:raise ValueError('Expected one frozen fold plan')
    return found[0]

def check_features(plan,cfg):
    validate_config(cfg)
    audit=json.loads(Path(plan['feature_inventory']).read_text())
    phases={phase:rows(Path(cfg['split_dir'])/f"fold{plan['fold']}/{phase}.csv") for phase in ['train','val','test']}
    validate_donors(phases,json.loads(Path(plan['donor_maps']).read_text()))
    for values in phases.values():
        for row in values:
            for path,bound in audit['slides'][row['slide_id']]['files'].items():
                stat=Path(path).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(bound['size'],bound['mtime_ns']):
                    raise ValueError(f'Feature changed since full-payload audit: {path}')

def evaluate(loader,bridge,model,metric_fn):
    import pandas as pd
    from run_vila_guided import _run_epoch,_metric_bundle
    result=_run_epoch(loader,bridge,model,metric_fn)
    p=result['probabilities'];frame=pd.DataFrame(result['metadata'])
    frame['label']=result['labels'];frame['prediction']=p.argmax(-1)
    frame[['probability_0','probability_1']]=p
    return frame,_metric_bundle(p,result['labels'],result['metadata'],metric_fn)

def check_model(bridge,model,batch,native_cfg,arm):
    import torch
    from pathotme.mscpt_smoke_checks import require_tme_margin_effect
    method=native_cfg['method'];encoder=native_cfg['backbone']
    native=native_type(method,encoder)(native_cfg,bridge.device)
    model.eval();projection=model.conditioner.output_projection.weight
    relabeled=list(batch);relabeled[-1]=1-batch[-1]
    with torch.no_grad():
        saved=projection.clone()
        try:
            projection.zero_()
            reference=native.eval_step(batch,model.base)['logits'].softmax(-1)
            observed=bridge.eval_step(batch,model)['logits'].softmax(-1)
            torch.testing.assert_close(observed,reference,rtol=1e-5,atol=1e-5)
        finally:projection.copy_(saved)
        reference=bridge.eval_step(batch,model)['logits']
        changed=bridge.eval_step(relabeled,model)['logits']
        torch.testing.assert_close(reference,changed,rtol=0,atol=0)
    inputs=(bridge._inputs(batch,model)[0] if method=='vila_mil' else bridge.inputs(batch)[0] if method=='mgpath' else bridge.inputs(batch,model)[0])
    def logits(tme):
        detail=model(*(*inputs[:-1],tme),return_details=True)
        return detail['logits'] if 'logits' in detail else detail['probabilities'].clamp_min(1e-30).log()
    with torch.no_grad():
        observed=logits(inputs[-1]);varied=logits(torch.full_like(inputs[-1],1e3))
    diagnostic={'zero_residual_native_equivalence':True,'label_independent_predictions':True}
    if arm=='zero':
        torch.testing.assert_close(observed,varied,rtol=0,atol=0)
        return {**diagnostic,'zero_input_independent':True}
    probe=inputs[-1].detach().clone().requires_grad_()
    result=logits(probe);gradient=torch.autograd.grad((result[:,1]-result[:,0]).sum(),probe,allow_unused=True)[0]
    effect = require_precision_resolved_margin if method=='dyko' else require_tme_margin_effect
    return {**diagnostic,**effect(observed,varied,gradient)}


def require_precision_resolved_margin(logits,varied_logits,gradient):
    """Check DyKo decision dependence relative to its small native logit scale.

    The legacy fixed 1e-7 absolute threshold exceeds the rounding scale of
    these ~0.02 logits. Require a nonzero decision gradient and a finite
    difference larger than eight epsilons at the observed output scale.
    This is a structural check; the initial effect may be extremely small.
    """
    import torch
    if logits.shape!=(1,2) or varied_logits.shape!=logits.shape:raise ValueError('Binary single-bag logits required')
    if not all(torch.isfinite(x).all() for x in [logits,varied_logits]):raise ValueError('Nonfinite logits')
    if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum()>0:raise ValueError('Disconnected or invalid TME decision gradient')
    before=logits.detach().double();after=varied_logits.detach().double()
    margin=before[:,1]-before[:,0];other=after[:,1]-after[:,0]
    delta=float((margin-other).abs().max())
    scale=max(float(before.abs().max()),float(after.abs().max()),torch.finfo(logits.dtype).tiny)
    resolution=8*torch.finfo(logits.dtype).eps*scale
    if delta<=resolution:raise ValueError('TME margin change does not exceed the output rounding scale')
    return {'policy':'nonzero_decision_gradient_and_eight_epsilon_margin_v1',
            'decision_margin':float(margin.item()),'varied_decision_margin':float(other.item()),
            'decision_margin_abs_delta':delta,'output_rounding_bound':resolution,
            'tme_margin_gradient_l1':float(gradient.abs().sum()),
            'fixed_absolute_1e_minus7_check_would_pass':not torch.allclose(logits[:,1]-logits[:,0],varied_logits[:,1]-varied_logits[:,0],rtol=1e-7,atol=1e-7),
            'probability_max_abs_delta':float((logits.softmax(-1)-varied_logits.softmax(-1)).abs().max())}
