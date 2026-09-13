"""Structural DyKo checks at the binary decision margin, before softmax."""
import torch


def require_tme_margin_effect(logits, varied_logits, tme_gradient):
    """Reject disconnected/common-mode changes even when probabilities saturate."""
    if logits.shape != (1, 2) or varied_logits.shape != logits.shape:
        raise ValueError('binary single-bag logits required')
    if not all(torch.isfinite(x).all() for x in (logits, varied_logits)):
        raise ValueError('nonfinite TME probe logits')
    margin = logits[:, 1] - logits[:, 0]
    varied_margin = varied_logits[:, 1] - varied_logits[:, 0]
    if torch.allclose(margin, varied_margin, rtol=1e-7, atol=1e-7):
        raise ValueError('real TME does not affect the class decision margin')
    if (tme_gradient is None or not torch.isfinite(tme_gradient).all()
            or not tme_gradient.abs().sum() > 0):
        raise ValueError('missing, zero or nonfinite TME-to-margin gradient')
    probabilities = logits.softmax(-1)
    varied_probabilities = varied_logits.softmax(-1)
    return {
        'decision_margin': float(margin.item()),
        'varied_decision_margin': float(varied_margin.item()),
        'decision_margin_abs_delta': float((margin-varied_margin).abs().max()),
        'probability_max_abs_delta': float((probabilities-varied_probabilities).abs().max()),
        'old_probability_check_would_pass': not torch.allclose(
            probabilities, varied_probabilities, rtol=1e-7, atol=1e-7),
        'tme_margin_gradient_l1': float(tme_gradient.abs().sum()),
    }


def check_model(model, inputs, arm):
    """Check native parity, label independence, zero control and real TME path."""
    features, labels, tme = inputs
    model.eval()
    projection = model.conditioner.output_projection.weight
    with torch.no_grad():
        saved_projection = projection.clone()
        try:
            projection.zero_()
            native = model.base(features)['logits']
            observed = model(*inputs, return_details=True)['logits']
            torch.testing.assert_close(observed, native, rtol=1e-5, atol=1e-5)
            relabeled = model(features, 1-labels, tme, return_details=True)['logits']
            torch.testing.assert_close(observed, relabeled, rtol=0, atol=0)
        finally:
            projection.copy_(saved_projection)
        observed = model(*inputs, return_details=True)['logits']
        other = model(features, labels, torch.full_like(tme, 1e3), return_details=True)['logits']
        relabeled = model(features, 1-labels, tme, return_details=True)['logits']
        torch.testing.assert_close(observed, relabeled, rtol=0, atol=0)
    if arm == 'zero':
        torch.testing.assert_close(observed, other, rtol=0, atol=0)
        if not torch.isfinite(observed).all():
            raise ValueError('nonfinite zero-control logits')
        return {'policy': 'decision_margin_and_tme_gradient_v2',
                'zero_input_independent': True, 'zero_residual_native_equivalence': True,
                'label_independent_predictions': True}
    if arm not in ('actual', 'shuffled'):
        raise ValueError('unexpected TME arm')
    probe = tme.detach().clone().requires_grad_()
    logits = model(features, labels, probe, return_details=True)['logits']
    margin = logits[:, 1] - logits[:, 0]
    gradient = torch.autograd.grad(margin.sum(), probe, allow_unused=True)[0]
    return {'policy': 'decision_margin_and_tme_gradient_v2',
            'zero_residual_native_equivalence': True, 'label_independent_predictions': True,
            **require_tme_margin_effect(observed, other, gradient)}


def regression_checks():
    """Small numerical counterexamples: saturated signal and false-positive paths."""
    probe = torch.tensor([[1.0]], requires_grad=True)
    logits = torch.cat((torch.full_like(probe, 32), probe), dim=1)
    other = torch.tensor([[32.0, 2.0]])
    gradient = torch.autograd.grad((logits[:,1]-logits[:,0]).sum(),probe)[0]
    result = require_tme_margin_effect(logits, other, gradient)
    assert not result['old_probability_check_would_pass']
    assert result['decision_margin_abs_delta'] == 1.0
    rejected = 0
    for varied, grad in ((logits.detach()+5,gradient), (other,torch.zeros_like(gradient)),
                         (other,None), (other,torch.full_like(gradient,float('nan')))):
        try:
            require_tme_margin_effect(logits.detach(), varied, grad)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('invalid TME path passed')
    assert rejected == 4
    return {'saturated_margin_signal_accepted': True,
            'common_mode_disconnected_zero_and_nonfinite_rejected': rejected}
