"""Check zero-TME masking and gradients independently of FP32 logit rounding."""
import torch


def require_zero_tme_independence(model, inputs, observed, varied_logits):
    if model.training or model.tme_mode != 'zero':
        raise ValueError('zero control must be evaluated in eval mode')
    if observed.dtype != torch.float32 or varied_logits.dtype != torch.float32:
        raise ValueError('this bounded tolerance is validated for FP32 logits')
    tolerance = 4 * torch.finfo(torch.float32).eps
    if not torch.isfinite(observed).all() or not torch.isfinite(varied_logits).all():
        raise ValueError('nonfinite zero-control logits')
    torch.testing.assert_close(observed, varied_logits, rtol=tolerance, atol=tolerance)
    calls = []

    def inspect_mask(module, args):
        values = args[1]
        if values.shape != (1, 62) or not torch.isfinite(values).all() or torch.count_nonzero(values):
            raise ValueError('conditioner received unmasked TME values')
        calls.append(True)

    handle = model.conditioner.register_forward_pre_hook(inspect_mask)
    try:
        for raw, reference in [(inputs[-1], observed),
                               (torch.full_like(inputs[-1], 1e3), varied_logits)]:
            values = raw.detach().clone().requires_grad_()
            count = len(calls)
            logits = model(*inputs[:-1], values, return_details=True)['logits']
            if len(calls) - count != 2:
                raise ValueError('expected exactly two masked conditioner calls per slide')
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite zero-control probe')
            torch.testing.assert_close(logits.detach(), reference, rtol=tolerance, atol=tolerance)
            # Check each class, including common-mode leakage that a margin can hide.
            for index in range(logits.shape[-1]):
                gradient = torch.autograd.grad(logits[:, index].sum(), values,
                                               retain_graph=True, allow_unused=True)[0]
                if gradient is not None and (not torch.isfinite(gradient).all() or torch.count_nonzero(gradient)):
                    raise ValueError('zero control has a nonzero TME gradient')
    finally:
        handle.remove()
    return {'policy': 'fp32_parity_exact_mask_and_per_logit_zero_gradient',
            'zero_input_independent': True, 'conditioner_mask_calls': len(calls),
            'exact_zero_conditioner_inputs': True, 'per_logit_tme_gradients_zero_or_unused': True,
            'logit_max_abs_delta': float((observed - varied_logits).abs().max()),
            'rtol': tolerance, 'atol': tolerance}
