"""Small numerical regression checks; native compatibility is gated by GPU smoke."""
import json
import torch
from zero_smoke_checks import require_zero_tme_independence


class Conditioner(torch.nn.Module):
    def forward(self, centers, values):
        return centers + values.sum() * 1e-9


class Fixture(torch.nn.Module):
    def __init__(self, mask=True, leak=0.):
        super().__init__()
        self.centers = torch.nn.Parameter(torch.tensor([[30., 31.]]))
        self.conditioner = Conditioner()
        self.tme_mode = 'zero'
        self.mask = mask
        self.leak = leak

    def forward(self, values, return_details=True):
        masked = torch.zeros_like(values) if self.mask else values
        low = self.conditioner(self.centers, masked)
        high = self.conditioner(self.centers, masked)
        return {'logits': (low + high) / 2 + self.leak * values.sum()}


def check(model, observed=None, other=None):
    model.eval()
    raw = torch.ones(1, 62)
    a = model(raw)['logits'].detach() if observed is None else observed
    b = model(torch.full_like(raw, 1e3))['logits'].detach() if other is None else other
    try:
        return require_zero_tme_independence(model, (raw,), a, b)
    finally:
        assert not model.conditioner._forward_pre_hooks, 'temporary hook leaked'


def main():
    outcomes = []
    a = torch.tensor([[30., 31.]])
    b = a.clone()
    b[0, 1] = torch.nextafter(b[0, 1], torch.tensor(float('inf')))
    assert float((a-b).abs().max()) == 1.9073486328125e-6
    assert check(Fixture(), a, b)['exact_zero_conditioner_inputs']
    outcomes.append('recorded_FP32_rounding_accepted_with_exact_mask_and_gradient_checks')
    for name, model, x, y, error in [
        ('tiny_common_mode_leak_rejected', Fixture(leak=1e-12), None, None, 'gradient'),
        ('missing_zero_mask_rejected', Fixture(mask=False), a, a, 'unmasked'),
        ('nonfinite_rejected', Fixture(), a, torch.full_like(a, float('nan')), 'nonfinite'),
        ('meaningful_logit_change_rejected', Fixture(), a, a+1e-3, 'Tensor-likes')]:
        try:
            check(model, x, y)
        except (ValueError, AssertionError) as exc:
            assert error in str(exc), str(exc)
        else:
            raise AssertionError(name)
        outcomes.append(name)
    print(json.dumps({'status': 'passed', 'checks': outcomes, 'count': len(outcomes),
                      'scope': 'CPU synthetic numerical regression; native GPU smoke still required'}))


if __name__ == '__main__':
    main()
