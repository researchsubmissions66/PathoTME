"""Prediction attribution for the existing quantitative TME panels.

Feature/group replacement is performed after the checkpoint's fold-fitted
standardizer. Zero here means the training mean in transformed feature space,
not absence of cells/tissue. Scores are not additive or causal explanations.
"""
from __future__ import annotations

import random


def panel_layout(cohort):
    """Return the exact ordered measurements and native semantic groups."""
    from pathotme import features as f
    from pathotme.brca_features import panel_spec
    if cohort == 'brca':
        spec = panel_spec('brca_morph64_v1')
        names = list(spec['feature_names'])
        low, high = spec['low_tokens'], spec['high_tokens']
    elif cohort == 'nsclc':
        names = list(f.FEATURE_NAMES)
        low = dict(zip(('global_tissue', 'compartment_extent', 'tumor_core_composition',
                        'invasive_margin_composition', 'tls'),
                       (f.GLOBAL_TISSUE, f.WHOLE_TUMOR_TISSUE[:3],
                        f.WHOLE_TUMOR_TISSUE[3:6], f.WHOLE_TUMOR_TISSUE[6:8], f.TLS_DERIVED)))
        high = {f'cell_{cell.lower()}': [n for n in f.CELL_COMPOSITION if f'_{cell}_IN_' in n]
                for cell in f.CELL_CLASSES}
        high.update({f'spatial_{cell.lower()}': [n for n in f.SPATIAL_INTERACTIONS
                                               if f'_OF_{cell}_AROUND_' in n]
                     for cell in f.SPATIAL_CELL_GROUPS})
    else:
        raise ValueError('Only the current NSCLC core62 and BRCA morph64 panels are supported')
    groups = [{'name': scale + '/' + name, 'indices': [names.index(n) for n in columns]}
              for scale, mapping in [('low', low), ('high', high)] for name, columns in mapping.items()]
    covered = [i for group in groups for i in group['indices']]
    if sorted(covered) != list(range(len(names))) or len(groups) != 16:
        raise ValueError('Native groups must cover every feature exactly once')
    return names, groups


def ablate_tme(model, forward, feature_names, groups, *, granularity='both', progress=None):
    """Return signed class-probability deltas from a fixed slide and checkpoint.

    ``forward`` must evaluate that same slide and return [1, classes]
    probabilities. Each intervention reruns the complete downstream model,
    including hard selection/routing. Model modes, hooks and RNG are restored
    even if a forward fails. No parameters, input tensors or buffers are edited.
    """
    import numpy as np
    import torch
    if granularity not in ('groups', 'features', 'both'):
        raise ValueError('Unknown granularity')
    width = len(feature_names)
    if width != model.standardizer.feature_count or not bool(model.standardizer.fitted.item()):
        raise ValueError('Load the fitted adapter standardizer before attribution; never refit it')
    if len(set(feature_names)) != width:
        raise ValueError('Feature names must be unique')
    if sorted(i for g in groups for i in g['indices']) != list(range(width)):
        raise ValueError('Groups must partition the complete feature vector')
    modes = [(module, module.training) for module in model.modules()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

    def reset_rng():
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])

    captured = {}
    def run(indices=None, capture=False):
        reset_rng()
        calls = []
        def intervention(_module, inputs, output):
            if output.shape != (1, width) or not torch.isfinite(output).all():
                raise ValueError('Expected one finite standardized TME row')
            calls.append(True)
            if capture:
                captured['standardized'] = output.detach().cpu().tolist()[0]
                captured['imputed'] = torch.isnan(inputs[0]).detach().cpu().reshape(-1).tolist()
            if indices is None:
                return output
            replacement = output.clone()
            replacement[:, list(indices)] = 0
            return replacement
        handle = model.standardizer.register_forward_hook(intervention)
        try:
            probability = forward().detach().clone()
            if len(calls) != 1:
                raise ValueError('The standardizer must be consumed exactly once per prediction')
            if (probability.ndim != 2 or probability.shape[0] != 1 or probability.shape[1] < 2
                    or not torch.isfinite(probability).all() or (probability < 0).any()
                    or (probability > 1).any()
                    or not torch.allclose(probability.sum(-1), probability.new_ones(1), atol=1e-5, rtol=1e-5)):
                raise ValueError('The provider must return class probabilities')
            return probability
        finally:
            handle.remove()

    try:
        model.eval()
        with torch.no_grad():
            baseline = run(capture=True)
            all_reference = run(range(width))
            units = []
            if granularity in ('groups', 'both'):
                units.extend(dict(g, level='group') for g in groups)
            if granularity in ('features', 'both'):
                units.extend({'name': n, 'indices': [i], 'level': 'feature'} for i, n in enumerate(feature_names))
            results = []
            for position, unit in enumerate(units):
                changed = run(unit['indices'])
                results.append({**unit, 'probability_without': changed.cpu().tolist()[0],
                                'score_pp': ((baseline - changed) * 100).cpu().tolist()[0]})
                if progress is not None:
                    progress(position + 1, len(units))
            restored = run()
            torch.testing.assert_close(restored, baseline, rtol=0, atol=1e-7)
        return {'probabilities': baseline.cpu().tolist()[0],
                'all_reference_probabilities': all_reference.cpu().tolist()[0],
                'all_tme_score_pp': ((baseline - all_reference) * 100).cpu().tolist()[0],
                'standardized_values': captured['standardized'], 'imputed_features': captured['imputed'],
                'scores': results, 'forward_calls': len(units) + 3, 'restoration_checked': True}
    finally:
        reset_rng()
        for module, training in modes:
            module.training = training
