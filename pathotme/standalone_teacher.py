"""Standalone subtype teachers; no Torch or WSI-model dependency."""
from __future__ import annotations

CONDITIONS = ('student_ce', 'kd_visual', 'kd_tme_only', 'kd_tme_only_shuffled')
NEW_CONDITIONS = CONDITIONS[2:]


def subtype_logits(classifier, values):
    """Return ordered IDC/ILC logits without clipping or double tempering.

    sklearn's binary decision function is the ILC-versus-IDC log odds.
    Symmetric logits preserve its probabilities and are safe at temperature 2.
    """
    import numpy as np
    if list(classifier.classes_) != [0, 1]:
        raise ValueError('teacher class order must be IDC=0, ILC=1')
    score = np.asarray(classifier.decision_function(values), dtype=float)
    if score.shape != (len(values),) or not np.isfinite(score).all():
        raise ValueError('nonfinite or nonbinary teacher decision function')
    return np.stack([-.5*score, .5*score], axis=1)


def source_arrays(phases, rows, panel, donors=None):
    """Transform only train/validation rows; never construct test TME targets."""
    import numpy as np
    from pathotme.brca_features import transform_rows
    if set(phases) != {'train', 'val'}:
        raise ValueError('teacher accepts train/val phases only')
    lookup = {}
    for row in rows:
        key = row['slide_id']
        if key in lookup:
            raise ValueError('duplicate quantitative slide')
        lookup[key] = row
    seen = set()
    arrays = {}; labels = {}
    for phase, split in phases.items():
        ids = [r['slide_id'] for r in split]
        cases = {r['case_id'] for r in split}
        if len(set(ids)) != len(ids) or seen & cases:
            raise ValueError('teacher split duplication or patient leakage')
        seen.update(cases)
        mapping = {s: s for s in ids} if donors is None else donors[phase]
        if set(mapping) != set(ids) or set(mapping.values()) != set(ids):
            raise ValueError('teacher donors must be a split-local bijection')
        if donors is not None:
            patients = {r['slide_id']: r['case_id'] for r in split}
            if any(patients[s] == patients[d] for s, d in mapping.items()):
                raise ValueError('teacher donor must be a different patient')
        arrays[phase] = np.asarray(transform_rows([lookup[mapping[s]] for s in ids], panel))
        labels[phase] = np.asarray([int(r['label_id']) for r in split])
        if set(labels[phase]) != {0, 1}:
            raise ValueError('teacher needs both ordered classes')
    return arrays, labels


def complementarity(visual, teacher, labels):
    """Describe validation error overlap; this is not an automatic launch gate."""
    import numpy as np
    v = np.asarray(visual); t = np.asarray(teacher); y = np.asarray(labels)
    if v.shape != t.shape or v.shape != (len(y), 2) or not len(y):
        raise ValueError('unmatched validation predictions')
    for p in (v, t):
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
            raise ValueError('invalid validation probabilities')
    vc = v.argmax(1) == y; tc = t.argmax(1) == y
    return {'slides': len(y), 'visual_accuracy': float(vc.mean()),
            'teacher_accuracy': float(tc.mean()),
            'both_correct': int((vc & tc).sum()), 'both_wrong': int((~vc & ~tc).sum()),
            'teacher_correct_visual_wrong': int((~vc & tc).sum()),
            'visual_correct_teacher_wrong': int((vc & ~tc).sum()),
            'oracle_either_correct_accuracy': float((vc | tc).mean()),
            'note': 'Source validation only; oracle is not an achievable model score.'}
