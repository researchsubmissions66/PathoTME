"""Dependency-light contracts for the controlled CONCH ViLa study."""
from __future__ import annotations

import hashlib
import json
import random

CONDITIONS = (
    "actual", "zero", "shuffled", "fusion", "student_ce",
    "aux_real", "aux_shuffled", "kd_visual", "kd_real", "kd_shuffled",
)
STUDENTS = CONDITIONS[4:]
SHUFFLE_POLICY = "split_local_patient_excluding_slide_bijection_v1"


def donor_maps(phases, seed, fold):
    """Permute complete rows without labels, self-patients or cross-split donors.

    This is a slide bijection, not a patient-block permutation when a patient
    has multiple test slides. A separate local RNG leaves training RNG intact.
    """
    result = {}
    patients = {}
    all_slides = set()
    for phase, rows in phases.items():
        cases = {str(r["slide_id"]): str(r["case_id"]) for r in rows}
        if len(cases) != len(rows) or set(cases) & all_slides:
            raise ValueError("duplicate or cross-split slide")
        all_slides.update(cases)
        patients[phase] = set(cases.values())
        slides = sorted(cases)
        if not slides or max(list(cases.values()).count(c) for c in patients[phase]) * 2 > len(slides):
            raise ValueError("patient-excluding bijection is impossible")
        token = json.dumps([SHUFFLE_POLICY, seed, fold, phase])
        rng = random.Random(int(hashlib.sha256(token.encode()).hexdigest(), 16))
        donors = slides.copy()
        for _ in range(10000):
            rng.shuffle(donors)
            if all(cases[s] != cases[d] for s, d in zip(slides, donors)):
                break
        else:
            raise ValueError("could not construct a valid donor permutation")
        result[phase] = dict(zip(slides, donors))
    keys = list(patients)
    if any(patients[a] & patients[b] for i, a in enumerate(keys) for b in keys[i+1:]):
        raise ValueError("patient leakage across phases")
    return result


def select_fusion(visual, tme, labels, grid):
    """Select a convex blend on validation error, with visual-favoring ties."""
    import numpy as np
    visual, tme, labels = np.asarray(visual), np.asarray(tme), np.asarray(labels)
    if visual.shape != tme.shape or visual.shape != (len(labels), 2) or not len(labels):
        raise ValueError("invalid fusion inputs")
    for p in (visual, tme):
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
            raise ValueError("invalid probabilities")
    grid = sorted(set(float(a) for a in grid))
    if not grid or any(not 0 <= a <= 1 for a in grid):
        raise ValueError("invalid fusion grid")
    scores = [{"alpha": a, "val_error": float((((1-a)*visual+a*tme).argmax(1) != labels).mean())}
              for a in grid]
    return min(scores, key=lambda r: (r["val_error"], r["alpha"]))["alpha"], scores
