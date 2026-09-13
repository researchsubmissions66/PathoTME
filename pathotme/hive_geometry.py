"""CPU-only copy of native HiVE geometry validation, plus coordinate audit.

The two geometry functions are copied unchanged from PGVL methods/hive_mil/dataset.py.
"""
import numpy as np
import hashlib
from pathlib import Path


def hierarchy_indices(low_coords, high_coords, span):
    """Reproduce native stored child order using an x-sorted window search."""
    low_coords, high_coords = np.asarray(low_coords), np.asarray(high_coords)
    for coords in (low_coords, high_coords):
        if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
            raise ValueError('nonempty [N,2] coordinates required')
        if len(np.unique(coords, axis=0)) != len(coords):
            raise ValueError('duplicate coordinates')
    if span <= 0:
        raise ValueError('positive parent span required')
    order = np.argsort(high_coords[:, 0], kind='stable')
    xs = high_coords[order, 0]
    parents, children = [], []
    for i, (x, y) in enumerate(low_coords):
        candidate = order[np.searchsorted(xs, x):np.searchsorted(xs, x + span)]
        inside = np.sort(candidate[(high_coords[candidate, 1] >= y)
                                  & (high_coords[candidate, 1] < y + span)])
        if len(inside) > 16:
            raise ValueError(f'parent {i} has {len(inside)} children; native maximum is 16')
        if len(inside):
            parents.append(i)
            children.append(np.pad(inside, (0, 16-len(inside)), constant_values=-1))
    if not parents:
        raise ValueError('no parent with a high-power child')
    return np.asarray(parents, dtype=np.int64), np.asarray(children, dtype=np.int64)


def array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype='<i8').tobytes()).hexdigest()


def audit_pair(task):
    """Read two headers and coordinate arrays; no image feature payloads."""
    import h5py
    slide_id, low_path, high_path, encoder, width = task
    coordinates, attrs, files = [], [], {}
    for scale, path in [(5, low_path), (20, high_path)]:
        path = Path(path)
        before = path.stat()
        with h5py.File(path, 'r') as h:
            f, c = h['features'], h['coords']
            if (f.ndim != 2 or f.shape[1] != width or not f.shape[0]
                    or c.shape != (f.shape[0], 2) or f.attrs.get('encoder') != encoder
                    or int(c.attrs.get('patch_size', -1)) != 224
                    or float(c.attrs.get('target_magnification', -1)) != scale):
                raise ValueError(f'feature identity mismatch: {path}')
            coords = np.asarray(c[:], dtype=np.int64)
            coordinates.append(coords); attrs.append(dict(c.attrs))
            files[str(path)] = dict(size=before.st_size, mtime_ns=before.st_mtime_ns,
                shape=list(f.shape), encoder=encoder, magnification=scale, patch_size=224,
                coords_sha256=array_sha(coords),
                extraction_checkpoint_attestation='historical digest absent; registry/header binding only')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'feature changed during audit: {path}')
    geometry = resolve_hierarchy_geometry(*attrs)
    parents, children = hierarchy_indices(*coordinates, geometry['low_patch_span_level0'])
    counts = (children >= 0).sum(1)
    return slide_id, dict(files=files, geometry=geometry,
        raw_parents=len(coordinates[0]), retained_parents=len(parents),
        dropped_empty_parents=len(coordinates[0])-len(parents),
        valid_child_count=int(counts.sum()), padded_child_count=int((children < 0).sum()),
        child_count_histogram={str(i):int((counts == i).sum()) for i in range(1,17)},
        retained_parent_indices_sha256=array_sha(parents), child_indices_sha256=array_sha(children))

def _positive_integral_attr(
    attrs: dict[str, object], key: str, *, scale: str,
) -> int:
    """Read a required positive integer-valued HDF5 coordinate attribute."""
    if key not in attrs:
        raise ValueError(
            f"HiVE-MIL {scale} coords are missing required HDF5 attr {key!r}")
    try:
        value = float(attrs[key])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"HiVE-MIL {scale} coords attr {key!r} is not numeric: "
            f"{attrs[key]!r}") from error
    rounded = int(round(value))
    if not np.isfinite(value) or value <= 0 or not np.isclose(value, rounded):
        raise ValueError(
            f"HiVE-MIL {scale} coords attr {key!r} must be a positive "
            f"integer, got {attrs[key]!r}")
    return rounded

def resolve_hierarchy_geometry(
    low_attrs: dict[str, object],
    high_attrs: dict[str, object],
    *,
    max_children: int = 16,
) -> dict[str, int]:
    """Validate paired Trident metadata and derive the 5x parent span.

    Trident coordinates are expressed in the source slide's level-0 frame.
    Consequently the absolute span depends on the scanner magnification (for
    example, 2048 at 20x and 4096 at 40x) and cannot be a global constant.
    """
    if max_children != 16:
        raise ValueError(
            "HiVE-MIL's released 5x-to-20x hierarchy requires 16 children")

    for key in ("level0_width", "level0_height", "level0_magnification"):
        low_value = _positive_integral_attr(low_attrs, key, scale="low")
        high_value = _positive_integral_attr(high_attrs, key, scale="high")
        if low_value != high_value:
            raise ValueError(
                f"HiVE-MIL paired coordinate frames disagree on {key}: "
                f"low={low_value}, high={high_value}")

    low_target = _positive_integral_attr(
        low_attrs, "target_magnification", scale="low")
    high_target = _positive_integral_attr(
        high_attrs, "target_magnification", scale="high")
    if (low_target, high_target) != (5, 20):
        raise ValueError(
            "HiVE-MIL requires paired 5x/20x bags, got "
            f"{low_target}x/{high_target}x")

    low_patch_size = _positive_integral_attr(
        low_attrs, "patch_size", scale="low")
    high_patch_size = _positive_integral_attr(
        high_attrs, "patch_size", scale="high")
    low_span = _positive_integral_attr(
        low_attrs, "patch_size_level0", scale="low")
    high_span = _positive_integral_attr(
        high_attrs, "patch_size_level0", scale="high")
    level0_mag = _positive_integral_attr(
        low_attrs, "level0_magnification", scale="low")

    expected_low_span = low_patch_size * level0_mag / low_target
    expected_high_span = high_patch_size * level0_mag / high_target
    if not np.isclose(low_span, expected_low_span):
        raise ValueError(
            "HiVE-MIL low patch span contradicts its HDF5 magnification "
            f"metadata: stored={low_span}, expected={expected_low_span:g}")
    if not np.isclose(high_span, expected_high_span):
        raise ValueError(
            "HiVE-MIL high patch span contradicts its HDF5 magnification "
            f"metadata: stored={high_span}, expected={expected_high_span:g}")

    if low_span % high_span:
        raise ValueError(
            f"HiVE-MIL patch spans are not integral: {low_span}/{high_span}")
    linear_ratio = low_span // high_span
    magnification_ratio = high_target // low_target
    if (linear_ratio != magnification_ratio
            or linear_ratio * linear_ratio != max_children):
        raise ValueError(
            "HiVE-MIL paired spans do not define the released 4x4 hierarchy: "
            f"low={low_span}, high={high_span}, children={max_children}")

    return {
        "low_patch_span_level0": low_span,
        "high_patch_span_level0": high_span,
        "level0_magnification": level0_mag,
        "linear_ratio": linear_ratio,
    }
