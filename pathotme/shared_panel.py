"""One fixed core62 feature definition for four TCGA cohorts.

The legacy measurement order, unit transforms and 5+11 semantic groups are
reused unchanged. Only source directories/project membership vary by cohort.
Old campaign contracts and checkpoints are not modified by this module.
"""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

from pathotme import features
from pathotme.brca_features import panel_spec as breast_panel_spec, number

COHORT_SOURCES = {
    'nsclc': ('lung_cancer', ('TCGA-LUAD', 'TCGA-LUSC')),
    'brca': ('breast_cancer', ('TCGA-BRCA',)),
    'crc': ('colorectal_cancer', ('TCGA-COAD', 'TCGA-READ')),
    'blca': ('bladder_cancer', ('TCGA-BLCA',)),
}


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def common_spec():
    """Return a cohort-independent numeric and biological group identity."""
    reference = breast_panel_spec('brca_shared62_v1')
    spec = dict(panel='shared_core62_v1', revision=reference['revision'],
        feature_names=features.FEATURE_NAMES,
        low_tokens=reference['low_tokens'], high_tokens=reference['high_tokens'],
        percent_to_fraction=features.PERCENT_TO_FRACTION,
        log1p_nonnegative=features.LOG1P_NONNEGATIVE,
        tls_derivation='legacy_missing_count_zero_then_presence_and_log1p',
        other_missing='preserve_NaN_until_training_fold_median_imputation',
        fold_fitted='training_median_then_training_mean_population_std_no_indicators',
        constant_training_scale='standard_deviation_below_1e-8_becomes_1',
        all_missing_training_column='reject_fold_do_not_select_a_new_panel')
    covered = [c for g in (*spec['low_tokens'].values(), *spec['high_tokens'].values()) for c in g]
    if len(covered) != 62 or len(set(covered)) != 62 or set(covered) != set(features.FEATURE_NAMES):
        raise ValueError('Common groups must cover exactly the unchanged core62 panel')
    spec['feature_schema_sha256'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


def panel_spec(cohort):
    if cohort not in COHORT_SOURCES:
        raise ValueError(f'Unsupported cohort: {cohort}')
    site, projects = COHORT_SOURCES[cohort]
    spec = deepcopy(common_spec())
    spec.update(cohort=cohort, site=site, projects=projects,
        sources={name.replace('lung_cancer', site): columns
                 for name, columns in features.SOURCE_COLUMNS.items()})
    return spec


def selected_rows(root, cohort):
    """Join the full original filename/UUID without dropping source rows."""
    spec = panel_spec(cohort)
    root = Path(root)
    directory = root if root.name == spec['site'] else root/'data'/spec['site']
    tables, source_hashes = [], {}
    for filename, columns in spec['sources'].items():
        path = directory/filename
        source_hashes[filename] = digest_file(path)
        table = {}
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            missing = {'TCGA_FILE_NAME', 'TCGA_PROJECT_ID', *columns} - set(reader.fieldnames)
            if missing:
                raise ValueError(f'{filename} missing columns: {sorted(missing)}')
            for row in reader:
                if row['TCGA_PROJECT_ID'].replace('_', '-') not in spec['projects']:
                    raise ValueError(f'Unexpected project in {filename}')
                key = row['TCGA_FILE_NAME'].lower().removesuffix('.svs')
                if key in table:
                    raise ValueError(f'Duplicate original slide identity in {filename}')
                values = {c:number(row[c]) for c in columns}
                if any(v < 0 for v in values.values()):
                    raise ValueError(f'Negative source measurement in {filename}')
                table[key] = values
        tables.append(table)
    if not tables[0] or any(set(t) != set(tables[0]) for t in tables[1:]):
        raise ValueError('Unequal or empty source slide universes; no inner-join dropping')
    rows, identities = [], {}
    for key in sorted(tables[0]):
        barcode = key.split('.')[0].upper()
        if barcode in identities:
            raise ValueError('Multiple UUIDs share a slide barcode')
        identities[barcode] = key
        values = {c:v for t in tables for c,v in t[key].items()}
        for stage in ('IMMATURE', 'MATURE'):
            raw = values[f'COUNT_TLS_{stage}']
            count = 0.0 if math.isnan(raw) else raw
            values[f'TLS_{stage}_PRESENT'] = float(count > 0)
            values[f'LOG1P_COUNT_TLS_{stage}'] = math.log1p(count)
        if all(math.isnan(values[c]) for c in features.FEATURE_NAMES[:-4]):
            raise ValueError('Slide has no non-TLS measurements')
        rows.append(dict(slide_id=barcode, **{c:values[c] for c in features.FEATURE_NAMES}))
    return rows, identities, source_hashes


def transform_rows(rows):
    """Use the same deterministic transforms; learn no statistics here."""
    result = []
    for row in rows:
        values = []
        for column in features.FEATURE_NAMES:
            value = number(row[column])
            if value < 0:
                raise ValueError('Negative measurement')
            if column in features.PERCENT_TO_FRACTION:
                value /= 100.0
            if column in features.LOG1P_NONNEGATIVE:
                value = math.log1p(value)
            values.append(value)
        result.append(values)
    return result


def write_panel(root, cohort, output):
    """Write a fresh private numeric table, identities and source provenance."""
    output = Path(output)
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Raw numeric features must be stored outside the code repository')
    metadata_path = output.with_suffix('.metadata.json')
    if output.exists() or metadata_path.exists():
        raise FileExistsError(output)
    rows, identities, source_hashes = selected_rows(root, cohort)
    spec = panel_spec(cohort)
    metadata = dict(spec=spec, slide_count=len(rows), source_file_sha256=source_hashes,
        original_slide_identities=identities, redistribution='prohibited',
        implementation_sha256=digest_file(__file__),
        missing_counts={c:sum(math.isnan(r[c]) for r in rows) for c in features.FEATURE_NAMES},
        label_data_used=False, folds_fitted=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['slide_id', *features.FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    metadata['selected_table_sha256'] = digest_file(output)
    with metadata_path.open('x') as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return metadata
