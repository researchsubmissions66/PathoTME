"""Read the existing TME panels for CPU-only MUSE study preparation."""
import pandas as pd
from pathotme.features import FEATURE_NAMES, transform_nsclc_features, validate_feature_table
from pathotme.brca_features import panel_spec, transform_rows
from pathotme.locked_tcga import rows


def load_tme(path, cohort):
    """Apply the same core62/morph64 transforms used by the current study."""
    table = pd.read_csv(path)
    if cohort == 'nsclc':
        result = transform_nsclc_features(validate_feature_table(table))
        if tuple(result.columns) != FEATURE_NAMES:
            raise ValueError('core62 order mismatch')
        result.index = table.slide_id.astype(str)
    elif cohort == 'brca':
        names = panel_spec('brca_morph64_v1')['feature_names']
        if list(table.columns) != ['slide_id', *names]:
            raise ValueError('morph64 order mismatch')
        result = pd.DataFrame(transform_rows(rows(path), 'brca_morph64_v1'),
                              columns=names, index=table.slide_id.astype(str))
    else:
        raise ValueError('MUSE study requires NSCLC or BRCA')
    if result.index.has_duplicates:
        raise ValueError('duplicate TME rows')
    return result
