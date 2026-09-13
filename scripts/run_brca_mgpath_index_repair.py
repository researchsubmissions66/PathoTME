#!/usr/bin/env python3
"""Apply one source-bound BRCA indexing correction in this process only."""
import json
from pathlib import Path
import runpy
import sys
sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym']

if __name__=='__main__':
    def argument(name):return sys.argv[sys.argv.index(name)+1]
    if argument('--cohort')!='brca' or argument('--method')!='mgpath':
        raise ValueError('this correction is restricted to BRCA MGPATH')
    launch=json.loads(Path(argument('--launch')).read_text())
    if launch.get('runtime_amendment',{}).get('kind')!='brca_pandas_column_tuple_to_list':
        raise ValueError('explicit source-bound repair contract required')
    import pathotme.locked_models as factory
    from pathotme.brca_mgpath_index_repair import IndexedBreastGuidedMGPathMethod
    factory.BreastGuidedMGPathMethod=IndexedBreastGuidedMGPathMethod
    # The unchanged runner verifies the full amended source/asset contract
    # before doing any work. Other Slurm processes retain their original code.
    runpy.run_path('/path/to/PathoTME/scripts/run_locked_tcga.py',run_name='__main__')
