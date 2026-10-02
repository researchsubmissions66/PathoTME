"""Small numerical checks for split, preprocessing and fusion selection boundaries."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[2]), '/path/to/PGVL-Gym']
from baseline_core import fit_scaler, scale_values, select_alpha, fuse, checked_predictions, read_splits


def main():
    # An extreme held-out value cannot change training median/mean/std.
    train = np.array([[1., 2.], [3., np.nan], [5., 6.]])
    state = fit_scaler(train)
    assert state['median'] == [3., 4.] and state['mean'] == [3., 4.]
    assert np.allclose(scale_values(train, state).mean(0), 0)
    snapshot = str(state)
    scale_values([[1e8, np.nan]], state)
    assert str(state) == snapshot
    try:
        fit_scaler([[1., np.nan], [2., np.nan]])
    except ValueError:
        pass
    else:
        raise AssertionError('All-missing training feature was accepted')

    val = pd.DataFrame({'slide_id': ['a', 'b', 'c', 'd'], 'case_id': ['A', 'B', 'C', 'D'], 'label': [0, 0, 1, 1]})
    visual = np.array([.1, .2, .8, .9]); tme = 1-visual
    selected, _ = select_alpha(val, visual, tme)
    assert selected['alpha'] == 0
    assert np.array_equal(fuse(visual, tme, 0), visual)
    assert np.array_equal(fuse(visual, tme, 1), tme)
    assert np.allclose(fuse(visual, tme, .5), .5)
    identical, _ = select_alpha(val, visual, visual)
    assert identical['alpha'] == 0  # fully tied weights favor native

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pred = val.assign(probability_0=1-visual, probability_1=visual)
        pred.iloc[::-1].to_csv(root/'pred.csv', index=False)
        checked_predictions(root/'pred.csv', val)
        pred.loc[0, 'label'] = 1
        pred.to_csv(root/'bad.csv', index=False)
        try:
            checked_predictions(root/'bad.csv', val)
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong prediction labels were accepted')
        for phase in ['train', 'val', 'test']:
            val.to_csv(root/f'{phase}.csv', index=False)
        try:
            read_splits({'label_dict': {}, 'splits': {p: str(root/f'{p}.csv') for p in ['train', 'val', 'test']}})
        except ValueError:
            pass
        else:
            raise AssertionError('Patient leakage was accepted')
    print('PASS: train-only preprocessing, missing-column rejection, fusion selection/endpoints, exact prediction joins and patient leakage checks')


if __name__ == '__main__':
    main()
