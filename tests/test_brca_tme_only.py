"""TME-only model selection does not fit preprocessing on validation data."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_brca_tme_only import fit_classifier


def test_training_statistics_remain_identical_with_extreme_validation_values():
    train=np.asarray([[0.,1.],[1.,np.nan],[2.,3.],[3.,4.]])
    labels=np.asarray([0,0,1,1])
    val=np.asarray([[0.,1.],[3.,4.]])
    first,c,error=fit_classifier(train,labels,val,np.asarray([0,1]),[.01,1.],7)
    second,_,_=fit_classifier(train,labels,val+10000,np.asarray([0,1]),[.01,1.],7)
    np.testing.assert_array_equal(first.named_steps["simpleimputer"].statistics_,[1.5,3.])
    for key in ("mean_","scale_"):
        np.testing.assert_array_equal(getattr(first.named_steps["standardscaler"],key),
                                      getattr(second.named_steps["standardscaler"],key))
    np.testing.assert_array_equal(first.classes_,[0,1])
    assert c==.01 and error==0
    assert first.predict_proba(val).shape==(2,2)
