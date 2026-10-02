"""Small numerical checks of reference replacement, fusion and patient weighting."""
import numpy as np
from score_core import sigmoid, linear_scores, patient_scores, rank_fold, aggregate_ranks, make_units

def main():
    z=np.array([[1.,2.],[-2.,.5],[0.,-1.]])
    coef=np.array([.7,-.3]);b=.15
    units=make_units(['a','b'],[{'name':'both','indices':[0,1]}])
    p,scores,logparts=linear_scores(z,coef,b,units)
    for j,u in enumerate(units):
        ref=z.copy();ref[:,u['indices']]=0
        expected=100*(sigmoid(z@coef+b)-sigmoid(ref@coef+b))
        np.testing.assert_allclose(scores[:,j,1],expected,atol=1e-12)
    np.testing.assert_allclose(logparts.sum(1)+b,z@coef+b)
    assert not np.allclose(scores[:,:2,1].sum(1),scores[:,2,1])
    np.testing.assert_allclose(scores[:,:,0],-scores[:,:,1])
    visual=np.array([.1,.9,.5])
    for alpha in [0.,.25,.5,1.]:
        ref=z.copy();ref[:,0]=0
        original=(1-alpha)*visual+alpha*p
        changed=(1-alpha)*visual+alpha*sigmoid(ref@coef+b)
        np.testing.assert_allclose(100*(original-changed),alpha*scores[:,0,1],atol=1e-12)
    rows=[{'case_id':'p1','label':0},{'case_id':'p1','label':0},{'case_id':'p2','label':1}]
    ids,labels,values=patient_scores(rows,scores)
    np.testing.assert_allclose(values[0],scores[:2].mean(0))
    meta={'variant':'Native','cohort':'test','panel':'two','method':'none','encoder':'none','shots':16,'fold':0,'classes':['negative','positive']}
    ranks=rank_fold(meta,rows,np.zeros_like(scores),units)
    summary=aggregate_ranks(ranks)
    assert all(r['mean_rank'] is None and r['top10_fold_fraction']==0 for r in summary)
    try:aggregate_ranks(ranks+ranks)
    except ValueError:pass
    else:raise AssertionError('Duplicate folds accepted')
    print('Passed: direct feature/group replay, nonadditivity, class signs, fusion endpoints, log-odds identity, patient weighting, zero ranking and duplicate rejection.')

if __name__=='__main__':main()
