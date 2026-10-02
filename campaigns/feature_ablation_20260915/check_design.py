"""Small checks for exact panel partitions, mask nesting, and RFE agreement."""
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from run_ablation import groups_for, fixed_conditions, training_subsets, common_spec, panel_spec


def main():
    for panel in ['shared_core62_v1','brca_morph64_v1']:
        spec=common_spec() if panel.startswith('shared') else panel_spec(panel)
        task={'panel':panel,'feature_names':list(spec['feature_names'])}
        groups=groups_for(task)
        expected=[14,28,16,4] if panel.startswith('shared') else [16,28,12,8]
        assert [len(v) for v in groups.values()]==expected
        conditions=fixed_conditions(task)
        assert len(conditions)==39
        assert len({r['name'] for r in conditions})==39
        for repeat in range(10):
            masks=[set(r['feature_names']) for r in conditions if r['kind']=='random' and r['repeat']==repeat]
            assert len(masks)==3 and masks[0]<masks[1]<masks[2]
        for g,members in groups.items():
            only=next(r for r in conditions if r['name']=='group_only_'+g)
            drop=next(r for r in conditions if r['name']=='leave_group_out_'+g)
            assert set(only['feature_names']).isdisjoint(drop['feature_names'])
            assert set(only['feature_names'])|set(drop['feature_names'])==set(task['feature_names'])
    rng=np.random.default_rng(29311)
    x=rng.normal(size=(40,36)); y=(x[:,0]-2*x[:,1]+x[:,2]>.1).astype(int)
    names=[f'f{i:02d}' for i in range(36)]
    subsets,history=training_subsets(x,y,names,1)
    assert len(history)==28
    for r in subsets:
        reference=RFE(LogisticRegression(C=1,solver='lbfgs',max_iter=5000,tol=1e-8,random_state=1),
                      n_features_to_select=r['k'],step=1).fit(x,y)
        assert r['feature_names']==[n for n,keep in zip(names,reference.support_) if keep]
    print('Passed: core62/morph64 partitions, nested random subsets, group complements, training RFE agreement at 8/16/32 features.')


if __name__=='__main__':main()
