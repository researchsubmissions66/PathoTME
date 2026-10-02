"""Small numerical/registration checks; no native weights or training."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from pathotme.attribution import ablate_tme, panel_layout
from pathotme.attribution_io import aggregate_results, render_html
from pathotme.attribution_spatial import thumbnail_source, write_thumbnail_overlay
from pathotme.guided_vila import FoldStandardizer, SemanticTMETokenizer
from pathotme.brca_vila import BreastTMETokenizer


class SmallModel(nn.Module):
    def __init__(self, zero=False):
        super().__init__()
        self.standardizer = FoldStandardizer(2)
        self.standardizer.fit(np.array([[1., 3.], [3., 5.]]))
        self.zero = zero
    def forward(self, value):
        z = self.standardizer(value)
        if self.zero:
            z = torch.zeros_like(z)
        score = z[:,0] - 2*z[:,1]
        return torch.stack([-score, score], -1).softmax(-1)


GROUPS = [{'name': 'joint', 'indices': [0,1]}]


def test_signed_feature_and_joint_scores_and_frozen_state():
    model = SmallModel(); model.train()
    inputs = torch.tensor([[3., 5.]])
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = ablate_tme(model, lambda: model(inputs), ['a','b'], GROUPS)
    by_name = {u['name']:u for u in result['scores']}
    assert by_name['a']['score_pp'][1] > 0 > by_name['b']['score_pp'][1]
    assert by_name['joint']['score_pp'] == result['all_tme_score_pp']
    assert not np.isclose(sum(by_name[x]['score_pp'][1] for x in ('a','b')), by_name['joint']['score_pp'][1])
    assert model.training and model.standardizer.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert not model.standardizer._forward_hooks
    for k,v in model.state_dict().items():
        assert torch.equal(v,before[k])


def test_zero_model_has_zero_tme_dependence():
    model = SmallModel(zero=True)
    result = ablate_tme(model, lambda: model(torch.tensor([[3.,5.]])), ['a','b'], GROUPS)
    assert all(u['score_pp'] == [0.,0.] for u in result['scores'])


def test_exception_restores_hook_and_modes():
    model = SmallModel(); model.train(); model.standardizer.eval()
    count = []
    def fail():
        count.append(1)
        if len(count) == 3:
            raise RuntimeError('intentional interruption')
        return model(torch.tensor([[3.,5.]]))
    with pytest.raises(RuntimeError, match='intentional'):
        ablate_tme(model, fail, ['a','b'], GROUPS)
    assert model.training and not model.standardizer.training
    assert not model.standardizer._forward_hooks


@pytest.mark.parametrize('cohort,width,cls', [('nsclc',62,SemanticTMETokenizer),('brca',64,BreastTMETokenizer)])
def test_groups_match_native_tokenizer(cohort,width,cls):
    names, groups = panel_layout(cohort)
    tokenizer = cls(8, dropout=0)
    high = getattr(tokenizer, 'high_indices', None)
    if high is None:
        high = (*tokenizer.cell_indices, *tokenizer.spatial_indices)
    expected = [list(x) for x in (*tokenizer.low_indices, *high)]
    assert len(names) == width and [g['indices'] for g in groups] == expected


def example_document():
    return dict(score_version='test', cohort='brca', method='muse', encoder='plip', shots=16, arm='actual',
                feature_names=['a'], groups=[{'name':'a','indices':[0]}], classes=['IDC','ILC'], granularity='groups',
                fold=0, provenance={'checkpoint_sha256':{'native':'a','adapter':'b'}},
                expected_test_slides=[{'slide_id':'s1','case_id':'p1','label':0},
                                      {'slide_id':'s2','case_id':'p1','label':0},
                                      {'slide_id':'s3','case_id':'p2','label':1}], slides=[])


def slide(name,case,label,score):
    return dict(slide_id=name,case_id=case,label=label,probabilities=[.5,.5],
                scores=[dict(level='group',name='a',indices=[0],score_pp=[-score,score],probability_without=[.5+score/100,.5-score/100])])


def test_patient_weighting_partial_coverage_and_duplicate_rejection():
    doc = example_document(); doc['slides'] = [slide('s1','p1',0,2),slide('s2','p1',0,4),slide('s3','p2',1,9)]
    result = aggregate_results([doc])
    assert result['ranking'][0]['mean_absolute_pp'] == 6  # mean(patient mean=3, patient=9)
    assert not result['full_five_fold_cohort']
    doc['slides'].pop(1)
    result = aggregate_results([doc])
    assert len(result['incomplete_patients_omitted']) == 1
    assert result['ranking'][0]['patients'] == 1
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate_results([doc,doc])


def test_thumbnail_registration_and_real_background(tmp_path):
    from PIL import Image
    thumb = tmp_path/'thumbnail.png'
    Image.new('RGB',(100,50),(95,130,155)).save(thumb)
    with pytest.raises(ValueError, match='aspect'):
        thumbnail_source(thumbnail=thumb,dimensions=(1000,1000))
    spatial = {'maps':[{'scale':'low','coordinates':[[100,100]],'patch_size_level0':100,
                       'source_patch_indices':[0],'group_ids':[0],
                       'groups':[{'group_id':0,'patch_count':1,'observed_scores_pp':[-3.,3.],
                                  'reference_scores_pp':[-1.,1.],'difference_scores_pp':[-2.,2.]}]}],
               'forward_calls':8}
    out = tmp_path/'overlay'
    write_thumbnail_overlay(spatial,out,classes=['IDC','ILC'],reference_name='all',thumbnail=thumb,dimensions=(1000,500))
    image = Image.open(out/'overlay.png')
    assert image.getpixel((0,0)) == (95,130,155)
    assert image.getpixel((15,15)) != (95,130,155)
    assert 'data:image/png;base64,' in (out/'heatmap.html').read_text()


def test_html_escapes_and_does_not_embed_patient_identifiers(tmp_path):
    doc = example_document(); doc['classes'][0]='</script><img src=x onerror=alert(1)>'
    doc['slides']=[slide('s1','p1',0,1),slide('s2','p1',0,1),slide('s3','p2',1,1)]
    doc['slides'][0]['case_id']='PRIVATE_PATIENT'
    doc['expected_test_slides'][0]['case_id']='PRIVATE_PATIENT'
    doc['aggregate']=aggregate_results([doc])
    render_html(doc,tmp_path/'heatmap.html')
    output=(tmp_path/'heatmap.html').read_text()
    assert 'PRIVATE_PATIENT' not in output and '</script><img' not in output


def test_spatial_difference_measures_tme_visual_interaction(monkeypatch):
    from types import SimpleNamespace
    import common.spatial_attribution as common_spatial
    import common.interpretability as interpretation
    from pathotme.attribution_spatial import spatial_tme_scores
    class SpatialModel(SmallModel):
        def forward(self, values, tme):
            z = self.standardizer(tme)
            score = values.mean() * (.1 + .3*z[:,0])
            return torch.stack([-score,score],-1).softmax(-1)
    model = SpatialModel().eval()
    tme = torch.tensor([[3.,5.]])
    batch = [torch.tensor([[[1.,2.],[1.,4.],[4.,5.],[6.,7.]]])]
    original = batch[0].clone()
    bridge = SimpleNamespace(cfg={'n_classes':2}, name='muse',
                             eval_step_with_details=lambda value, model: {'probabilities':model(value[0],tme)})
    binding = common_spatial.SpatialInput(0,'patch','feature_path_column',Path('synthetic.h5'),
                torch.tensor([[0,0],[16,0],[32,0],[48,0]]),torch.arange(4),(2,),torch.arange(4))
    monkeypatch.setattr(common_spatial,'bind_spatial_inputs',lambda *a:[binding])
    monkeypatch.setattr(interpretation,'load_coordinate_bag',lambda *a:SimpleNamespace(patch_size_level0=16))
    result = spatial_tme_scores(bridge,model,batch,{'slide_id':'synthetic'},reference_indices=[0,1],max_groups=2)
    scores = result['maps'][0]['groups']
    assert any(abs(g['difference_scores_pp'][1])>1e-4 for g in scores)
    for group in scores:
        assert np.allclose(group['difference_scores_pp'],np.array(group['observed_scores_pp'])-group['reference_scores_pp'])
    assert torch.equal(original,batch[0]) and not model.standardizer._forward_hooks
