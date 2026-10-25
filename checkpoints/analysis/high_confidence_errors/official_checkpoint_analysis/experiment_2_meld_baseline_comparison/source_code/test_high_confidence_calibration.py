"""Hand-calculated, CPU-only calibration regression cases."""
import copy
import importlib
import math

import pytest


def module():
    return importlib.import_module('scripts.high_confidence_calibration')


def records(probabilities, labels):
    rows=[]
    for i,(probs,true) in enumerate(zip(probabilities,labels,strict=True)):
        pred=max(range(len(probs)),key=probs.__getitem__)
        names=[f'class_{j}' for j in range(len(probs))]
        r=dict(dataset='D',model='baseline',input_condition='full',dialogue_id='d',
               utterance_index=i,source_utterance_id=f'u{i}',utterance_text=f'text {i}',
               true_label=true,true_emotion=names[true],predicted_label=pred,
               predicted_emotion=names[pred],predicted_confidence=probs[pred],
               true_label_probability=probs[true],correct=true==pred)
        r.update({f'probability_{name}':p for name,p in zip(names,probs,strict=True)})
        rows.append(r)
    return rows


def test_hand_calculated_calibration_brier_and_nll():
    rows=records([[.8,.2],[.6,.4],[.3,.7],[.1,.9]],[0,1,1,0])
    before=copy.deepcopy(rows)
    summary,bins=module().summarize_calibration(rows)
    assert rows==before
    assert summary['dataset']=='D' and summary['model']=='baseline'
    assert summary['input_condition']=='full'
    assert summary['valid_utterances']==4
    assert summary['accuracy']==.5
    assert summary['mean_confidence']==pytest.approx(.75)
    assert summary['ece']==pytest.approx((.2+.6+.3+.9)/4)
    assert summary['mce']==pytest.approx(.9)
    assert summary['brier_score']==pytest.approx((.08+.72+.18+1.62)/4)
    assert summary['nll']==pytest.approx(-sum(map(math.log,[.8,.4,.7,.1]))/4)
    assert summary['zero_true_probability_count']==0
    assert summary['n_bins']==10
    assert len(bins)==10
    assert sum(b['count'] for b in bins)==4
    assert sum(b['ece_contribution'] for b in bins)==pytest.approx(summary['ece'])
    assert bins[8]['signed_gap']==pytest.approx(-.2)
    assert bins[9]['signed_gap']==pytest.approx(.9)


@pytest.mark.parametrize('confidence,index',[(0,0),(.1,1),(.3,3),(.9,9),(1,9),(math.nextafter(.3,0),2)])
def test_bin_boundaries(confidence,index):
    assert module()._bin_index(confidence,10)==index


def test_confidence_point_three_is_in_fourth_bin_with_valid_probabilities():
    rows=records([[.3,.25,.25,.2]],[0])
    _,bins=module().summarize_calibration(rows)
    assert bins[3]['count']==1
    assert bins[2]['count']==0


def test_empty_bins_use_undefined_accuracy_and_gap():
    _,bins=module().summarize_calibration(records([[.9,.1]],[0]))
    for b in bins[:9]:
        assert b['count']==0 and b['weight']==0 and b['ece_contribution']==0
        for field in ('mean_confidence','accuracy','signed_gap','absolute_gap'):
            assert b[field] is None
    assert bins[9]['lower_bound']==.9 and bins[9]['upper_bound']==1


def test_perfect_predictions_have_zero_losses_and_calibration_errors():
    summary,bins=module().summarize_calibration(records([[1.,0.],[0.,1.]],[0,1]))
    assert summary['accuracy']==1
    assert summary['ece']==summary['mce']==summary['brier_score']==summary['nll']==0
    assert bins[9]['count']==2


def test_all_wrong_one_hot_predictions_have_infinite_nll_and_brier_two():
    summary,bins=module().summarize_calibration(records([[1.,0.],[0.,1.]],[1,0]))
    assert summary['accuracy']==0
    assert summary['ece']==summary['mce']==1
    assert summary['brier_score']==2
    assert math.isinf(summary['nll'])
    assert summary['zero_true_probability_count']==2
    assert bins[9]['accuracy']==0


def test_mixed_bin_uses_bin_accuracy_and_mean_confidence_not_mean_absolute_errors():
    summary,bins=module().summarize_calibration(records([[.81,.19],[.89,.11]],[0,1]))
    assert bins[8]['accuracy']==.5
    assert bins[8]['mean_confidence']==pytest.approx(.85)
    assert summary['ece']==pytest.approx(.35)
    assert summary['mce']==pytest.approx(.35)


@pytest.mark.parametrize('corruption', ['mapping','nan','duplicate','label','order','confidence','true_probability','condition','group','id'])
def test_invalid_prediction_records_are_rejected(corruption):
    rows=records([[.8,.2],[.3,.7]],[0,1])
    if corruption=='mapping': rows[0]['true_emotion']='different'
    elif corruption=='nan': rows[0]['probability_class_0']=float('nan')
    elif corruption=='duplicate': rows[1]=copy.deepcopy(rows[0])
    elif corruption=='label': rows[0]['true_label']=2
    elif corruption=='order':
        value=rows[1].pop('probability_class_0');rows[1]['probability_class_0']=value
    elif corruption=='confidence': rows[0]['predicted_confidence']=float('inf')
    elif corruption=='true_probability': rows[0]['true_label_probability']=-1e-9
    elif corruption=='condition': rows[0]['input_condition']='w/o_text'
    elif corruption=='group': rows[1]['dataset']='different'
    else: rows[0]['dialogue_id']=''
    with pytest.raises(ValueError): module().summarize_calibration(rows)


@pytest.mark.parametrize('n_bins',[0,-1,True,1.5,None])
def test_invalid_bin_count_is_rejected(n_bins):
    with pytest.raises(ValueError): module().summarize_calibration(records([[.8,.2]],[0]),n_bins)


def test_empty_input_requires_identifiable_model_and_is_rejected():
    with pytest.raises(ValueError): module().summarize_calibration([])


def test_custom_bin_count_preserves_sample_count_and_brier_scaling():
    summary,bins=module().summarize_calibration(records([[.5,.5]],[1]),2)
    assert len(bins)==2 and sum(b['count'] for b in bins)==1
    assert bins[1]['count']==1
    assert summary['brier_score']==.5
    assert summary['n_bins']==2
