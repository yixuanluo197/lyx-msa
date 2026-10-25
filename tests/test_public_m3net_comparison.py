import pytest
from scripts.analyze_public_m3net_comparison import (
    pair_records, summarize_high_confidence, select_typical_cases, NAMES)


def prediction(index, true, predicted, confidence, model):
    probabilities=[(1-confidence)/5]*6;probabilities[predicted]=confidence
    return dict(dataset='IEMOCAP',model=model,dialogue_id='dialogue',utterance_index=index,
        source_utterance_id=f'source-{index}',utterance_text=f'utterance {index}',true_label=true,
        true_emotion=NAMES[true],predicted_label=predicted,predicted_emotion=NAMES[predicted],
        predicted_confidence=confidence,true_label_probability=probabilities[true],correct=true==predicted,
        **{'probability_'+n:probabilities[i] for i,n in enumerate(NAMES)})


def test_threshold_boundary_new_error_subtypes_and_overlapping_remaining_hc():
    # HCE corrected; lower but remains HCE; lower and leaves HCE; unchanged HCE;
    # new HCE from a formerly correct prediction; new HCE from a low-confidence error.
    base=[prediction(i,0,p,c,'Official M³Net') for i,p,c in [(0,1,.9),(1,1,.95),(2,1,.91),(3,1,.9),(4,0,.95),(5,1,.89)]]
    cm=[prediction(i,0,p,c,'Official CMERC') for i,p,c in [(0,0,.9),(1,2,.91),(2,1,.89),(3,1,.9),(4,1,.9),(5,2,.9)]]
    paired=pair_records(base,cm);summary=summarize_high_confidence(paired,.9)
    assert summary['baseline_high_confidence_errors']==4
    assert (summary['corrected'],summary['still_wrong_confidence_lower'],summary['still_wrong_confidence_not_lower'])==(1,2,1)
    assert summary['still_wrong_above_threshold']==2
    assert summary['new_cmerc_hc_errors']==2
    assert summary['new_cmerc_hc_errors_from_baseline_correct']==1
    assert summary['new_cmerc_hc_errors_from_baseline_wrong_below_threshold']==1
    assert summary['baseline_high_confidence_predictions']==5
    assert summary['cmerc_high_confidence_predictions']==5
    assert summary['baseline_high_confidence_error_rate']==.8
    assert summary['corrected_ratio']==.25
    assert summary['new_cmerc_hc_error_ratio_of_eligible']==1
    # A lower maximum with a different wrong class is explicitly separate from p(true).
    assert paired[1]['predicted_class_changed']
    assert paired[1]['confidence_delta']==pytest.approx(-.04)
    assert paired[1]['true_label_probability_delta']!=paired[1]['confidence_delta']
    cases=[r for r in select_typical_cases(paired) if r['threshold']==.9]
    counts={t:sum(r['case_type']==t for r in cases) for t in {r['case_type'] for r in cases}}
    assert counts=={'corrected_baseline_hc_error':1,'still_wrong_confidence_lower':2,
        'baseline_correct_new_cmerc_hc_error':1,'both_high_confidence_wrong':2}


def test_empty_high_confidence_denominators_are_missing():
    rows=pair_records([prediction(0,0,0,.6,'Official M³Net')],[prediction(0,0,0,.7,'Official CMERC')])
    result=summarize_high_confidence(rows,.95)
    assert result['baseline_high_confidence_error_rate'] is None
    assert result['cmerc_high_confidence_error_rate'] is None
    assert result['corrected_ratio'] is None


@pytest.mark.parametrize('failure',['duplicate','text','label','order'])
def test_matching_length_does_not_imply_alignment(failure):
    base=[prediction(i,0,0,.8,'Official M³Net') for i in range(2)]
    cm=[prediction(i,0,0,.8,'Official CMERC') for i in range(2)]
    if failure=='duplicate':cm[1]['utterance_index']=0
    if failure=='text':cm[1]['utterance_text']='changed'
    if failure=='label':cm[1]['true_label']=1
    if failure=='order':cm.reverse()
    with pytest.raises(ValueError):pair_records(base,cm)
