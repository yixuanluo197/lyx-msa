import pytest
from scripts.analyze_meld_baseline_comparison import (
    pair_records,summarize_high_confidence,select_typical_cases,NAMES,THRESHOLDS,AUXILIARY_THRESHOLDS)


def row(index,pred,conf,model):
    values=[(1-conf)/6]*7;values[pred]=conf
    return dict(dataset='MELD',dialogue_id='dialogue',utterance_index=index,source_utterance_id=f'id-{index}',
        utterance_text=f'original-{index}',true_label=0,true_emotion='neutral',model=model,
        predicted_label=pred,predicted_emotion=NAMES[pred],predicted_confidence=conf,
        true_label_probability=values[0],correct=pred==0,**{'probability_'+n:values[k] for k,n in enumerate(NAMES)})


def test_original_and_predeclared_auxiliary_thresholds_remain_distinct():
    assert THRESHOLDS==(.8,.9,.95)
    assert AUXILIARY_THRESHOLDS==(.4,.5,.6,.7)
    paired=pair_records([row(0,1,.61,'baseline')],[row(0,2,.59,'cmerc')])
    cases=select_typical_cases(paired)
    assert all(r['threshold_group']=='auxiliary' for r in cases)
    assert all(r['threshold'] in AUXILIARY_THRESHOLDS for r in cases)
    assert all('baseline_hc_error_ge_'+str(t).replace('.','p') in paired[0] for t in THRESHOLDS+AUXILIARY_THRESHOLDS)


def test_new_hc_errors_separate_original_correct_from_already_wrong():
    base=[row(0,1,.61,'baseline'),row(1,0,.7,'baseline'),row(2,2,.59,'baseline')]
    cm=[row(0,2,.59,'cmerc'),row(1,1,.6,'cmerc'),row(2,2,.6,'cmerc')]
    paired=pair_records(base,cm);result=summarize_high_confidence(paired,.6)
    assert result['baseline_high_confidence_predictions']==2
    assert result['baseline_high_confidence_errors']==1
    assert result['cmerc_high_confidence_predictions']==2
    assert result['cmerc_high_confidence_errors']==2
    assert result['still_wrong_confidence_lower']==1
    assert result['still_wrong_above_threshold']==0
    assert result['new_cmerc_hc_errors']==2
    assert result['new_cmerc_hc_errors_from_baseline_correct']==1
    assert result['new_cmerc_hc_errors_from_baseline_wrong_below_threshold']==1
    assert paired[0]['predicted_class_changed']
    assert paired[0]['confidence_delta']<0<paired[0]['true_label_probability_delta']


def test_zero_hc_errors_and_zero_hc_predictions_are_different():
    paired=pair_records([row(0,0,.8,'baseline')],[row(0,0,.8,'cmerc')])
    assert summarize_high_confidence(paired,.8)['baseline_high_confidence_error_rate']==0
    assert summarize_high_confidence(paired,.9)['baseline_high_confidence_error_rate'] is None
    assert summarize_high_confidence(paired,.9)['corrected_ratio'] is None


@pytest.mark.parametrize('field',['utterance_text','source_utterance_id','true_label'])
def test_metadata_misalignment_rejected(field):
    b=row(0,0,.6,'baseline');c=row(0,0,.6,'cmerc');c[field]='different'
    with pytest.raises(ValueError,match=field):pair_records([b],[c])
