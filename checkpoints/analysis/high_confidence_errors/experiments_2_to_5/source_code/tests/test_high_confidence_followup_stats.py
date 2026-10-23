"""CPU-only checks for follow-up diagnostic accounting and alignment."""
import copy
import csv
import importlib
import pytest


def mod():
    return importlib.import_module('scripts.high_confidence_followup_stats')


def row(index, confidence, true=1, pred=0, model='baseline', condition='full'):
    probabilities = [0.0, 0.0]
    probabilities[pred] = confidence
    probabilities[1-pred] = 1-confidence
    return dict(dataset='D', model=model, dialogue_id='d', utterance_index=index,
                source_utterance_id=str(index), utterance_text=f'text {index}',
                true_label=true, true_emotion=['a', 'b'][true], predicted_label=pred,
                predicted_emotion=['a', 'b'][pred], predicted_confidence=confidence,
                true_label_probability=probabilities[true], correct=true == pred,
                input_condition=condition, probability_a=probabilities[0], probability_b=probabilities[1])


def inputs():
    b = [row(0,.8),row(1,.9),row(2,.95),row(3,1.0,true=0),row(4,.5)]
    c = [row(0,.9,true=1,pred=1,model='cmerc'),row(1,.8,model='cmerc'),
         row(2,.95,model='cmerc'),row(3,.95,true=0,pred=1,model='cmerc'),row(4,.5,model='cmerc')]
    full={('D','baseline'):b,('D','cmerc'):c}
    masked={key:[dict(r,input_condition=cond) for cond in ('w/o_text','w/o_audio','w/o_visual') for r in records]
            for key,records in full.items()}
    return full,masked


def test_threshold_outcomes_and_distinct_denominators():
    tables=mod().generate_tables(*inputs())
    summary=next(r for r in tables['exp2_summary'] if r['threshold']==.8)
    assert summary['baseline_high_confidence_errors']==3
    assert summary['corrected_count']==1
    assert summary['still_wrong_confidence_lower_count']==1
    assert summary['still_wrong_confidence_not_lower_count']==1
    assert summary['new_cmerc_high_confidence_errors']==1
    assert summary['new_error_eligible_count']==2
    assert summary['new_error_ratio']==.5
    exp5=next(r for r in tables['exp5_overall_summary'] if r['model']=='baseline')
    assert exp5['errors_ge_0p8']==3
    assert exp5['high_confidence_predictions_ge_0p8']==4
    assert exp5['high_confidence_error_rate_ge_0p8']==.75
    assert exp5['errors_ge_0p9']==2
    assert exp5['errors_ge_0p95']==1
    assert exp5['weighted_f1']==pytest.approx(1/15)


def test_bins_cover_half_open_boundaries_and_one():
    full,masked=inputs()
    tables=mod().generate_tables(full,masked)
    bins=tables['exp5_error_confidence_distribution']
    assert sum(r['baseline_error_count'] for r in bins)==4
    assert [r['baseline_error_count'] for r in bins]==[0,1,0,0,1,1,1]
    assert sum(r['cmerc_error_count'] for r in bins)==4
    assert sum(r['cmerc_ratio_of_errors'] for r in bins)==1


def test_zero_denominators_are_none_and_empty_schema_available():
    full={('D','baseline'):[row(0,.6,true=0)],('D','cmerc'):[row(0,.6,true=0,model='cmerc')]}
    tables=mod().generate_tables(full,{})
    for r in tables['exp5_overall_summary']:
        assert r['high_confidence_error_rate_ge_0p9'] is None
        assert r['mean_error_confidence'] is None
    assert tables['exp2_paired_outcomes']==[]
    for name,rows in tables.items():
        assert name in mod().TABLE_FIELDS
        assert all(set(r)<=set(mod().TABLE_FIELDS[name]) for r in rows)


@pytest.mark.parametrize('field,value',[('true_label',0),('utterance_text','other'),('source_utterance_id','bad')])
def test_alignment_rejects_metadata_differences(field,value):
    full,masked=inputs()
    full[('D','cmerc')][0][field]=value
    with pytest.raises(ValueError):
        mod().generate_tables(full,masked)


@pytest.mark.parametrize('corruption',['missing','duplicate','order','condition','probability','argmax'])
def test_masked_validation_rejects_bad_inputs(corruption):
    full,masked=inputs(); rows=masked[('D','baseline')]
    if corruption=='missing': rows.pop()
    elif corruption=='duplicate': rows.append(copy.deepcopy(rows[0]))
    elif corruption=='order': rows[0],rows[1]=rows[1],rows[0]
    elif corruption=='condition': rows[0]['input_condition']='unknown'
    elif corruption=='probability': rows[0]['probability_a']=float('nan')
    else: rows[0]['predicted_label']=1
    with pytest.raises(ValueError): mod().generate_tables(full,masked)


def test_masked_new_errors_excludes_already_wrong():
    full,masked=inputs()
    replacement=row(3,.95,true=0,pred=1,condition='w/o_text')
    masked[('D','baseline')][3]=replacement
    tables=mod().generate_tables(full,masked)
    stats=next(r for r in tables['exp4_full_correct_to_masked_error'] if r['model']=='baseline' and r['input_condition']=='w/o_text' and r['threshold']==.9)
    assert stats['full_correct_samples']==1
    assert stats['new_high_confidence_error_count']==1
    assert stats['ratio']==1
    assert stats['full_wrong_still_wrong_count']==4
    assert stats['full_wrong_still_wrong_high_confidence_count']==2


def test_load_original_csv_keeps_text_and_maps_booleans(tmp_path):
    path=tmp_path/'D'/'baseline';path.mkdir(parents=True)
    original=dict(row(0,.8)); original['text']=original.pop('utterance_text')
    original['max_probability']=original.pop('predicted_confidence')
    original['true_class_probability']=original.pop('true_label_probability')
    original['is_error']=not original.pop('correct')
    with (path/'all_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(original));writer.writeheader();writer.writerow(original)
    loaded=mod().load_full_records(tmp_path)[('D','baseline')][0]
    assert loaded['correct'] is False
    assert loaded['utterance_text']=='text 0'
    assert loaded['predicted_confidence']==.8


def test_none_mask_input_defers_all_experiment_four_tables():
    full,_=inputs()
    tables=mod().generate_tables(full)
    assert tables['exp2_summary']
    assert tables['exp3_cm_remaining_high_confidence_errors']
    assert tables['exp5_overall_summary']
    assert all(not rows for name,rows in tables.items() if name.startswith('exp4_'))


def test_every_bin_boundary_including_confidence_one_is_counted_once():
    rows=[]
    for i,confidence in enumerate((.49,.5,.6,.7,.8,.9,.95,1.0)):
        r=row(i,confidence)
        r['probability_b']=(1-confidence)*.7
        r['probability_c']=(1-confidence)*.3
        r['true_label_probability']=r['probability_b']
        rows.append(r)
    tables=mod().generate_tables({('D','baseline'):rows})
    bins=tables['exp5_error_confidence_distribution']
    assert [r['baseline_error_count'] for r in bins]==[1,1,1,1,1,1,2]
    assert sum(r['baseline_error_count'] for r in bins)==8
    summary=tables['exp5_overall_summary'][0]
    assert summary['errors_ge_0p8']==4
    assert summary['errors_ge_0p9']==3
    assert summary['errors_ge_0p95']==2
    assert all(r['cmerc_error_count'] is None for r in bins)


def test_missing_baseline_preserves_cmerc_and_does_not_fabricate_pairs():
    full,_=inputs()
    tables=mod().generate_tables({('D','cmerc'):full[('D','cmerc')]})
    assert tables['exp2_paired_outcomes']==[]
    assert tables['exp2_summary']==[]
    assert tables['exp3_cm_remaining_high_confidence_errors']
    assert {r['error_type'] for r in tables['exp3_cm_remaining_high_confidence_errors']}=={'baseline_unavailable'}


@pytest.mark.parametrize('field,value', [('predicted_confidence',float('inf')),('true_label_probability',float('nan')),('correct','False'),('probability_a',1.1)])
def test_full_prediction_validation_rejects_scalar_corruptions(field,value):
    full,_=inputs()
    full[('D','baseline')][0][field]=value
    with pytest.raises(ValueError): mod().generate_tables(full)
