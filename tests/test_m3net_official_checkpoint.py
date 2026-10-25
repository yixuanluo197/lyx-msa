from pathlib import Path
import pytest
from scripts.verify_m3net_official_checkpoint import (
    FIXED_CHECKPOINT, fixed_checkpoint_for_request, assert_reference_alignment)


def test_official_default_request_maps_only_to_fixed_author_m3net_weight(tmp_path):
    requested=tmp_path/'checkpoints/IEMOCAP_checkpoint.pkl'
    assert fixed_checkpoint_for_request(requested,tmp_path)==tmp_path/FIXED_CHECKPOINT


@pytest.mark.parametrize('name',['MELD_checkpoint.pkl','Roberta_IEMOCAP1.pth.tar','Roberta_IEMOCAP2.pth.tar'])
def test_no_other_candidate_or_dataset_can_be_loaded(tmp_path,name):
    with pytest.raises(ValueError):
        fixed_checkpoint_for_request(tmp_path/'checkpoints'/name,tmp_path)


@pytest.mark.parametrize('field',['dialogue_id','utterance_index','source_utterance_id','utterance_text','true_label','true_emotion'])
def test_matching_row_count_is_insufficient_for_alignment(field):
    row=dict(dataset='IEMOCAP',dialogue_id='Ses05F_test',utterance_index=1,
             source_utterance_id='original-source',utterance_text='Original text',true_label=2,true_emotion='neutral')
    original=dict(row);original[field]=str(row[field])+'-different'
    with pytest.raises(ValueError,match=field):assert_reference_alignment(row,original)


def test_csv_numeric_strings_align_with_original_integer_fields():
    row=dict(dataset='IEMOCAP',dialogue_id='Ses05F_test',utterance_index=1,
             source_utterance_id='source',utterance_text='Text',true_label=2,true_emotion='neutral')
    original={k:str(v) for k,v in row.items()}
    assert_reference_alignment(row,original)

from scripts.summarize_m3net_verification import independent_metrics


def test_independent_f1_weights_true_support_and_handles_unpredicted_class():
    rows=[{'true_label':'0','predicted_label':'0'}]*3+[{'true_label':'1','predicted_label':'0'}]
    metrics,report,matrix=independent_metrics(rows,['first','second'])
    assert metrics['accuracy']==75
    assert metrics['weighted_f1']==pytest.approx(100*(6/7)*(3/4))
    assert metrics['macro_f1']==pytest.approx(100*(6/7)/2)
    assert report[1]['precision']==report[1]['recall']==report[1]['f1']==0
    assert matrix==[[3,0],[1,0]]


def test_confusion_matrix_orientation_is_true_rows_predicted_columns():
    rows=[{'true_label':'0','predicted_label':'1'}]*2+[{'true_label':'1','predicted_label':'1'}]
    metrics,report,matrix=independent_metrics(rows,['first','second'])
    assert matrix==[[0,2],[0,1]]
    assert report[1]['precision']==pytest.approx(100/3)
    assert report[1]['recall']==100
    assert metrics['weighted_f1']==pytest.approx(100/6)
