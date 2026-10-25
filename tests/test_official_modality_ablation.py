import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_official_modality_ablation import replace_call_inputs

@pytest.mark.parametrize('condition,target', [('w/o_text',0),('w/o_audio',4),('w/o_visual',5)])
def test_mean_includes_padding_without_modifying_other_inputs(condition,target):
    x=torch.tensor([[[2.,4.]],[[4.,8.]],[[0.,0.]]])
    text=[x.clone()+i for i in range(4)]
    call=(text,torch.ones(3,1,2),torch.tensor([[1.,1.,0.]]),[2],x.clone()*2,x.clone()*3,0)
    before=[t.clone() for t in text]+[call[4].clone(),call[5].clone()]
    result,audit=replace_call_inputs(call,condition)
    values=result[0] if target==0 else [result[target]]
    sources=call[0] if target==0 else [call[target]]
    for source,value in zip(sources,values):
        assert torch.equal(value,source.mean(0).repeat(3,1,1))
        assert not torch.equal(value[:2].mean(0),source[:2].mean(0))
    assert all(result[i] is call[i] for i in range(7) if i!=target)
    assert all(torch.equal(a,b) for a,b in zip(before,text+[call[4],call[5]]))
    assert audit['target_changed_elements']>0
    assert audit['padding_included_in_mean']

def test_reject_invalid_condition():
    with pytest.raises(ValueError): replace_call_inputs((), 'full')

from summarize_official_ablation import summarize_transitions

def test_new_hc_errors_exclude_existing_errors_and_use_full_correct_denominator():
    full=[{'correct':True},{'correct':True},{'correct':False}]
    masked=[{'correct':False,'predicted_confidence':.9},
            {'correct':False,'predicted_confidence':.79},
            {'correct':False,'predicted_confidence':.99}]
    rows=summarize_transitions(full,masked,'IEMOCAP','w/o_audio')
    assert [r['new_high_confidence_errors'] for r in rows]==[1,1,0]
    assert [r['ratio'] for r in rows]==[.5,.5,0]
    assert all(r['still_wrong_count']==1 for r in rows)


def test_zero_full_correct_denominator_is_missing():
    rows=summarize_transitions([{'correct':False}],
            [{'correct':True,'predicted_confidence':.95}],'MELD','w/o_text')
    assert all(r['ratio'] is None for r in rows)
