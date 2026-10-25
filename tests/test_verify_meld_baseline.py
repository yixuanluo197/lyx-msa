from pathlib import Path
import pytest
from scripts.verify_meld_baseline import FIXED_CHECKPOINT, fixed_checkpoint_for_request, state_compatibility


def test_fixed_request_maps_to_audited_local_meld_state(tmp_path):
    assert fixed_checkpoint_for_request(tmp_path/'checkpoints/MELD_checkpoint.pkl',tmp_path)==tmp_path/FIXED_CHECKPOINT


@pytest.mark.parametrize('wrong',['IEMOCAP_checkpoint.pkl','best.pt','other_MELD.pt'])
def test_reject_other_dataset_or_candidate(tmp_path,wrong):
    with pytest.raises(ValueError):fixed_checkpoint_for_request(tmp_path/'checkpoints'/wrong,tmp_path)


def test_scalar_and_shape_mismatch_are_not_silently_accepted():
    class Value:
        def __init__(self,shape):self.shape=shape
    result=state_compatibility({'scalar':Value(()),'a':Value((2,3))},{'scalar':Value(()),'a':Value((2,4))})
    assert not result['missing_keys'] and not result['unexpected_keys']
    assert result['shape_mismatches']==[dict(key='a',expected=[2,3],checkpoint=[2,4])]
