import importlib.util
from pathlib import Path
import pytest
import numpy as np

SPEC = importlib.util.spec_from_file_location('finalize_formal', Path(__file__).parents[1]/'scripts/finalize_formal.py')
m = importlib.util.module_from_spec(SPEC)
# Implementation is imported inside each test so the initial red run is a clear missing implementation.
def load():
    SPEC.loader.exec_module(m)
    return m

def test_resolve_ledger_never_accepts_process_exit_without_completed_state():
    module=load()
    with pytest.raises(ValueError, match='incomplete'):
        module.resolve_output({'status':'failed','attempts':[{'returncode':0,'directory':'bad'}]})

def test_calibration_mask_excludes_unknown_labels_and_checks_full_argmax():
    module=load()
    p=np.array([[.8,.2],[.3,.7],[.55,.45]])
    rows={'supervision_mask':np.array([1,0,1]),'labels':np.array([0,-999,1]),'E0_probabilities':p}
    for k in ('E0','E1','E2','E3'):
        rows[k+'_probabilities']=p.copy();rows[k+'_oof_probabilities']=p.copy();rows[k+'_temperatures']=np.ones(3)
    for k in ('E0','E1','E2','E3'):rows[k+'_oof_probabilities'][1]=np.nan
    result=module.audit_calibration_arrays(rows)
    assert result['E0']['n']==2
    rows['E1_probabilities'][1]=[.9,.1]
    with pytest.raises(ValueError,match='argmax'):
        module.audit_calibration_arrays(rows)

def test_difficulty_formula_and_exact_ids():
    module=load()
    row={'dialogue_id':'a','valid_label_count':2,'total_length':3,'speaker_term':2.,'total_gap':2.,'final_difficulty':1.}
    for k in ('text','audio','visual','intra','inter'):row[k+'_gap']=.4
    assert module.audit_difficulty({'dialogues':[row]}, {'a'})['count']==1
    row['final_difficulty']=.8
    with pytest.raises(ValueError,match='formula'):module.audit_difficulty({'dialogues':[row]},{'a'})

def test_evaluation_metadata_coverage_requires_once_per_fold_pool():
    module=load()
    roles=lambda a,b:{'training':{'dialogue_ids':[],'group_ids':[]},'development':{'dialogue_ids':[],'group_ids':[]},'calibration':{'dialogue_ids':[b],'group_ids':[b]},'evaluation':{'dialogue_ids':[a],'group_ids':[a]}}
    manifest={'protocols':{'IEMOCAP':{'folds':{'fold1':{'roles':roles('a','b')},'fold2':{'roles':roles('b','a')}}}}}
    report=module.audit_manifest_metadata(manifest,expected_folds=2)
    assert report['IEMOCAP']['evaluation_dialogues']==2
    manifest['protocols']['IEMOCAP']['folds']['fold2']['roles']['evaluation']['dialogue_ids']=['c']
    with pytest.raises(ValueError,match='coverage'):module.audit_manifest_metadata(manifest,expected_folds=2)
