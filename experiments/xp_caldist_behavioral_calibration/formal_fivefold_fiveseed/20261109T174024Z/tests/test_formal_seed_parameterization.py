import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[5]
FORMAL = ROOT / 'experiments/xp_caldist_behavioral_calibration/formal_fivefold_fiveseed/20261109T174024Z/scripts/cmerc_runner.py'
PILOT = ROOT / 'experiments/xp_caldist_behavioral_calibration/independent_cmerc_pilot/20261109T164030Z/scripts/cmerc_runner.py'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod

def test_seed1475_preserves_pilot_configuration():
    pilot = load('pilot_seed_reference', PILOT)
    formal = load('formal_seed_target', FORMAL)
    for dataset in ('IEMOCAP', 'MELD'):
        assert formal.config_for(dataset, 1475) == pilot.config_for(dataset)

def test_only_seed_changes_across_preregistered_runs():
    formal = load('formal_seed_variants', FORMAL)
    base = formal.config_for('IEMOCAP', 1475)
    for seed in (67137, 2024, 3407, 42):
        candidate = formal.config_for('IEMOCAP', seed)
        assert candidate['seed'] == seed
        assert {k: v for k, v in candidate.items() if k != 'seed'} == {k: v for k, v in base.items() if k != 'seed'}

@pytest.mark.parametrize('seed', [0, -1, 1, 1476, 999999])
def test_rejects_non_preregistered_seed(seed):
    formal = load(f'formal_bad_seed_{seed}', FORMAL)
    with pytest.raises(ValueError, match='preregistered'):
        formal.config_for('MELD', seed)
