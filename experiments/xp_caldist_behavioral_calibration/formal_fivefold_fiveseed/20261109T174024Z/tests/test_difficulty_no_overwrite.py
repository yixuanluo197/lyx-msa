import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[5]
MODULE = ROOT / 'experiments/xp_caldist_behavioral_calibration/formal_fivefold_fiveseed/20261109T174024Z/scripts/generate_fold_local_difficulty.py'

def load():
    spec = importlib.util.spec_from_file_location('formal_difficulty_target', MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(MODULE.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod

def test_refuses_to_overwrite_either_frozen_output(tmp_path):
    mod = load()
    output = tmp_path / 'difficulty.json'
    probabilities = tmp_path / 'six_view.npz'
    output.write_text('frozen')
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        mod.refuse_existing_outputs(output, probabilities)
    output.unlink()
    probabilities.write_bytes(b'frozen')
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        mod.refuse_existing_outputs(output, probabilities)

def test_allows_new_outputs(tmp_path):
    mod = load()
    mod.refuse_existing_outputs(tmp_path / 'difficulty.json', tmp_path / 'six_view.npz')
