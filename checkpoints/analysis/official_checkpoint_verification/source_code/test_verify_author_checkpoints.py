import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    from scripts import verify_author_checkpoints
    return verify_author_checkpoints


def test_shape_audit_accepts_complete_state_including_scalar():
    state = {'w': SimpleNamespace(shape=(2, 3)), 'counter': SimpleNamespace(shape=())}
    assert module().state_compatibility(state, state) == {
        'missing_keys': [], 'unexpected_keys': [], 'shape_mismatches': []}


def test_shape_audit_reports_all_incompatibilities():
    expected = {'w': SimpleNamespace(shape=(2, 3)), 'bias': SimpleNamespace(shape=(2,))}
    loaded = {'w': SimpleNamespace(shape=(3, 2)), 'extra': SimpleNamespace(shape=())}
    result = module().state_compatibility(expected, loaded)
    assert result['missing_keys'] == ['bias']
    assert result['unexpected_keys'] == ['extra']
    assert result['shape_mismatches'] == [
        {'key': 'w', 'expected': [2, 3], 'checkpoint': [3, 2]}]


def test_observer_line_is_after_label_trim_and_before_loss_in_official_function():
    path = Path(__file__).resolve().parents[1] / 'experiments/official_readme_reproduction/official_code/train.py'
    line = module().observer_line(path)
    source = path.read_text().splitlines()
    assert source[line - 1].strip() == 'loss = loss_function(log_prob, label)'
    assert source[line - 2].strip().startswith('label = torch.cat(')


def test_observer_rejects_unknown_source_instead_of_guessing(tmp_path):
    path = tmp_path / 'train.py'
    path.write_text('def train_or_eval_graph_model():\n    return None\n')
    with pytest.raises(ValueError, match='observer'):
        module().observer_line(path)
