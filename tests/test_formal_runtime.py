"""Focused runtime safeguards for formal paired runs; no formal training."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_cm
from causal_run import run_config_digest
from test_followup_training import batches, config, models


@pytest.mark.parametrize('defer', [True, False])
def test_deferred_test_skips_actual_final_test_block(tmp_path, defer):
    source = ast.parse(Path(train_cm.__file__).read_text())
    main = next(node for node in source.body if isinstance(node, ast.If)
                and ast.unparse(node.test) == "__name__ == '__main__'")
    block = main.body[-1]
    assert isinstance(block, ast.If)
    calls = []
    class ForbiddenTorch:
        def load(self, *args, **kwargs):
            calls.append('load')
            raise AssertionError('deferred final test must not load/test the selected checkpoint')
    checkpoint = tmp_path / 'best.pt'
    checkpoint.touch()
    namespace = dict(args=SimpleNamespace(selection_protocol='validation', defer_final_test=defer),
                     best_checkpoint_path=checkpoint, torch=ForbiddenTorch())
    code = compile(ast.Module(body=[block], type_ignores=[]), '<final-test-guard>', 'exec')
    if defer:
        exec(code, namespace)
        assert calls == []
    else:
        with pytest.raises(AssertionError, match='deferred final test'):
            exec(code, namespace)
        assert calls == ['load']


def test_initial_fingerprint_is_stable_content_hash_and_changes_on_update():
    torch.manual_seed(19)
    model = torch.nn.Linear(3, 2)
    original = train_cm.model_state_sha256(model)
    assert len(original) == 64
    assert original == train_cm.model_state_sha256(copy.deepcopy(model))
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert original != train_cm.model_state_sha256(model)


def test_runtime_flags_do_not_change_training_resume_identity():
    original = run_config_digest(config())
    assert original == run_config_digest(config(defer_final_test=True, diagnostic_batches=1,
                                                initial_model_sha256='a' * 64,
                                                fail_on_nonfinite=True))


def test_selected_diagnostics_leave_training_parameters_unchanged(monkeypatch):
    train_cm.seed_everything(25)
    monkeypatch.setattr(train_cm, 'args', config(calibrate=True, contrastlearning=True), raising=False)
    initial = models(dropout=.15)
    data = batches()
    plain = copy.deepcopy(initial)
    instrumented = copy.deepcopy(initial)
    records = []
    for group, callback, limit in ((plain, None, None), (instrumented, records.append, 1)):
        optimizer = torch.optim.Adam(group[0].parameters(), lr=.001)
        train_cm.train_or_eval_graph_model(*group, torch.nn.NLLLoss(), data, 0, False,
            'avl', optimizer, True, diagnostic_callback=callback, diagnostic_batches=limit)
    assert [record['batch'] for record in records] == [0]
    assert records[0]['finite_gradients']
    assert records[0]['finite_losses']
    assert records[0]['weighted_gradient_norms']['classification_loss'] > 0
    assert any(value > 0 for value in records[0]['parameter_update_norms'].values())
    for name, value in plain[0].state_dict().items():
        torch.testing.assert_close(value, instrumented[0].state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('bad_component', ['loss', 'gradient'])
def test_nonfinite_guard_rejects_before_optimizer_update(monkeypatch, bad_component):
    class NonfiniteGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return value.new_tensor(1.)

        @staticmethod
        def backward(ctx, gradient):
            value, = ctx.saved_tensors
            return torch.full_like(value, float('nan'))

    def loss(log_prob, label):
        if bad_component == 'loss':
            return log_prob.sum() * float('nan')
        return NonfiniteGradient.apply(log_prob)

    train_cm.seed_everything(25)
    monkeypatch.setattr(train_cm, 'args', config(fail_on_nonfinite=True), raising=False)
    group = models()
    initial = {name: value.clone() for name, value in group[0].state_dict().items()}
    optimizer = torch.optim.Adam(group[0].parameters(), lr=.001)
    with pytest.raises(FloatingPointError, match=bad_component):
        train_cm.train_or_eval_graph_model(*group, loss, batches(count=1), 0, False,
            'avl', optimizer, True)
    assert optimizer.state == {}
    for name, value in initial.items():
        torch.testing.assert_close(value, group[0].state_dict()[name], rtol=0, atol=0)
