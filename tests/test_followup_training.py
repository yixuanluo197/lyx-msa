"""Real-function integration and run/provenance regression checks; no experiments."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import train_cm
from test_causal_model import make_model, inputs
from model_cm import Model_Intra, Model_Inter


def config(**changes):
    values = dict(causal=True, norm='LN', Dataset='IEMOCAP', base_model='GRU',
                  num_L=2, num_K=2, av_using_lstm=True, use_residue=False,
                  seed=25, split_seed=12, dropout=0., rec_dropout=0., batch_size=2,
                  lr=.001, l2=0., calibrate=False, contrastlearning=False,
                  rank_coff=.005, mscl_coff=.05, cscl_coff=.05,
                  courselearning=False, epoch_ratio=.5, scheduler_steps=1,
                  selection_protocol='validation', multi_modal=True,
                  mm_fusion_mthd='concat_DHT', modals='avl', tensorboard=False,
                  auxiliary_policy='legacy_epoch_snapshot', replacement_policy='prefix_mean_v1')
    values.update(changes)
    return SimpleNamespace(**values)


def batches(count=2, device='cpu'):
    result = []
    for i in range(count):
        text, speaker, mask, lengths, audio, visual = inputs((3, 2))
        labels = torch.tensor([[0, 1, 2], [1, 3, 0]])
        result.append([x.to(device) for x in (*text, visual, audio, speaker, mask, labels)] +
                      [[f'synthetic-{i}-a', f'synthetic-{i}-b']])
    return result


def models(dropout=0., device='cpu'):
    return tuple(make_model(kind=kind, dropout=dropout).to(device)
                 for kind in (train_cm.Model, train_cm.Model, Model_Intra, Model_Inter))


def test_existing_run_refused_and_config_id_separates_ablations(tmp_path):
    from causal_run import prepare_run, run_config_digest
    args = config(output_dir=str(tmp_path), resume=None)
    paths = [run_config_digest(config(**change)) for change in
             ({}, {'calibrate': True}, {'contrastlearning': True},
              {'calibrate': True, 'contrastlearning': True, 'courselearning': True},
              {'courselearning': True}, {'num_L': 3}, {'num_K': 3},
              {'use_residue': True}, {'seed': 26})]
    assert len(set(paths)) == len(paths)
    run = prepare_run(args)
    (run / 'metrics.jsonl').write_text('existing\n')
    with pytest.raises(FileExistsError):
        prepare_run(args)
    assert (run / 'metrics.jsonl').read_text() == 'existing\n'


def test_resume_pins_difficulty_before_loading_parameters(tmp_path):
    a = config(difficulty_artifact={'sha256': 'a'*64})
    full, *_ = models()
    opt = torch.optim.Adam(full.parameters())
    path = tmp_path / 'test.pt'
    train_cm.save_training_checkpoint(path, full, opt, 0, a, None, None)
    b = config(difficulty_artifact={'sha256': 'b'*64})
    with pytest.raises(ValueError, match='difficulty'):
        train_cm.load_training_checkpoint(path, full, opt, 'cpu', b)


@pytest.mark.parametrize('cc,hscl', [(False, False), (True, False), (False, True), (True, True)])
def test_real_training_two_steps_and_read_only_auxiliaries(monkeypatch, cc, hscl):
    train_cm.seed_everything(25)
    monkeypatch.setattr(train_cm, 'args', config(calibrate=cc, contrastlearning=hscl), raising=False)
    full, mm, intra, inter = models(dropout=.15)
    opt = torch.optim.Adam(full.parameters(), lr=.001)
    before = full.gru_l.weight_ih_l0.detach().clone()
    records = []
    result = train_cm.train_or_eval_graph_model(full, mm, intra, inter,
        torch.nn.NLLLoss(), batches(), 0, False, 'avl', opt, True,
        return_components=True, diagnostic_callback=records.append)
    assert len(records) == 2
    assert records[0]['main_aux_max_abs_diff'] == 0.
    assert records[1]['main_aux_max_abs_diff'] > 0.
    assert records[0]['main_training'] and not records[0]['aux_training']
    assert not torch.equal(before, full.gru_l.weight_ih_l0)
    assert all(p.grad is None for m in (mm, intra, inter) for p in m.parameters())
    assert all(torch.isfinite(torch.tensor(x)) for x in result[-1].values())
    assert records[0]['gradient_norms']['classification_loss'] > 0.
    if cc or hscl:
        assert len(records[0]['hscl_labels']) == 5
        assert len(records[0]['cc_per_view']) == 5


def test_resume_run_requires_available_matching_historical_best(tmp_path):
    from causal_run import prepare_run
    a = config(output_dir=str(tmp_path), resume=None)
    run = prepare_run(a)
    full, *_ = models()
    opt = torch.optim.Adam(full.parameters())
    path = run / 'latest.pt'
    train_cm.save_training_checkpoint(path, full, opt, 1, a, .7, 0)
    with pytest.raises(ValueError, match='best'):
        prepare_run(config(output_dir=str(tmp_path), resume=str(path)))


def test_latest_resume_and_best_resume_preserve_history(tmp_path):
    from causal_run import prepare_run
    a = config(output_dir=str(tmp_path), resume=None)
    run = prepare_run(a)
    full, *_ = models()
    opt = torch.optim.Adam(full.parameters())
    train_cm.save_training_checkpoint(run/'best.pt', full, opt, 0, a, .7, 0)
    train_cm.save_training_checkpoint(run/'latest.pt', full, opt, 1, a, .7, 0)
    (run/'metrics.jsonl').write_text('{"epoch":1}\n{"epoch":2}\n')
    original = (run/'metrics.jsonl').read_bytes()
    latest = copy.deepcopy(a); latest.resume = str(run/'latest.pt'); latest.epochs=99
    assert prepare_run(latest) == run
    best = copy.deepcopy(a); best.resume = str(run/'best.pt')
    branch = prepare_run(best)
    assert branch != run and (branch/'best.pt').is_file()
    assert (run/'metrics.jsonl').read_bytes() == original
    assert json.loads((branch/'run_manifest.json').read_text())['parent_checkpoint']['epoch'] == 0


def test_resume_rejects_replaced_best_and_metrics_ahead(tmp_path):
    from causal_run import prepare_run
    a = config(output_dir=str(tmp_path), resume=None)
    run = prepare_run(a)
    full, *_ = models()
    opt = torch.optim.Adam(full.parameters())
    train_cm.save_training_checkpoint(run/'best.pt', full, opt, 0, a, .7, 0)
    train_cm.save_training_checkpoint(run/'latest.pt', full, opt, 1, a, .7, 0)
    resume = copy.deepcopy(a); resume.resume = str(run/'latest.pt')
    (run/'metrics.jsonl').write_text('{"epoch":3}\n')
    with pytest.raises(ValueError, match='metrics'):
        prepare_run(resume)
    (run/'metrics.jsonl').write_text('{"epoch":2}\n')
    with (run/'best.pt').open('ab') as handle:
        handle.write(b'replaced')
    with pytest.raises(ValueError, match='best checkpoint hash'):
        prepare_run(resume)


def test_same_step_policy_synchronizes_before_each_batch(monkeypatch):
    train_cm.seed_everything(25)
    monkeypatch.setattr(train_cm, 'args', config(calibrate=True, contrastlearning=True,
        auxiliary_policy='same_step_same_mode'), raising=False)
    full, mm, intra, inter = models(dropout=.15)
    records = []
    train_cm.train_or_eval_graph_model(full, mm, intra, inter,
        torch.nn.NLLLoss(), batches(), 0, False, 'avl',
        torch.optim.Adam(full.parameters(), lr=.001), True,
        diagnostic_callback=records.append)
    assert [r['main_aux_max_abs_diff'] for r in records] == [0., 0.]
    assert all(r['main_training'] and not r['aux_training'] for r in records)
    assert all(p.grad is None for m in (mm,intra,inter) for p in m.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_gpu_same_step_eval_confidence_has_backward_graph(monkeypatch):
    train_cm.seed_everything(25)
    monkeypatch.setattr(train_cm, 'args', config(calibrate=True, contrastlearning=True,
        auxiliary_policy='same_step_same_mode'), raising=False)
    full, mm, intra, inter = models(dropout=.15, device='cuda:0')
    records = []
    train_cm.train_or_eval_graph_model(full, mm, intra, inter,
        torch.nn.NLLLoss(), batches(device='cuda:0'), 0, True, 'avl',
        torch.optim.Adam(full.parameters(), lr=.001), True,
        diagnostic_callback=records.append)
    assert all(r['finite_gradients'] for r in records)
    assert records[0]['gradient_norms']['classification_loss'] > 0


def test_real_training_checkpoint_resume_is_exact(tmp_path):
    from scripts.verify_training_integration import recovery_check, synthetic_batches
    train_cm.seed_everything(25)
    result = recovery_check(torch.device('cpu'), synthetic_batches('cpu'), tmp_path)
    assert result['weight_max_abs_diff'] == 0.
    assert result['optimizer_max_abs_diff'] == 0.
    assert result['next_step_loss_equal'] and result['logs_equal']


@pytest.mark.parametrize('history', [None, '', '{"epoch":1}\n{"epoch":3}\n',
                                     '{"epoch":1}\n{"epoch":1}\n{"epoch":2}\n'])
def test_latest_resume_rejects_missing_empty_or_noncontiguous_metrics(tmp_path, history):
    from causal_run import prepare_run
    a = config(output_dir=str(tmp_path), resume=None, epochs=4)
    run = prepare_run(a)
    full, *_ = models(); opt = torch.optim.Adam(full.parameters())
    train_cm.save_training_checkpoint(run/'best.pt', full, opt, 0, a, .7, 0)
    train_cm.save_training_checkpoint(run/'latest.pt', full, opt, 1, a, .7, 0)
    if history is not None:
        (run/'metrics.jsonl').write_text(history)
    a.resume = str(run/'latest.pt')
    with pytest.raises(ValueError, match='metrics'):
        prepare_run(a)


@pytest.mark.parametrize('requested_epochs', [1, 2, 3])
def test_resume_rejects_empty_loop_or_shortened_epoch_budget(tmp_path, requested_epochs):
    from causal_run import prepare_run
    a = config(output_dir=str(tmp_path), resume=None, epochs=4)
    run = prepare_run(a)
    full, *_ = models(); opt = torch.optim.Adam(full.parameters())
    train_cm.save_training_checkpoint(run/'best.pt', full, opt, 0, a, .7, 0)
    train_cm.save_training_checkpoint(run/'latest.pt', full, opt, 1, a, .7, 0)
    (run/'metrics.jsonl').write_text('{"epoch":1}\n{"epoch":2}\n')
    a.resume = str(run/'latest.pt'); a.epochs=requested_epochs
    with pytest.raises(ValueError, match='epoch'):
        prepare_run(a)


def test_repeated_best_resume_rebinds_imported_best_without_changing_training_state(tmp_path):
    from causal_run import prepare_run
    a = config(output_dir=str(tmp_path), resume=None, epochs=4)
    run = prepare_run(a)
    full, *_ = models(); opt = torch.optim.Adam(full.parameters())
    train_cm.save_training_checkpoint(run/'best.pt', full, opt, 0, a, .7, 0)
    train_cm.save_training_checkpoint(run/'latest.pt', full, opt, 1, a, .7, 0)
    (run/'metrics.jsonl').write_text('{"epoch":1}\n{"epoch":2}\n')
    original = torch.load(run/'best.pt', map_location='cpu')
    first = copy.deepcopy(a); first.resume=str(run/'best.pt')
    branch = prepare_run(first)
    imported = torch.load(branch/'best.pt', map_location='cpu')
    assert imported['run_manifest_sha256'] == first.run_manifest['sha256']
    for name, value in original['model_state_dict'].items():
        torch.testing.assert_close(value, imported['model_state_dict'][name], atol=0, rtol=0)
    assert original['optimizer_state_dict'] == imported['optimizer_state_dict']
    second = copy.deepcopy(first); second.resume=str(branch/'best.pt')
    second_branch = prepare_run(second)
    assert second_branch != branch
    second_imported = torch.load(second_branch/'best.pt', map_location='cpu')
    assert len(second_imported['imported_checkpoint_lineage']) == 2
    assert second_imported['best_epoch'] == 0 and second_imported['best_metric'] == .7
    train_cm.load_training_checkpoint(second_branch/'best.pt', full, opt, 'cpu', second)


def test_noncausal_run_keeps_legacy_automation_path_and_exclusive_create(tmp_path):
    from causal_run import prepare_run
    a = config(causal=False, output_dir=str(tmp_path), resume=None)
    assert prepare_run(a) == tmp_path/'IEMOCAP_validation_seed25'
    with pytest.raises(FileExistsError):
        prepare_run(a)


@pytest.mark.parametrize('tamper', ['manifest', 'difficulty'])
def test_low_level_resume_rechecks_files_before_mutating_model(tmp_path, tamper):
    from causal_run import prepare_run, sha256
    artifact = tmp_path/'difficulty.fixture'; artifact.write_bytes(b'synthetic flow fixture')
    a = config(output_dir=str(tmp_path/'runs'), resume=None,
        difficulty_artifact={'path':str(artifact), 'sha256':sha256(artifact)})
    run = prepare_run(a)
    full, *_ = models(); opt = torch.optim.Adam(full.parameters())
    train_cm.save_training_checkpoint(run/'best.pt', full, opt, 0, a, .7, 0)
    if tamper == 'manifest':
        Path(a.run_manifest['path']).write_text('{}')
    else:
        artifact.write_bytes(b'replaced fixture')
    fresh, *_ = models(); fresh_opt=torch.optim.Adam(fresh.parameters())
    before = {key:value.clone() for key,value in fresh.state_dict().items()}
    with pytest.raises(ValueError, match='manifest|difficulty'):
        train_cm.load_training_checkpoint(run/'best.pt', fresh, fresh_opt, 'cpu', a)
    for key,value in fresh.state_dict().items():
        torch.testing.assert_close(value,before[key],atol=0,rtol=0)
