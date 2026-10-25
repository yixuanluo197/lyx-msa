"""Read-only CPU audit of the preselected trusted local checkpoint; no inference."""
import ast
import hashlib
import json
import pathlib
from datetime import datetime, timezone

import torch

ROOT = pathlib.Path(__file__).resolve().parents[6]
OUT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'experiments/official_readme_reproduction'
CHECKPOINT = SOURCE / 'checkpoints/MELD_baseline_seed67137/best.pt'
EXPECTED = 'a6ab85cb6c0c12054dade9c6bcdf01cfaa6abc417625059a927de5ba386d3738'
EXPORT = ROOT / 'checkpoints/baseline_candidates/meld_readme_fixed_20261025/model_state.pt'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def state_fingerprint(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode() + b'\0')
        h.update(str(value.dtype).encode() + b'\0')
        h.update(json.dumps(list(value.shape)).encode() + b'\0')
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

assert sha(CHECKPOINT) == EXPECTED, 'Refuse loading unknown or changed checkpoint'
try:
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    safe_error = None
except Exception as exc:
    safe_error = {'type': type(exc).__name__, 'message': str(exc)}
    # This exact hash is documented in the previous trusted local checkpoint inventory.
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
state = payload['model']
reference_path = ROOT / 'checkpoints/MELD_checkpoint.pkl'
reference = torch.load(reference_path, map_location='cpu', weights_only=True)
missing, extra = sorted(set(reference) - set(state)), sorted(set(state) - set(reference))
shapes = {k: [list(state[k].shape), list(reference[k].shape)] for k in set(reference) & set(state) if state[k].shape != reference[k].shape}
dtype_mismatches = {k: [str(state[k].dtype), str(reference[k].dtype)] for k in set(reference) & set(state) if state[k].dtype != reference[k].dtype}
assert not missing and not extra and not shapes and not dtype_mismatches
args = payload['args']
assert all(args[k] is False for k in ('calibrate', 'contrastlearning', 'courselearning'))
assert payload['epoch'] == 5 and payload['selection_protocol'] == 'official_test_selection'
assert args['Dataset'] == 'MELD' and args['batch_size'] == 16
assert args['dropout'] == .4 and args['base_model'] == 'GRU'
assert args['num_L'] == args['num_K'] == 3
# Freeze a tensor-only input for subsequent safe GPU evaluation. Do not overwrite it.
if not EXPORT.exists():
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, EXPORT)
exported = torch.load(EXPORT, map_location='cpu', weights_only=True)
assert state_fingerprint(state) == state_fingerprint(exported)
inputs = json.loads((SOURCE / 'provenance/inputs.json').read_text())
data = {}
for name in ('MELD_features_raw1.pkl', 'meld_features_roberta.pkl'):
    p = ROOT / 'data' / name
    observed = sha(p)
    assert observed == inputs['data'][name]['sha256']
    data[name] = {**inputs['data'][name], 'current_path': str(p.resolve()), 'current_sha256': observed}
code = {}
for f in ('model.py', 'model_cm.py', 'model_hyper.py', 'HypergraphConv.py', 'dataloader.py', 'train.py'):
    p = SOURCE / 'runtime_code' / f
    q = ROOT / 'checkpoints/analysis/official_checkpoint_verification/code_runtime' / f
    assert sha(p) == sha(q)
    code[f] = {'sha256': sha(p), 'identical_to_verified_cmerc_runtime': True}
manifest = {
    'schema_version': 1, 'audited_at': datetime.now(timezone.utc).isoformat(),
    'decision': 'Preselected first candidate technically compatible; do not evaluate alternatives to select by test score.',
    'selection_fixed_before_new_inference': True,
    'identity': 'Local M3Net baseline reproduction using CMERC official README training command; not an author-released MELD baseline.',
    'checkpoint': {'path': str(CHECKPOINT), 'sha256': EXPECTED, 'bytes': CHECKPOINT.stat().st_size, 'internal_keys': list(payload), 'model_state_key': 'model', 'epoch_1based': payload['epoch'], 'selection_protocol': payload['selection_protocol'], 'args': args},
    'safe_loading': {'weights_only_initial_attempt': True, 'weights_only_error': safe_error, 'fallback': 'Only after exact prior trusted local SHA verification: torch.load(map_location=cpu, weights_only=False).', 'unknown_remote_pickle_loaded': False},
    'state': {'tensor_count': len(state), 'keys_missing': missing, 'keys_extra': extra, 'shape_mismatches': shapes, 'dtype_mismatches': dtype_mismatches, 'tensor_content_fingerprint_sha256': state_fingerprint(state), 'fingerprint_protocol': 'sorted key + NUL + dtype + NUL + JSON shape + NUL + contiguous CPU numpy bytes', 'inventory': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in state.items()}},
    'export': {'path': str(EXPORT), 'sha256': sha(EXPORT), 'bytes': EXPORT.stat().st_size, 'tensor_content_fingerprint_sha256': state_fingerprint(exported), 'safe_weights_only_load_passed': True, 'source_state_identical': True, 'weight_is_not_a_report_artifact': True},
    'official_cmerc_reference': {'path': str(reference_path), 'sha256': sha(reference_path), 'tensor_count': len(reference), 'training_seed_epoch_selection': 'unknown'},
    'data': data, 'runtime_code': code,
    'training_server': {'server': 'local', 'host': 'user23', 'gpu': 'TITAN Xp', 'uuid': 'GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456'},
    'training_backend': {'torch': '1.13.1+cu117', 'cuda': '11.7', 'cudnn_version': 8500, 'implementation': 'Official nn.GRU, bidirectional, default cuDNN path; benchmark=False, deterministic=True; no ATen override in frozen entry point or launcher.', 'limitation': 'Historical environment and frozen-source evidence; checkpoint does not embed a per-forward dispatched-kernel trace.'},
    'causal': {'argument_present': 'causal' in args, 'conclusion': 'Official noncausal/bidirectional full-context model; no local causal variant loaded.', 'evidence': 'Frozen model_cm.py nn.GRU(... bidirectional=True), official model and hypergraph source hashes; Model.__init__ AST identical to official inference model.py.'},
    'prospective_evaluation': {'server': 'biggpu', 'gpu': 'healthy V100, explicitly excluding physical GPU 4', 'reason': 'User explicitly requests the verified official CMERC test protocol/environment.', 'strict_load_required': True, 'batch_size': 16, 'num_workers': 2, 'full_tav_only': True, 'cudnn_enabled': True, 'rnn_backend_matches_official_cmerc': True, 'no_training': True, 'no_modality_deletion': True, 'label_order': ['neutral','surprise','fear','sadness','joy','disgust','anger']},
    'risks': ['Historical checkpoint was selected by first maximum rounded TEST weighted-F1 over 40 epochs, selected epoch 5: test-selection leakage; no independent validation selection.', 'Author CMERC training seed/epoch/model selection and correspondence to this local baseline are unknown.', 'Training was on local XP; this requested standardized inference is on biggpu V100. Do not conflate training server and evaluation server.', 'Descriptive fixed-weight behavior comparison may be feasible after inference alignment; strict training pairing and module causal attribution are not established.'],
    'candidate_policy': ['User-prioritized official_readme_reproduction MELD baseline (selected).', 'runs/formal/baseline/MELD_official_test_seed67137/best.pt only if first technically incompatible.', 'runs/formal/baseline/MELD_validation_seed67137/best.pt only if preceding incompatible.', 'xp_mainline_validation MELD baseline only if preceding incompatible.'],
    'alternatives_checked_without_inference': [], 'gpu_inference_in_this_audit': False,
    'source_checkpoint_sha256_after': sha(CHECKPOINT), 'source_unchanged': sha(CHECKPOINT) == EXPECTED,
}
for p in [ROOT/'runs/formal/baseline/MELD_official_test_seed67137/best.pt', ROOT/'runs/formal/baseline/MELD_validation_seed67137/best.pt', *ROOT.glob('experiments/xp_mainline_validation/*/runs/baseline/MELD_validation_seed67137/best.pt')]:
    manifest['alternatives_checked_without_inference'].append({'path': str(p), 'exists': p.exists(), 'sha256': sha(p) if p.exists() else None, 'selected': False, 'reason': 'First candidate already compatible; no test-score comparison or inference.'})
(OUT/'baseline_checkpoint_audit.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
print(json.dumps({'technical_compatibility': True,'tensor_count':len(state),'source_sha256':EXPECTED,'export_path':str(EXPORT),'export_sha256':sha(EXPORT),'tensor_fingerprint':state_fingerprint(state)},indent=2))
