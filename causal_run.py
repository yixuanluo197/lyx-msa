"""Immutable run identity and pre-load resume validation (no experiment execution)."""
import hashlib
import json
from pathlib import Path

import torch


# These controls may change without changing the completed training prefix.
_RUNTIME_KEYS = {'epochs', 'output_dir', 'resume', 'num_workers', 'cuda', 'no_cuda',
                 'testing', 'tensorboard', 'run_manifest', 'run_dir'}


def _values(args):
    return dict(args) if isinstance(args, dict) else vars(args).copy()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_config(args):
    values = _values(args)
    # Artifact identity is content based; moving an immutable input is allowed.
    artifact = values.get('difficulty_artifact')
    if artifact:
        values['difficulty_artifact'] = {'sha256': artifact['sha256']}
    values.pop('difficulty_path', None)
    return {key: value for key, value in values.items() if key not in _RUNTIME_KEYS}


def run_config_digest(args):
    return hashlib.sha256(json.dumps(scientific_config(args), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def _set(args, key, value):
    if isinstance(args, dict):
        args[key] = value
    else:
        setattr(args, key, value)


def _historical_best(checkpoint, source):
    best_epoch = checkpoint.get('best_epoch')
    if best_epoch is None:
        return None
    reference = checkpoint.get('best_checkpoint')
    candidate = (source if source.name == 'best.pt' or (source.name != 'latest.pt' and not reference and checkpoint['epoch'] == best_epoch)
                 else source.parent / 'best.pt')
    if not candidate.is_file():
        raise ValueError('historical best checkpoint is missing: {}'.format(candidate))
    if reference and reference.get('sha256') != sha256(candidate):
        raise ValueError('historical best checkpoint hash mismatch')
    best = torch.load(candidate, map_location='cpu')
    if best.get('epoch') != best_epoch or best.get('best_metric') != checkpoint.get('best_metric'):
        raise ValueError('historical best checkpoint does not match saved best epoch/metric')
    return candidate


def validate_epoch_budget(args, checkpoint):
    values = _values(args)
    requested = values.get('epochs')
    if requested is None:
        return
    if not isinstance(requested, int) or requested <= checkpoint['epoch'] + 1:
        raise ValueError('resume epoch budget must include at least one new epoch')
    saved = checkpoint.get('args', {}).get('epochs')
    if saved is not None and requested < saved:
        raise ValueError('resume may extend the epoch budget but cannot shorten it')


def prepare_run(args):
    """Create a fresh config-named run or validate an existing manifest before resume.

    Resuming an older best checkpoint creates a new branch, preserving later history.
    Epoch extension is the only scientific training control excluded from identity.
    """
    values = _values(args)
    digest = run_config_digest(args)
    manifest = {'schema_version': 1, 'config_sha256': digest,
                'config': scientific_config(args)}
    resume = values.get('resume')
    root = Path(values.get('output_dir', 'runs/train_cm')).resolve()
    name = '{}_{}_seed{}'.format(values.get('Dataset', 'unknown'),
            values.get('selection_protocol', 'unknown'), values.get('seed'))
    if values.get('causal'):
        name += '_causal_' + digest[:16]
    run = root / name
    historical_best = None
    created = False
    if resume:
        source = Path(resume).resolve()
        manifest_path = source.parent / 'run_manifest.json'
        if not manifest_path.is_file():
            raise ValueError('resume requires an existing run manifest')
        saved_manifest = json.loads(manifest_path.read_text())
        if saved_manifest.get('config_sha256') != digest:
            raise ValueError('resume configuration/data/difficulty does not match run manifest')
        checkpoint = torch.load(source, map_location='cpu')
        if checkpoint.get('run_manifest_sha256') != sha256(manifest_path):
            raise ValueError('checkpoint run manifest hash mismatch')
        validate_epoch_budget(args, checkpoint)
        historical_best = _historical_best(checkpoint, source)
        metrics = source.parent / 'metrics.jsonl'
        if not metrics.is_file():
            raise ValueError('resume metrics history is missing')
        try:
            history = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
            recorded_epochs = [record.get('epoch') for record in history]
        except (ValueError, AttributeError) as error:
            raise ValueError('resume metrics history is malformed') from error
        if (not history or recorded_epochs != list(range(1, len(history) + 1)) or
                len(history) < checkpoint['epoch'] + 1):
            raise ValueError('resume metrics history must contain each completed epoch exactly once')
        history_prefix = history[:checkpoint['epoch'] + 1]
        if source.name == 'latest.pt' and source.parent.parent == root:
            if len(history) != checkpoint['epoch'] + 1:
                raise ValueError('resume metrics history extends beyond checkpoint epoch')
            run = source.parent
        else:
            # A best/older checkpoint is a new branch rather than a log rollback.
            run = root / (name + '_resume_' + sha256(source)[:12])
            manifest['parent_checkpoint'] = {'path': str(source), 'sha256': sha256(source),
                                              'epoch': checkpoint['epoch']}
    if not resume or run != Path(resume).resolve().parent:
        run.mkdir(parents=True, exist_ok=False)
        created = True
        (run / 'run_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        if resume:
            with (run / 'metrics.jsonl').open('x') as handle:
                for record in history_prefix:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
    _set(args, 'run_dir', str(run))
    _set(args, 'run_manifest', {'path': str(run / 'run_manifest.json'),
                              'sha256': sha256(run / 'run_manifest.json'),
                              'config_sha256': digest})
    if created and historical_best:
        # Bind an imported best to this branch while preserving its complete
        # training state. Keep explicit ancestry instead of copying stale run
        # metadata that would prevent a later best-checkpoint resume.
        imported = torch.load(historical_best, map_location='cpu')
        ancestry = list(imported.get('imported_checkpoint_lineage', []))
        ancestry.append({'path': str(historical_best.resolve()),
                         'sha256': sha256(historical_best),
                         'source_run_manifest_sha256': imported.get('run_manifest_sha256')})
        imported['imported_checkpoint_lineage'] = ancestry
        imported['args'] = _values(args)
        imported['run_manifest_sha256'] = _values(args)['run_manifest']['sha256']
        imported.pop('best_checkpoint', None)
        torch.save(imported, run / 'best.pt')
    return run
