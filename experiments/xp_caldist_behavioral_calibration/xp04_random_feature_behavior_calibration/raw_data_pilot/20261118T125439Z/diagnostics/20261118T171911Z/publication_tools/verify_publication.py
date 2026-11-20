"""Static public evidence verification: no training, inference or private data access."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-source', type=Path)
    args = parser.parse_args()
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'pass': bool(passed)})
        if not passed:
            raise AssertionError(name)

    manifest = json.loads((ROOT / 'publication_manifest.json').read_text())
    entries = manifest['files'] + manifest['generated_files']
    for row in entries:
        path = ROOT / row['file']
        check('public_hash:' + row['file'], path.is_file() and sha(path) == row['public_sha256'])
        if args.local_source and row.get('local_source'):
            check('local_source_hash:' + row['file'], sha(args.local_source / row['local_source']) == row['local_source_sha256'])
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    check('explicit_complete_export', actual - {r['file'] for r in entries} == ({'publication_manifest.json', 'publication_verification.json'} & actual))

    def inspect_data(value):
        if isinstance(value, dict):
            check('no_raw_data_fields', not {'original_text', 'processed_text', 'raw_text', 'sentences', 'input_ids', 'targets'}.intersection(value))
            for item in value.values():
                inspect_data(item)
        elif isinstance(value, list):
            for item in value:
                inspect_data(item)

    for name in sorted(actual):
        path = ROOT / name
        check('no_private_assets:' + name, 'local_only' not in path.relative_to(ROOT).parts and path.suffix not in {'.pt', '.pkl', '.npz', '.npy', '.safetensors', '.bin', '.hdf5', '.wav', '.mp4', '.gz', '.zip', '.bpe'} and 'sample_manifest' not in path.name)
        check('bounded_size:' + name, path.stat().st_size < 5 * 1024 * 1024)
        if path.suffix in {'.png', '.pdf'}:
            continue
        text = path.read_text()
        check('no_original_local_identity:' + name, not re.search(r'/(?:home|data[0-9]+)/|\buser[0-9]+\b|GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}', text))
        check('no_credentials:' + name, not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----', text))
        check('no_utterance_identifiers:' + name, not re.search(r'Ses0[1-5][FM]_(?:impro|script)', text))
        if path.suffix == '.py':
            ast.parse(text)
        elif path.suffix == '.json':
            inspect_data(json.loads(text))
        elif path.suffix == '.csv':
            rows = list(csv.DictReader(io.StringIO(text)))
            check('no_raw_csv_columns:' + name, not rows or not {'text', 'original_text', 'processed_text', 'input_ids', 'targets'}.intersection(rows[0]))

    local = json.loads((ROOT / 'verification.json').read_text())
    check('diagnostic_verification_pass', local['status'] == 'PASS' and local['checks_passed'] == 855 and local['original_input_file_count'] == 26 and local['all_original_allowed_inputs_unchanged'])
    framework = json.loads((ROOT / 'results/framework_summary.json').read_text())
    check('framework_eight_comparisons_pass', framework['actual_same_weights_verified'] and framework['all_eight_layer_comparisons_pass'] and len(framework['metrics']) == 8)
    execution = json.loads((ROOT / 'results/overfit_execution.json').read_text())
    check('execution_completed', execution['status'] == 'COMPLETED')
    for ds, fit, dev in [('IEMOCAP', 4747, 536), ('MELD', 8989, 1000)]:
        completion = json.loads((ROOT / 'training_records' / ds / 'completion.json').read_text())
        check(ds + ':500_steps', completion['actual_optimizer_steps'] == 500 and completion['status'] == 'complete')
        check(ds + ':residual_not_compatibility', all(v > 1e-8 for v in completion['final_layer_mse']))
        with (ROOT / 'training_records' / ds / 'optimizer_steps.csv').open() as handle:
            check(ds + ':all_steps_retained', [int(r['step']) for r in csv.DictReader(handle)] == list(range(1, 501)))
        gap = json.loads((ROOT / 'results' / f'train_dev_gap_{ds}.json').read_text())
        check(ds + ':all_fit_dev_results', len(gap['results']) == 4 and all(r['n'] == {'fit': fit, 'inner_dev': dev}[r['subset']] for r in gap['results']))
        for result in gap['results']:
            check(ds + ':layer_mean', math.isclose(result['overall_mse'], sum(result[f'layer{i}_mse'] for i in range(1, 5))/4, rel_tol=1e-12))
        if args.local_source:
            original = json.loads((args.local_source / 'results' / f'train_dev_gap_{ds}.json').read_text())
            check(ds + ':scientific_results_unchanged', original['results'] == gap['results'])
    if args.local_source:
        for row in manifest['files']:
            source, public = args.local_source / row['local_source'], ROOT / row['file']
            if public.suffix == '.csv' and Path(row['file']).parts[0] in {'results', 'training_records'}:
                check('csv_values_unchanged:' + row['file'], list(csv.DictReader(io.StringIO(source.read_text()))) == list(csv.DictReader(io.StringIO(public.read_text()))))
    for link in re.findall(r'\]\(([^)]+)\)', (ROOT/'README.md').read_text()):
        check('README_link:' + link, link == 'publication_verification.json' or (ROOT/link).is_file())
    result = {'verified': True, 'created_at': datetime.now(timezone.utc).isoformat(), 'checks_passed': len(checks), 'files_published': len(entries)+2, 'manifest_sha256': sha(ROOT/'publication_manifest.json'), 'local_source_compared': bool(args.local_source), 'private_assets_uploaded': False, 'training_or_inference_rerun': False, 'diagnostic_delivery_pass_is_not_feature_compatibility': True}
    (ROOT/'publication_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
