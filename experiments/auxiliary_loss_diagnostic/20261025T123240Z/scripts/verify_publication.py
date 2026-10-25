"""Verify the public XP13 delivery without loading models or private examples."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
ARCHIVE = PROJECT / 'experiment_records/13_auxiliary_loss_diagnostic'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ARCHIVE / 'publication_manifest.json').read_text())
    verification = json.loads((ROOT / 'provenance/final_verification.json').read_text())
    summary = json.loads((ROOT / 'results/metrics_summary.json').read_text())
    assert manifest['status'] == 'completed_and_accepted'
    assert verification['status'] == summary['verification_status'] == 'passed'
    assert summary['case_count'] == 128
    for key in ('training_runs', 'optimizer_steps', 'test_forwards'):
        assert summary['execution_counts'][key] == 0
    assert summary['execution_counts']['model_forward_calls'] == 768
    assert summary['execution_counts']['autograd_calls'] <= 512
    checked, links, python_files, json_files, csv_files = [], 0, 0, 0, 0
    forbidden_fields = {'dialogue_id', 'dialogue_ids', 'utterance_id', 'utterance_index',
                        'raw_text', 'text', 'main_pred', 'removed_pred'}
    def check_json_privacy(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in forbidden_fields, (path, key)
                assert not (key.endswith('_ids') and isinstance(item, list)
                            and key not in ('checkpoint_ids', 'batch_ids')), (path, key)
                check_json_privacy(item, path)
        elif isinstance(value, list):
            for item in value:
                check_json_privacy(item, path)
    for record in manifest['files']:
        path = PROJECT / record['path']
        assert path.is_file() and not path.is_symlink(), path
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], path
        relative_parts = path.relative_to(ROOT).parts if path.is_relative_to(ROOT) else ()
        assert not {'private', 'logs', 'state', '__pycache__'}.intersection(relative_parts), path
        assert path.suffix.lower() not in ('.pt', '.pth', '.npy', '.npz', '.jsonl'), path
        if path.suffix == '.py':
            compile(path.read_bytes(), str(path), 'exec')
            python_files += 1
        elif path.suffix == '.json':
            check_json_privacy(json.loads(path.read_text()), path)
            json_files += 1
        elif path.suffix == '.csv':
            with path.open() as stream:
                reader = csv.DictReader(stream)
                assert not forbidden_fields.intersection(reader.fieldnames or []), path
                for row in reader:
                    assert None not in row, path
            csv_files += 1
        if path.suffix == '.md' and path not in (PROJECT / 'README.md', PROJECT / 'experiment_records/README.md'):
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
                target = target.strip('<>').split('#')[0]
                if not target or '://' in target or target.startswith('mailto:'):
                    continue
                assert (path.parent / target).exists(), (path, target)
                links += 1
        checked.append(record['path'])
    figures = json.loads((ROOT / 'results/figures/manifest.json').read_text())
    assert len(figures['artifacts']) == 18 and figures['completed']
    for record in figures['artifacts']:
        assert sha(ROOT / record['path']) == record['sha256']
    for relative, expected in figures['source_data'].items():
        assert sha(ROOT / relative) == expected
    for relative, expected in summary['inputs_sha256'].items():
        assert sha(ROOT / relative) == expected
    for capsule_name in ('frozen_execution', 'analysis_frozen'):
        capsule = json.loads((ROOT / ('provenance/' + capsule_name + '.json')).read_text())
        records = capsule['files']
        if isinstance(records, dict):
            records = [{'path': name, 'sha256': value} for name, value in records.items()]
        for record in records:
            path = Path(record['path'])
            if not path.is_absolute():
                path = ROOT / path
            assert sha(path) == record['sha256'], path
    batch_report = (ROOT / 'reports/CMERC_AUXILIARY_LOSS_DIAGNOSTIC.md').read_text()
    public_report = (PROJECT / 'reports/CMERC_AUXILIARY_LOSS_DIAGNOSTIC.md').read_text()
    prefix = '../' + str(ROOT.relative_to(PROJECT)) + '/'
    assert batch_report.replace('(../', '(' + prefix) == public_report
    result = {'status': 'passed', 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'public_files': len(checked), 'python_files_compiled': python_files,
              'json_files_parsed': json_files, 'csv_files_checked': csv_files,
              'relative_markdown_links_checked': links, 'figure_files_verified': 18,
              'frozen_capsules_unchanged': True, 'report_link_normalization_equal': True,
              'models_loaded': 0, 'new_training_runs': 0, 'optimizer_steps': 0, 'test_forwards': 0,
              'manifest_sha256': sha(ARCHIVE / 'publication_manifest.json')}
    (ROOT / 'state/publication_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
