"""Repair a clipped legend only; preserve original plots and their hashes privately."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import plot_diagnostic as original

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / 'results/figures'
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    names = ['07_pseudo_branch_coverage.png', '07_pseudo_branch_coverage.svg']
    backup = ROOT / 'private/figures_before_layout'
    backup.mkdir(exist_ok=False)
    for name in names + ['manifest.json']:
        shutil.copy2(output / name, backup / name)
    rows = original.readcsv(ROOT / 'results/branch_summary.csv')
    branches = sorted({row['branch'] for row in rows})
    branch_labels = ['Both correct: confidence up/equal (0)',
                     'Both correct: confidence down (1)',
                     'Both wrong: default 0', 'Correct to wrong (1)', 'Wrong to correct (0)']
    assert len(branches) == 5
    assert branches[0].startswith('both_correct_removed_conf_ge')
    assert branches[1].startswith('both_correct_removed_conf_lower')
    assert branches[2:] == ['both_wrong_default0', 'correct_to_wrong_explicit1', 'wrong_to_correct_explicit0']
    fig, axis = original.canvas(
        'Correctness and confidence branches behind pseudo-labels', 'Utterance-observation fraction',
        'Five implementation branches; both-wrong cases retain default 0.\n'
        'Relationship labels are not emotion categories. Denominator: all sampled utterances per checkpoint/relation.',
        width=12., height=6.0)
    fig.subplots_adjust(right=.70)
    labels = [cp.replace('XP', '') + '\n' + relation for cp in original.CP for relation in original.RELATIONS]
    bottom = np.zeros(20)
    for branch, label, color in zip(branches, branch_labels, ['#777777', '#0072B2', '#E69F00', '#009E73', '#CC79A7']):
        values = []
        for cp in original.CP:
            for relation in original.RELATIONS:
                part = [r for r in rows if r['checkpoint_id'] == cp and r['relation'] == relation]
                total = sum(float(r['n_utterances']) for r in part)
                values.append(sum(float(r['n_utterances']) for r in part if r['branch'] == branch) / total)
        axis.bar(range(20), values, bottom=bottom, width=.8, label=label, color=color)
        bottom += values
    assert np.allclose(bottom, 1)
    axis.set_xticks(range(20))
    axis.set_xticklabels(labels, rotation=65, ha='right', fontsize=6.5)
    axis.set_ylim(0, 1)
    axis.legend(loc='center left', bbox_to_anchor=(1.02, .5), fontsize=7)
    changes = []
    for name in names:
        target = output / name
        fig.savefig(target, dpi=240 if target.suffix == '.png' else None, facecolor='white')
        record = next(r for r in manifest['artifacts'] if r['path'] == str(target.relative_to(ROOT)))
        changes.append({'path': record['path'], 'before_sha256': record['sha256'], 'after_sha256': sha(target)})
        record.update(sha256=sha(target), bytes=target.stat().st_size)
    plt.close(fig)
    script = Path(__file__).resolve()
    manifest['source_data'][str(script.relative_to(ROOT))] = sha(script)
    manifest['layout_only_revision'] = {'script': str(script.relative_to(ROOT)), 'changes': changes,
                                      'data_or_statistics_changed': False, 'frozen_original_plot_script_unchanged': True}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    record = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'reason': 'Visual inspection found figure07 original long legend clipped at right edge.',
              'changes': changes, 'model_forwards': 0, 'autograd_calls': 0, 'data_changed': False,
              'original_outputs_preserved_private': True, 'frozen_script_unchanged': True}
    (ROOT / 'provenance/figure7_layout_revision.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
