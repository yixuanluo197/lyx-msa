#!/usr/bin/env python3
"""Plot frozen paired aggregates and complete trajectories; performs no inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {'baseline': '#0072B2', 'cmerc': '#D55E00'}
SEEDS = (1937, 2025, 3407)
DATASETS = ('IEMOCAP', 'MELD')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.statistics.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42})

    def save(fig, name):
        fig.tight_layout()
        for extension in ('pdf', 'svg', 'png'):
            fig.savefig(args.output_dir / f'{name}.{extension}', bbox_inches='tight', dpi=150)
        plt.close(fig)

    for dataset in DATASETS:
        if not any(key.startswith(dataset + '/') for key in data['training']):
            continue
        fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex='col', sharey=True)
        for column, seed in enumerate(SEEDS):
            variants = data['training'].get(f'{dataset}/{seed}', {})
            axes[0, column].set_title(f'{dataset} seed {seed}')
            for row, field in enumerate(('train_weighted_f1', 'validation_weighted_f1')):
                ax = axes[row, column]
                for variant, run in variants.items():
                    records = run['convergence']
                    epochs = [record['epoch'] for record in records]
                    ax.plot(epochs, [record[field] for record in records], color=COLORS[variant],
                            label='Baseline' if variant == 'baseline' else 'CMERC', linewidth=1.2)
                    if row == 1:
                        best = next(record for record in records if record['epoch'] == run['best_epoch'])
                        ax.scatter(best['epoch'], best[field], marker='*', s=65,
                                   color=COLORS[variant], edgecolor='black', linewidth=.3, zorder=4)
                ax.set_ylim(0, 100)
                ax.grid(alpha=.2)
                if column == 0:
                    ax.set_ylabel(('Train' if row == 0 else 'Validation') + ' W-F1 (%)')
                if row == 1:
                    ax.set_xlabel('Epoch')
                if not variants:
                    ax.text(.5, .5, 'Not completed', ha='center', transform=ax.transAxes)
        axes[0, 0].legend(frameon=False)
        save(fig, f'convergence_{dataset}')

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        components = ('classification_loss', 'cc_loss', 'mscl_loss', 'cscl_loss')
        colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
        for ax, seed in zip(axes, SEEDS):
            run = data['training'].get(f'{dataset}/{seed}', {}).get('cmerc')
            ax.set_title(f'{dataset} seed {seed}')
            if run:
                for name, color in zip(components, colors):
                    records = [row for row in run['gradient_trajectory']
                               if row['weighted_gradient_norms'].get(name, 0) > 0]
                    ax.plot([row['epoch'] + 1 for row in records],
                            [row['weighted_gradient_norms'][name] for row in records],
                            label=name.replace('_loss', ''), color=color, linewidth=1)
            ax.set_yscale('log')
            ax.set_xlabel('Epoch; first batch only')
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Actual weighted gradient L2 norm')
        axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle('First batch only; zero values omitted on log scale', fontsize=10)
        save(fig, f'weighted_gradients_{dataset}')

    for split in ('validation', 'test'):
        fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))
        for row, dataset in enumerate(DATASETS):
            for column, metric in enumerate(('weighted_f1', 'ece', 'cel.total')):
                ax = axes[row, column]
                entries = sorted([entry for entry in data['per_seed'] if entry['dataset'] == dataset
                                  and entry['split'] == split], key=lambda entry: SEEDS.index(entry['seed']))
                values = [entry['overall'][metric]['delta'] for entry in entries]
                positions = [SEEDS.index(entry['seed']) for entry in entries]
                ax.axhline(0, color='black', linewidth=.7)
                ax.scatter(positions, values, color=COLORS['cmerc'], s=28, zorder=3)
                if len(values) > 1:
                    ax.errorbar(3.1, np.mean(values), yerr=np.std(values, ddof=1),
                                fmt='s', color='black', capsize=4, markersize=4)
                ax.set_xticks([0, 1, 2, 3.1], ['1937', '2025', '3407', 'mean±SD'], rotation=20)
                ax.set_title(f'{dataset}: {metric}')
                ax.set_ylabel('CMERC − Baseline (' + ('CEL units' if metric == 'cel.total' else 'pp') + ')')
                ax.grid(axis='y', alpha=.2)
        fig.suptitle(f'{split.title()}: fixed paired seeds; n=3; descriptive uncertainty', fontsize=10)
        save(fig, f'paired_deltas_{split}')

    manifest = {'source_statistics_sha256': hashlib.sha256(args.statistics.read_bytes()).hexdigest(),
                'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope': 'all epochs and all fixed seeds; first batch weighted gradients; no inference',
                'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.output_dir.iterdir())}}
    (args.output_dir / 'FIGURE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'figures': len(manifest['files']), 'output': str(args.output_dir)}))


if __name__ == '__main__':
    main()
