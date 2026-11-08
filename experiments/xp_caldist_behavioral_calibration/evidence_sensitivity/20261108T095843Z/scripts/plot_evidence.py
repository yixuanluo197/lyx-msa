#!/usr/bin/env python
"""Python-only figures from real, saved validation sensitivity CSVs."""
from __future__ import annotations
import argparse
import csv
import json
import textwrap
from pathlib import Path
import numpy as np

COLORS = {'T-All': '#2166AC', 'A': '#D88330', 'V': '#3F8A79',
          'T1': '#A6C8E2', 'T2': '#7EADD2', 'T3': '#4E91C2', 'T4': '#2C6EA3'}
CLASS_NAMES = {'IEMOCAP': ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated'],
               'MELD': ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']}


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def curve_arrays(rows, metric):
    """Rho0 is observed baseline accuracy, or exactly-zero paired behavior."""
    rows = sorted(rows, key=lambda r: float(r['strength']))
    if not rows:
        raise ValueError('missing attenuation curve')
    x = np.array([0.] + [float(r['strength']) for r in rows])
    if metric == 'perturbed_accuracy':
        baseline = tuple(float(rows[0][key]) for key in
                         ('original_accuracy', 'original_accuracy_ci_low', 'original_accuracy_ci_high'))
    else:
        baseline = (0., 0., 0.)
    y = np.array([baseline[0]] + [float(r[metric]) for r in rows])
    low = np.array([baseline[1]] + [float(r[metric + '_ci_low']) for r in rows])
    high = np.array([baseline[2]] + [float(r[metric + '_ci_high']) for r in rows])
    return x, y, low, high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    contract = root / 'configs/figure_contract.md'
    if not contract.exists():
        raise ValueError('figure contract must exist before plotting')
    config = json.loads((root / 'configs/evidence_sensitivity_config.json').read_text())
    if config['figures']['backend'] != 'Python, matching supplied PyTorch/.py workflow':
        raise ValueError('Python backend must be explicitly selected')
    summary = read_csv(root / 'results/sensitivity_summary.csv')
    classes = read_csv(root / 'results/classwise_sensitivity.csv')
    missingness = read_csv(root / 'results/visual_missingness_analysis.csv')
    # No model/synthetic data generation occurs in this script.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
                         'axes.labelsize': 8, 'legend.fontsize': 7, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .75, 'lines.linewidth': 1.5, 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'xtick.labelsize': 7, 'ytick.labelsize': 7})
    out = root / 'figures'; out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def export(fig, name, source, note):
        fig.text(.5, .015, textwrap.fill(note, width=125), ha='center', va='bottom', fontsize=6.7)
        fig.subplots_adjust(bottom=.24, top=.85, wspace=.3)
        files = []
        for extension in ('svg', 'pdf', 'png'):
            path = out / f'{name}.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.06)
            files.append(str(path.relative_to(root)))
        # Verify the Python render exists and has nonzero pixel variance.
        from PIL import Image
        with Image.open(out / f'{name}.png') as im:
            pixels = np.asarray(im.convert('RGB'))
            if pixels.std() == 0 or min(im.size) < 200:
                raise ValueError('empty or undersized figure render: ' + name)
            dimensions = list(im.size)
        manifest.append(dict(name=name, source=source, outputs=files, png_dimensions=dimensions,
                             png_nonzero_pixel_variance=True, note=note))
        plt.close(fig)

    attenuation = [r for r in summary if r['kind'] == 'attenuation']
    sizes = config['validation_sizes']
    seeds = config['anchors']

    def curves(name, metric, ylabel, targets, visual_strata=False):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
        for panel, (ax, dataset) in enumerate(zip(axes, config['anchors'])):
            for target in targets:
                strata = ['all', 'visual_present'] if target == 'V' and visual_strata else ['all']
                for subgroup in strata:
                    rows = [r for r in attenuation if r['dataset'] == dataset and r['target'] == target and r['subgroup'] == subgroup]
                    x, y, lo, hi = curve_arrays(rows, metric)
                    label = f'V ({"present" if subgroup == "visual_present" else "all"})' if target == 'V' and visual_strata else target
                    style = '--' if subgroup == 'visual_present' else '-'
                    ax.plot(x, y, marker='o', markersize=3, color=COLORS[target], linestyle=style, label=label)
                    ax.fill_between(x, lo, hi, color=COLORS[target], alpha=.1 if subgroup == 'all' else .055, linewidth=0)
            ax.set(xlabel='Contraction toward training mean (rho)', ylabel=ylabel if panel == 0 else '',
                   xticks=[0, .25, .5, .75, 1.], xlim=(-.03, 1.03))
            ax.set_title(f'{chr(97+panel)}  {dataset} | seed {seeds[dataset]}\n{sizes[dataset]["utterances"]} utterances, {sizes[dataset]["dialogues"]} dialogues', loc='left')
            if metric in ('perturbed_accuracy', 'flip_rate'):
                ax.set_ylim(0, 1)
            else:
                ax.axhline(0, color='#999999', linewidth=.6, zorder=0)
            ax.legend(loc='best', ncol=2, handlelength=2)
        export(fig, name, ['results/sensitivity_summary.csv'],
               ('Validation only; shaded bands: 95% paired dialogue bootstrap. V-present excludes zero visual rows; V-all retains them.'
                if visual_strata else
                'Validation only; shaded bands: 95% paired dialogue bootstrap. T1-T4 are stored arrays; other utterances and modalities remain unchanged.'))

    curves('modality_acc_curves', 'perturbed_accuracy', 'Accuracy (fraction)', ['T-All', 'A', 'V'], True)
    curves('trueprob_curves', 'true_probability_change', 'Change in true-class probability', ['T-All', 'A', 'V'], True)
    curves('text_layer_comparison', 'perturbed_accuracy', 'Accuracy (fraction)', ['T1', 'T2', 'T3', 'T4', 'T-All'])
    curves('flip_curves', 'flip_rate', 'Prediction flip rate (fraction)', ['T-All', 'A', 'V'], True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    matrices, supports = [], []
    for dataset in config['anchors']:
        matrix = np.full((3, len(CLASS_NAMES[dataset])), np.nan)
        support = np.zeros(len(CLASS_NAMES[dataset]), int)
        for row in classes:
            if (row['dataset'] == dataset and row['phase'] == 'overall_attenuation' and float(row['strength']) == 1
                    and row['class_basis'] == 'true_class'):
                i, j = ['T-All', 'A', 'V'].index(row['target']), int(row['class_index'])
                matrix[i, j] = float(row['delta_accuracy']); support[j] = int(row['n_utterances'])
        matrices.append(matrix); supports.append(support)
    values = np.concatenate([m.ravel() for m in matrices])
    limit = max(float(np.max(np.abs(values[np.isfinite(values)]))) if np.isfinite(values).any() else 0., .01)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for panel, (ax, dataset, matrix, support) in enumerate(zip(axes, config['anchors'], matrices, supports)):
        image = ax.imshow(matrix, cmap='RdBu', norm=norm, aspect='auto')
        ax.set_xticks(range(matrix.shape[1]), [f'{name}\nn={n}' for name,n in zip(CLASS_NAMES[dataset], support)],
                      rotation=45, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(3), ['T-All', 'A', 'V'])
        ax.set_title(f'{chr(97+panel)}  {dataset}: rho=1, true classes', loc='left')
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f'{value:+.2f}' if np.isfinite(value) else 'NA', ha='center', va='center',
                        fontsize=6.7, color='white' if abs(value) > .65*limit else 'black')
        ax.spines[['top','right','bottom','left']].set_visible(False)
    fig.subplots_adjust(right=.88)
    cax = fig.add_axes([.91, .34, .018, .41])
    fig.colorbar(image, cax=cax, label='Accuracy change (fraction)')
    export(fig, 'class_heatmap', ['results/classwise_sensitivity.csv'],
           'Validation only; all rows retained. Negative values indicate lower accuracy. Classwise paired CIs are in the source CSV; exploratory.')

    meld = [r for r in missingness if r['dataset'] == 'MELD']
    order = ['visual_present', 'visual_missing']
    meld = [next(r for r in meld if r['subgroup'] == group) for group in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    x = np.arange(2); names = ['Visual present', 'All-zero visual']
    counts = [int(r['n_utterances']) for r in meld]
    bars = axes[0].bar(x, counts, color=[COLORS['V'], '#B8C2C0'], width=.6)
    bars[1].set_hatch('//')
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x()+bar.get_width()/2, count+max(counts)*.03,
                     f'{count} ({count/sum(counts):.1%})', ha='center', fontsize=8)
    axes[0].set(xticks=x, xticklabels=names, ylabel='Number of validation utterances', ylim=(0, max(counts)*1.18))
    axes[0].set_title('a  MELD visual feature availability', loc='left')
    for offset, kind, marker, color in [(-.09, 'original', 'o', '#687573'), (.09, 'V rho=1', 's', COLORS['V'])]:
        points, low, high = [], [], []
        for group, row in zip(order, meld):
            if kind == 'original':
                points.append(float(row['original_accuracy'])); low.append(float(row['original_accuracy_ci_low'])); high.append(float(row['original_accuracy_ci_high']))
            else:
                selected = next(r for r in summary if r['dataset'] == 'MELD' and r['target'] == 'V' and r['subgroup'] == group and r['phase'] == 'overall_attenuation' and float(r['strength']) == 1)
                points.append(float(selected['perturbed_accuracy'])); low.append(float(selected['perturbed_accuracy_ci_low'])); high.append(float(selected['perturbed_accuracy_ci_high']))
        points, low, high = map(np.asarray, (points, low, high))
        # Percentile intervals can exclude point estimates; draw segments rather than invalid negative yerr.
        axes[1].vlines(x+offset, low, high, color=color, linewidth=1)
        axes[1].scatter(x+offset, points, marker=marker, color=color, s=25, label=kind, zorder=3)
    axes[1].set(xticks=x, xticklabels=names, ylabel='Accuracy (fraction)', ylim=(0, 1))
    axes[1].set_title('b  Original vs visual contraction', loc='left')
    axes[1].legend(loc='best')
    export(fig, 'meld_visual_missingness', ['results/visual_missingness_analysis.csv', 'results/sensitivity_summary.csv'],
           'Missing rows receive no V intervention and remain in all-sample results. Missingness groups are observational; bars: 95% dialogue bootstrap.')
    (out / 'figure_manifest.json').write_text(json.dumps(dict(backend='Python/matplotlib', matplotlib_version=matplotlib.__version__,
                                            figures=manifest, real_saved_validation_data_only=True), indent=2) + '\n')
    print(json.dumps(dict(figures=len(manifest), formats=['svg','pdf','png'], backend=matplotlib.__version__), indent=2))


if __name__ == '__main__':
    main()
