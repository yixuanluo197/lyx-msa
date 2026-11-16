"""Static diagnostics using an existing plotting environment; no model fitting."""
import json
import sys
from pathlib import Path
import pandas as pd

def plots(response):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest = Path(__file__).resolve().parents[1] / 'figures'
    dest.mkdir(exist_ok=True)
    data = response[response.population.eq('all_retained')]
    for column, ylabel, filename in [
            ('mu_zero_fraction', 'Utterances with any flip (%)', 'flip_coverage'),
            ('mean_delta', 'Mean original confidence instability delta', 'confidence_instability')]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for row, dataset in enumerate(['IEMOCAP', 'MELD']):
            for col, corruption in enumerate(['gaussian', 'dropout']):
                ax = axes[row, col]
                for modality in ['text', 'audio', 'visual', 'joint']:
                    part = data[data.dataset.eq(dataset) & data.corruption.eq(corruption)
                                & data.modality.eq(modality)].sort_values('strength')
                    y = part[column].to_numpy(float)
                    if column == 'mu_zero_fraction':
                        y = 100 * (1 - y)
                    ax.plot(part.strength, y, marker='o', label=modality)
                ax.set_title(dataset + ' / ' + corruption)
                ax.set_xticks([.05, .10], ['0.05', '0.10'])
                ax.set_xlabel('Strength (dropout: fraction of eligible dimensions)')
                ax.set_ylabel(ylabel)
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.2)
        axes[0, 0].legend()
        fig.savefig(dest / (filename + '.png'), dpi=160)
        fig.savefig(dest / (filename + '.pdf'))
        plt.close(fig)


if __name__ == '__main__':
    import matplotlib
    root = Path(__file__).resolve().parents[1]
    plots(pd.read_csv(root/'results/behavior_summary.csv'))
    (root/'provenance/plot_environment.json').write_text(json.dumps({'python':sys.executable,'python_version':sys.version,'matplotlib':matplotlib.__version__,'pandas':pd.__version__,'purpose':'plots only; all fitting uses frozen cmerc env'},indent=2)+'\n')
