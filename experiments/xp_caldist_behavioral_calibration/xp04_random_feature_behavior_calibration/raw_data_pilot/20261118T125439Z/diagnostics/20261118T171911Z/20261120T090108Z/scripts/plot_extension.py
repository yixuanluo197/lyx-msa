"""Plot existing original and extended Train-only loss measurements."""
from pathlib import Path
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path(__file__).resolve().parents[1]


def main():
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,dataset in enumerate(['IEMOCAP','MELD']):
        old=json.loads((D.parent/'local_only/overfit'/dataset/'evaluations.json').read_text())
        new=json.loads((D/'local_only/formal'/dataset/'evaluations.json').read_text())
        points=[dict(global_step=x['step'],layer_mse=x['layer_mse'],mse=x['mse']) for x in old if x['step']<500]+new
        steps=[p['global_step'] for p in points]
        with (D.parent/'local_only/overfit'/dataset/'optimizer_steps.csv').open() as handle:
            original_steps=list(csv.DictReader(handle))
        with (D/'local_only/formal'/dataset/'optimizer_steps.csv').open() as handle:
            extended_steps=list(csv.DictReader(handle))
        train_steps=[int(x['step']) for x in original_steps]+[int(x['global_step']) for x in extended_steps]
        train_losses=[float(x['mean_microbatch_mse']) for x in original_steps+extended_steps]
        axes[row,0].plot(train_steps,train_losses,color='grey',alpha=.25,lw=.5,
                         label='Optimizer-batch train MSE')
        labels=['Last layer','Second last','Third last','Fourth last']
        for layer,label in enumerate(labels):
            values=[p['layer_mse'][layer] for p in points]
            axes[row,0].plot(steps,values,'o-',ms=3,label=label)
            axes[row,1].plot(steps,[math.sqrt(v) for v in values],'o-',ms=3,label=label)
        axes[row,0].plot(steps,[p['mse'] for p in points],'k--',lw=1.8,label='Four-layer mean')
        axes[row,1].axhline(1e-4,color='black',linestyle='--',label='Required RMSE 1e-4')
        for column,metric in enumerate(['MSE','RMSE']):
            ax=axes[row,column]
            ax.axvline(500,color='grey',linestyle=':',label='New AdamW state at 500')
            ax.set(title=f'{dataset}: fixed 128 Train Fit texts',xlabel='Total optimizer step',
                ylabel=f'FP32 eval {metric} (log scale)',yscale='log')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    for extension in ['png','pdf']:
        fig.savefig(D/'results'/f'training_extension.{extension}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
