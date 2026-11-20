"""Summarize completed fixed-budget extension measurements without model inference."""
import csv
import json
import math
from pathlib import Path

D = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    per_layer, stages, conclusions = [], [], {}
    for dataset in ['IEMOCAP', 'MELD']:
        run = D / 'local_only/formal' / dataset
        completion = json.loads((run / 'completion.json').read_text())
        assert completion['status'] == 'COMPLETE' and not completion['smoke']
        evaluations = json.loads((run / 'evaluations.json').read_text())
        assert evaluations[0]['global_step'] == 500
        for j, item in enumerate(evaluations):
            previous = evaluations[j-1] if j else None
            stages.append(dict(dataset=dataset, global_step=item['global_step'],
                overall_mse=item['mse'], overall_rmse=math.sqrt(item['mse']),
                previous_global_step=previous['global_step'] if previous else '',
                relative_decrease_from_previous=1-item['mse']/previous['mse'] if previous else '',
                relative_decrease_from_step500=1-item['mse']/evaluations[0]['mse']))
            for layer, value in enumerate(item['layer_mse'], 1):
                per_layer.append(dict(dataset=dataset, global_step=item['global_step'],
                    layer_from_end=layer, mse=value, rmse=math.sqrt(value),
                    relative_decrease_from_previous=1-value/previous['layer_mse'][layer-1] if previous else '',
                    relative_decrease_from_step500=1-value/evaluations[0]['layer_mse'][layer-1],
                    rmse_limit=1e-4, rmse_ratio_to_limit=math.sqrt(value)/1e-4,
                    rmse_necessary_condition_pass=math.sqrt(value)<=1e-4))
        final = evaluations[-1]
        # Apply the same 500-step window for the descriptive final trend.
        reference = next(x for x in evaluations if x['global_step']==final['global_step']-500)
        reduction = 1-final['mse']/reference['mse']
        conclusions[dataset] = dict(final_global_step=final['global_step'],
            initial_mse=evaluations[0]['mse'], final_mse=final['mse'],
            final_overall_rmse=math.sqrt(final['mse']),
            relative_reduction_from_step500=1-final['mse']/evaluations[0]['mse'],
            final_window_start=reference['global_step'], final_window_relative_decrease=reduction,
            final_window_plateau_by_5percent_rule=reduction<=0.05,
            final_layer_mse=final['layer_mse'], final_layer_rmse=final['layer_rmse'],
            rmse_necessary_condition_pass=all(x<=1e-4 for x in final['layer_rmse']),
            decision=completion['decision'], optimizer_reset=True,
            interpretation_limit='Train-only point estimates; endpoint window can be noisy; no held-out compatibility or XP04 probability inference')
    write_csv(D/'results/per_layer_comparison.csv', per_layer)
    write_csv(D/'results/stage_comparison.csv', stages)
    (D/'results/extension_summary.json').write_text(json.dumps(conclusions, indent=2)+'\n')
    print(json.dumps(conclusions, indent=2))


if __name__ == '__main__':
    main()
