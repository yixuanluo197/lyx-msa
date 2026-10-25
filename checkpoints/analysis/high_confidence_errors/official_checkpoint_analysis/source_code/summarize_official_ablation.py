#!/usr/bin/env python3
"""Validate fixed author-checkpoint ablations and align them to cached full inputs."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.analyze_official_reliability import high_confidence_counts, sha256, write_csv
from scripts.high_confidence_followup_stats import THRESHOLDS, _validate

CONDITIONS={'w/o_text':'text','w/o_audio':'audio','w/o_visual':'visual'}


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for key in ('true_label','predicted_label','utterance_index'):row[key]=int(row[key])
        for key in list(row):
            if key.startswith('probability_') or key in ('predicted_confidence','true_label_probability','true_label_log_probability'):
                row[key]=float(row[key])
        row['correct']=row['true_label']==row['predicted_label']
    return rows


def summarize_transitions(full, masked, dataset, condition):
    if len(full)!=len(masked):raise ValueError('Length mismatch')
    output=[]
    full_correct=sum(r['correct'] for r in full)
    full_wrong=len(full)-full_correct
    still_wrong=sum(not f['correct'] and not m['correct'] for f,m in zip(full,masked))
    for threshold in THRESHOLDS:
        count=sum(f['correct'] and not m['correct'] and m['predicted_confidence']>=threshold for f,m in zip(full,masked))
        output.append(dict(dataset=dataset,model='Official CMERC',input_condition=condition,
            removed_modality=CONDITIONS[condition],threshold=threshold,full_correct_samples=full_correct,
            new_high_confidence_errors=count,ratio=count/full_correct if full_correct else None,
            full_wrong_samples=full_wrong,still_wrong_count=still_wrong,
            still_wrong_ratio=still_wrong/full_wrong if full_wrong else None))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PROJECT/'checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis')
    parser.add_argument('--source-dir',type=Path,default=PROJECT/'checkpoints/analysis/official_checkpoint_verification')
    args=parser.parse_args(); output=args.output_dir.resolve(); source=args.source_dir.resolve()
    audits=[]; total_forwards=0
    for dataset in ('IEMOCAP','MELD'):
        directory=output/dataset
        full=read_rows(directory/'all_predictions.csv')
        expected=json.loads((source/f'{dataset}_run_manifest.json').read_text())
        rows_all=[];summaries=[];transitions=[];new_errors=[]
        class_count=sum(k.startswith('probability_') for k in full[0])
        keys=[k for k in full[0] if k.startswith('probability_')]
        condition_groups={'full':full}
        for condition,modality in CONDITIONS.items():
            run=directory/'inference'/f'without_{modality}'
            rows=read_rows(run/f'{dataset}_predictions.csv')
            manifest=json.loads((run/f'{dataset}_run_manifest.json').read_text())
            assert manifest['status']=='completed' and manifest['model_state_unchanged'] and manifest['original_inputs_unchanged']
            assert manifest['reference_artifacts_unchanged'] and manifest['all_parameter_grads_none']
            assert manifest['full_forward_calls']==0 and manifest['audit']['optimizer_steps']==0
            assert manifest['official_argv']==expected['official_argv']
            for key in ('cudnn_enabled','cudnn_deterministic','cudnn_version','torch','cuda_runtime'):
                if manifest[key]!=expected[key]:raise ValueError(f'Backend changed: {key}')
            if manifest['source_prediction_sha256']!=sha256(source/f'{dataset}_predictions.csv'):raise ValueError('Source changed')
            assert len(rows)==len(full)
            _validate(rows,(dataset,'Official CMERC'),condition)
            with np.load(run/f'{dataset}_predictions.npz',allow_pickle=False) as values:
                probabilities=np.array([[row[k] for k in keys] for row in rows])
                assert probabilities.shape==(len(full),class_count)
                assert np.array_equal(probabilities,values['probabilities'])
                assert np.array_equal(values['labels'],[r['true_label'] for r in rows])
                assert np.array_equal(values['predictions'],probabilities.argmax(1))
                assert np.isfinite(values['log_probabilities']).all()
                assert np.allclose(probabilities,np.exp(values['log_probabilities'].astype(np.float64)),rtol=0,atol=1e-15)
                for i,row in enumerate(rows):row['true_label_log_probability']=float(values['log_probabilities'][i,row['true_label']])
            for f,m in zip(full,rows):
                for key in ('dataset','model','dialogue_id','utterance_index','source_utterance_id','utterance_text','true_label','true_emotion'):
                    if f[key]!=m[key]:raise ValueError(f'Identity mismatch {key}')
            batches=manifest['audit']['batches']
            assert len(batches)==len(expected['audit']['batches'])
            for b,orig in zip(batches,expected['audit']['batches']):
                for key in ('dialogue_ids','lengths','padded_time','feature_shapes','batch_size','num_workers'):
                    assert b[key]==orig[key], key
            checks=manifest['audit']['replacement_checks']
            assert len(checks)==len(batches)==manifest['audit']['forward_calls']==manifest['audit']['masking_forward_calls']
            assert all(c['mean_axis']==0 and c['padding_included_in_mean'] and c['non_target_arguments_identical'] for c in checks)
            assert sum(c['target_changed_elements'] for c in checks)>0
            total_forwards+=len(checks)
            audits.append(dict(dataset=dataset,input_condition=condition,valid_samples=len(rows),masked_forward_calls=len(checks),
                gpu_host_index=manifest['gpu_host_index'],actual_gpu_uuids=manifest['actual_gpu_uuids'],
                source_manifest_sha256=sha256(run/f'{dataset}_run_manifest.json'),all_checks_passed=True))
            condition_groups[condition]=rows
            transitions+=summarize_transitions(full,rows,dataset,condition)
        for condition,rows in condition_groups.items():
            errors=[r for r in rows if not r['correct']]
            from sklearn.metrics import f1_score
            summary=dict(dataset=dataset,model='Official CMERC',input_condition=condition,
                valid_samples=len(rows),total_errors=len(errors),
                mean_error_confidence=sum(r['predicted_confidence'] for r in errors)/len(errors) if errors else None,
                accuracy=sum(r['correct'] for r in rows)/len(rows),
                weighted_f1=float(f1_score([r['true_label'] for r in rows],[r['predicted_label'] for r in rows],average='weighted')),
                **high_confidence_counts(rows))
            summaries.append(summary)
            for f,m in zip(full,rows):
                row=dict(m,removed_modality=CONDITIONS.get(condition,'none'),
                    full_prediction=f['predicted_label'],full_confidence=f['predicted_confidence'],
                    full_correct=f['correct'],full_true_label_probability=f['true_label_probability'],
                    true_label_probability_delta=m['true_label_probability']-f['true_label_probability'],
                    transition_type=('full_correct_' if f['correct'] else 'full_wrong_')+('masked_correct' if m['correct'] else 'masked_wrong'))
                rows_all.append(row)
                if condition!='full' and f['correct'] and not m['correct']:
                    for threshold in THRESHOLDS:
                        if m['predicted_confidence']>=threshold:new_errors.append(dict(row,threshold=threshold))
        write_csv(directory/'modality_ablation_predictions.csv',rows_all)
        write_csv(directory/'modality_ablation_summary.csv',summaries)
        write_csv(directory/'full_correct_to_masked_error.csv',transitions)
        write_csv(directory/'new_masked_high_confidence_errors.csv',new_errors,fields=list(rows_all[0])+['threshold'])
    assert total_forwards==60
    payload=dict(status='passed',full_forward_calls=0,masked_forward_calls=total_forwards,optimizer_steps=0,
        sample_alignment='Exact per-row dataset/dialogue/index/source_id/text/true_label alignment with cached full predictions',
        mean_protocol='torch.mean(feature, dim=0).repeat(time, 1, 1), includes padded zeros; official caldiff/cal_diff.py lines 241-247',
        unchanged_official_runtime=True,unchanged_checkpoint_and_input_files=True,datasets_and_conditions=audits)
    (output/'ablation_verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
