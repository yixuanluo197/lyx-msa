#!/usr/bin/env python3
"""Run the fixed official testing entry under no_grad, observing full outputs only."""
import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import runpy
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

CLASSES = {
    'IEMOCAP': ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
    'MELD': ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def state_compatibility(expected, checkpoint):
    return dict(missing_keys=sorted(set(expected)-set(checkpoint)),
                unexpected_keys=sorted(set(checkpoint)-set(expected)),
                shape_mismatches=[dict(key=key, expected=list(expected[key].shape),
                                       checkpoint=list(checkpoint[key].shape))
                                  for key in sorted(set(expected)&set(checkpoint))
                                  if expected[key].shape != checkpoint[key].shape])


def observer_line(path):
    tree = ast.parse(Path(path).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'train_or_eval_graph_model']
    matches = [node.lineno for function in functions for node in ast.walk(function)
               if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name) and node.value.func.id == 'loss_function'
               and [getattr(arg, 'id', None) for arg in node.value.args] == ['log_prob', 'label']]
    if len(matches) != 1:
        raise ValueError('Cannot uniquely locate the read-only evaluation observer')
    return matches[0]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, records):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=CLASSES, required=True)
    parser.add_argument('--runtime-code', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output, source, inputs = args.output_dir.resolve(), args.runtime_code.resolve(), args.input_root.resolve()
    dataset, names = args.dataset, CLASSES[args.dataset]
    if (output/f'{dataset}_run_manifest.json').exists():
        parser.error('refusing to overwrite an existing dataset evaluation')
    preflight = json.loads((output/'preflight.json').read_text())
    checkpoint = inputs/'checkpoints'/f'{dataset}_checkpoint.pkl'
    original_inputs = {str(inputs/path): digest for path,digest in preflight['input_sha256'].items()}
    for path, expected in original_inputs.items():
        if sha256(path) != expected: raise ValueError(f'Trusted input hash mismatch: {path}')
    for path, expected in preflight['runtime_sha256'].items():
        if sha256(source/path) != expected: raise ValueError(f'Frozen runtime changed: {path}')
    import numpy as np
    import torch
    import sklearn
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

    if not torch.cuda.is_available():
        raise RuntimeError('Official noncausal graph contains CUDA-only operations; no silent CPU rewrite')
    torch.cuda.set_device(0)
    if torch.cuda.get_device_properties(0).name != 'Tesla V100-SXM2-32GB':
        raise RuntimeError('Unexpected GPU; inspect server/device mapping before proceeding')
    started = datetime.now(timezone.utc).isoformat()
    entry = source/'train.py'
    line = observer_line(entry)
    audit = dict(strict_load_calls=0, forward_calls=0, optimizer_steps=0, masking_forward_calls=0,
                 batches=[], gradient_enabled_observations=[], model_training_observations=[])
    predictions, log_outputs, labels_outputs, official_returns = [], [], [], []
    model_holder, inventory = {}, {}
    old_load, old_state_load, old_step = torch.load, torch.nn.Module.load_state_dict, torch.optim.Adam.step

    def tensor_state_hash(model):
        digest = hashlib.sha256()
        for key, tensor in model.state_dict().items():
            digest.update(key.encode()); digest.update(str(tensor.dtype).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def trusted_load(path, *unused, **kwargs):
        if Path(path).resolve() != checkpoint.resolve():
            raise ValueError(f'Unexpected weight file requested by official entry: {path}')
        state = old_load(path, map_location='cpu', weights_only=True)
        if not state or not all(isinstance(k,str) and torch.is_tensor(v) for k,v in state.items()):
            raise ValueError('Expected a plain author state_dict, without training wrappers')
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Non-finite official checkpoint parameter')
        inventory.update(path=str(checkpoint),sha256=sha256(checkpoint),size_bytes=checkpoint.stat().st_size,
                         structure=type(state).__name__,tensor_keys=len(state),metadata_present=False,
                         training_epoch=None,training_seed=None,selection_protocol=None,
                         identity='Author release evaluated using the official CMERC testing command; internal training flags unavailable',
                         parameters={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in state.items()})
        return state

    def strict_load(model, state, strict=True):
        if strict is not True: raise ValueError('Non-strict weight loading is forbidden')
        comparison = state_compatibility(model.state_dict(),state)
        inventory['compatibility'] = comparison
        if any(comparison.values()): raise ValueError(f'Official weights do not match model: {comparison}')
        result = old_state_load(model,state,strict=True)
        audit['strict_load_calls'] += 1
        model_holder.update(model=model,before=tensor_state_hash(model))
        model.register_forward_pre_hook(lambda module, call: audit.update(forward_calls=audit['forward_calls']+1))
        return result

    def forbidden_step(*unused, **kwargs):
        audit['optimizer_steps'] += 1
        raise RuntimeError('An optimizer update was attempted during author checkpoint verification')

    def trace(frame, event, arg):
        if frame.f_code.co_filename != str(entry) or frame.f_code.co_name != 'train_or_eval_graph_model':
            return None
        if event == 'line' and frame.f_lineno == line:
            local = frame.f_locals
            if local['train'] or local['model'].training or torch.is_grad_enabled():
                raise RuntimeError('Evaluation must be full-input eval mode under no_grad')
            audit['gradient_enabled_observations'].append(torch.is_grad_enabled())
            audit['model_training_observations'].append(local['model'].training)
            batch, lengths = local['data'], [int(v) for v in local['lengths']]
            dataset_object = local['dataloader'].dataset
            logp = local['log_prob'].detach().cpu().numpy().copy()
            labels = local['label'].detach().cpu().numpy().copy()
            if logp.shape != (sum(lengths),len(names)) or len(labels) != sum(lengths):
                raise ValueError('Official output shape differs from valid labels')
            probabilities = np.exp(logp.astype(np.float64))
            if not np.isfinite(logp).all() or not np.allclose(probabilities.sum(1),1,rtol=0,atol=1e-5):
                raise ValueError('Invalid official log-softmax probabilities')
            offset = 0
            for b,(vid,length) in enumerate(zip(batch[-1],lengths)):
                mask = batch[7][b].cpu().numpy()
                if not (np.all(mask[:length]==1) and np.all(mask[length:]==0)):
                    raise ValueError('Padding mask is not a contiguous valid prefix')
                if len(dataset_object.videoLabels[vid]) != length:
                    raise ValueError('Original dialogue length differs from evaluation labels')
                for index in range(length):
                    true = int(labels[offset])
                    if true != int(dataset_object.videoLabels[vid][index]):
                        raise ValueError('Official flattened label order differs from raw data')
                    predicted = int(probabilities[offset].argmax())
                    row = dict(dataset=dataset,model='Official CMERC',dialogue_id=str(vid),utterance_index=index,
                               source_utterance_id=str(dataset_object.videoIDs[vid][index]),
                               utterance_text=dataset_object.videoSentence[vid][index],true_label=true,true_emotion=names[true],
                               predicted_label=predicted,predicted_emotion=names[predicted],
                               predicted_confidence=float(probabilities[offset,predicted]),
                               true_label_probability=float(probabilities[offset,true]),input_condition='full')
                    row.update({f'probability_{name}':float(probabilities[offset,k]) for k,name in enumerate(names)})
                    predictions.append(row); offset += 1
            log_outputs.append(logp); labels_outputs.append(labels)
            audit['batches'].append(dict(dialogue_ids=[str(v) for v in batch[-1]],lengths=lengths,
                valid_utterances=sum(lengths),padded_time=int(batch[0].shape[0]),
                feature_shapes=[list(t.shape) for t in batch[:-1]],batch_size=local['dataloader'].batch_size,
                num_workers=local['dataloader'].num_workers))
        elif event == 'return' and arg is not None:
            official_returns.append(arg)
        return trace

    official_argv = ['train.py','--base-model','GRU','--dropout','0.5' if dataset=='IEMOCAP' else '0.4',
        '--lr','0.0001','--batch-size','16','--graph_type','hyper','--epochs','0',
        '--graph_construct','direct','--multi_modal','--mm_fusion_mthd','concat_DHT',
        '--modals','avl','--Dataset',dataset,'--norm','BN','--testing']
    if dataset=='MELD': official_argv += ['--num_L','3','--num_K','3']
    print('Official argv:',json.dumps(official_argv),flush=True)
    print('Checkpoint:',str(checkpoint),'GPU:',torch.cuda.get_device_name(0),'cuDNN:',torch.backends.cudnn.enabled,flush=True)
    os.chdir(inputs); sys.path.insert(0,str(source)); sys.argv = official_argv
    torch.load, torch.nn.Module.load_state_dict, torch.optim.Adam.step = trusted_load, strict_load, forbidden_step
    try:
        sys.settrace(trace)
        with torch.no_grad(): namespace = runpy.run_path(str(entry),run_name='__main__')
    finally:
        sys.settrace(None)
        torch.load, torch.nn.Module.load_state_dict, torch.optim.Adam.step = old_load, old_state_load, old_step
    if len(official_returns)!=1 or audit['strict_load_calls']!=1:
        raise ValueError('Expected exactly one official testing invocation and one strict checkpoint load')
    labels=np.concatenate(labels_outputs); logp=np.concatenate(log_outputs)
    predicted=logp.argmax(1)
    returned=official_returns[0]
    if not np.array_equal(returned[2],labels) or not np.array_equal(returned[3],predicted):
        raise ValueError('Observed full outputs differ from the official evaluation return')
    if len(predictions)!= {'IEMOCAP':1623,'MELD':2610}[dataset]: raise ValueError('Wrong test sample count')
    if len({(r['dialogue_id'],r['utterance_index']) for r in predictions})!=len(predictions):
        raise ValueError('Duplicate valid utterance identity')
    if audit['forward_calls']!=len(audit['batches']): raise ValueError('Unexpected additional model forwards')
    model=model_holder['model']
    if tensor_state_hash(model)!=model_holder['before']: raise ValueError('Model parameters or buffers changed')
    for path,expected in original_inputs.items():
        if sha256(path)!=expected: raise ValueError('Checkpoint/data changed during inference')
    precision,recall,f1,support=precision_recall_fscore_support(labels,predicted,labels=list(range(len(names))),zero_division=0)
    report=[dict(class_id=k,class_name=name,precision=float(100*precision[k]),recall=float(100*recall[k]),
                 f1=float(100*f1[k]),support=int(support[k])) for k,name in enumerate(names)]
    metrics=dict(dataset=dataset,model='Official CMERC',accuracy=float(100*accuracy_score(labels,predicted)),
                 weighted_f1=float(100*f1_score(labels,predicted,average='weighted')),
                 macro_f1=float(100*f1_score(labels,predicted,average='macro')),valid_samples=len(labels),
                 paper_weighted_f1={'IEMOCAP':71.98,'MELD':66.85}[dataset],class_names=names)
    metrics['gap_percentage_points']=metrics['weighted_f1']-metrics['paper_weighted_f1']
    metrics['within_0p5_points']=abs(metrics['gap_percentage_points'])<=0.5
    if round(metrics['weighted_f1'],2)!=returned[4]: raise ValueError('Official rounded F1 differs from saved precision')
    write_csv(output/f'{dataset}_classification_report.csv',report)
    matrix=confusion_matrix(labels,predicted,labels=list(range(len(names))))
    write_csv(output/f'{dataset}_confusion_matrix.csv',[{'true_emotion':name,**{names[k]:int(value) for k,value in enumerate(row)}} for name,row in zip(names,matrix)])
    write_csv(output/f'{dataset}_predictions.csv',predictions)
    np.savez_compressed(output/f'{dataset}_predictions.npz',labels=labels,predictions=predicted,log_probabilities=logp,probabilities=np.exp(logp.astype(np.float64)))
    write_json(output/f'{dataset}_metrics.json',metrics)
    write_json(output/f'{dataset}_checkpoint_inventory.json',inventory)
    effective=vars(namespace['args'])
    manifest=dict(status='completed',dataset=dataset,started_at=started,completed_at=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(),pid=os.getpid(),python=platform.python_version(),torch=torch.__version__,
        numpy=np.__version__,sklearn=sklearn.__version__,cuda_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        cudnn_enabled=torch.backends.cudnn.enabled,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(0),
        official_argv=official_argv,effective_args=effective,audit=audit,model_state_unchanged=True,
        all_parameter_grads_none=all(p.grad is None for p in model.parameters()),original_inputs_unchanged=True,
        observer_line=line,official_rounded_accuracy=returned[1],official_rounded_weighted_f1=returned[4],
        classification_metrics=metrics,training=False,new_calibration_diagnostics=False,modality_removal=False)
    write_json(output/f'{dataset}_run_manifest.json',manifest)
    print('ACTUAL_RESULT',json.dumps(metrics),flush=True)


if __name__=='__main__':
    main()
