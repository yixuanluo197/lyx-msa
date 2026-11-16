"""Summarize measured local run resources; shared timing is not a benchmark."""
import csv
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    queue=json.loads((ROOT/'logs/queue_complete.json').read_text())
    assert queue['all_passed'] and len(queue['runs'])==12
    samples=[json.loads(v) for v in (ROOT/'logs/gpu_resource_samples.jsonl').read_text().splitlines()]
    rows=[]
    for run in queue['runs']:
        path=ROOT/'runs'/run['dataset']/f"seed{run['seed']}"/f"block{run['block']}"/'complete.json'
        info=json.loads(path.read_text())
        assert info['status']=='completed' and info['smoke'] is False
        begin=datetime.datetime.fromisoformat(run['started_at']);end=datetime.datetime.fromisoformat(run['ended_at'])
        selected=[s['gpus'][str(run['gpu'])] for s in samples if begin<=datetime.datetime.fromisoformat(s['utc'])<=end]
        row={k:run[k] for k in ['dataset','seed','block','modality','gpu','gpu_uuid','pid','started_at','ended_at','exit_code']}
        row.update(server='local',utterances=info['utterances'],target_forwards=info['target_forwards'],elapsed_seconds=info['seconds'],peak_allocated_mib=info['peak_allocated_mib'],peak_reserved_mib=info['peak_reserved_mib'],resource_samples=len(selected),mean_sampled_gpu_utilization=sum(v['utilization'] for v in selected)/len(selected) if selected else None,minimum_sampled_free_mib=min(v['free_mib'] for v in selected) if selected else None,weights_unchanged=info['weights_unchanged'],shared_timing_not_benchmark=True)
        rows.append(row)
    output=ROOT/'results/gpu_resource_summary.csv'
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'completed_runs':len(rows),'target_forwards':sum(v['target_forwards'] for v in rows),'output':str(output)}))

if __name__=='__main__':main()
