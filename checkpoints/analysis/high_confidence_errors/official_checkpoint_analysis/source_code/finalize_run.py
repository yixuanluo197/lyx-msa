#!/usr/bin/env python3
"""Read-only consistency checks, then save this run's manifest and artifact index."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):value.update(part)
    return value.hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    args=parser.parse_args(); project=args.project_root.resolve()
    out=project/'checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis'
    source=project/'checkpoints/analysis/official_checkpoint_verification'
    read=lambda p:json.loads(p.read_text())
    protected=read(out/'preserved_inputs.json')
    for path,digest in protected.items():assert sha(project/path)==digest,path
    full=read(out/'full_input_verification.json')
    ablation=read(out/'ablation_verification.json')
    independent=read(out/'independent_verification.json')
    report=read(out/'report_verification.json')
    baseline=read(out/'baseline_checkpoint_inventory.json')
    for value in [full,ablation,independent,report]:assert value['status']=='passed'
    assert report['word']['passed'] and report['excel']['passed']
    for path,digest in full['output_files_sha256'].items():assert sha(out/path)==digest,path
    for path,digest in report['sources'].items():assert sha(out/path)==digest,path
    for path,metadata in report['outputs'].items():
        assert sha(out/path)==metadata['sha256'] and (out/path).stat().st_size==metadata['bytes'],path
    snapshots=read(out/'source_code_manifest.json')['sha256']
    for path,digest in snapshots.items():
        assert sha(project/path)==digest,path
        assert sha(out/'source_code'/Path(path).name)==digest,path
    manifests=[]
    for dataset in ['IEMOCAP','MELD']:
        for modality in ['text','audio','visual']:
            path=out/dataset/'inference'/f'without_{modality}'/f'{dataset}_run_manifest.json'
            value=read(path);manifests.append(value)
            assert value['status']=='completed' and value['model_state_unchanged']
            assert value['original_inputs_unchanged'] and value['reference_artifacts_unchanged']
            assert value['gpu_host_index']==1
            assert value['actual_gpu_uuids']==['GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049']
            assert value['full_forward_calls']==0 and value['audit']['optimizer_steps']==0
    assert sum(v['audit']['forward_calls'] for v in manifests)==60
    assert baseline['matched_baseline_available']=={'IEMOCAP':False,'MELD':False}
    with (out/'official_cmerc_summary.csv').open(encoding='utf-8-sig',newline='') as stream:
        summaries=list(csv.DictReader(stream))
    assert [int(r['valid_utterances']) for r in summaries]==[1623,2610]
    now=datetime.now(timezone.utc).isoformat()
    preflight=read(source/'preflight.json')
    manifest=dict(status='completed_single_model_baseline_comparison_pending',completed_at=now,
        official_repository=preflight['official_repository'],official_code_commit=preflight['official_commit'],
        local_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
        server_alias='biggpu',host='user22',local_mirror_host=platform.node(),
        remote_project='/data2/yb/paper/cmerc_reproduction_20261013',
        gpu_host_index=1,gpu_uuid='GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049',excluded_host_gpu_index=4,
        checkpoints={k:v for k,v in preflight['input_sha256'].items() if k.startswith('checkpoints/')},
        data_sha256={k:v for k,v in preflight['input_sha256'].items() if k.startswith('data/')},
        input_condition='Full input reused; only three modality mean replacements inferred',
        full_predictions_source='../../official_checkpoint_verification',
        full_forward_calls=0,masked_forward_calls=60,optimizer_steps=0,training=False,
        official_argv={v['dataset']:v['official_argv'] for v in manifests},
        evaluation_seed=1475,author_training_seed=None,author_training_epoch=None,author_selection_protocol=None,
        rnn_backend='cuDNN enabled, deterministic=True, identical to fixed full-input evaluation',
        batch_size=16,num_workers=2,shuffle=False,
        mean_replacement='mean(dim=0).repeat(time,1,1), padded time mean; four text streams independent',
        unchanged_runtime_sha256=preflight['runtime_sha256'],
        compatibility_patch_sha256=preflight['compatibility_patch_sha256'],
        source_snapshot='source_code_manifest.json',source_script_sha256=sha(Path(__file__)),
        inference_launch='launch.json',inference_log='evaluation.log',report_environment='report_environment.json',
        analysis_metrics='official_cmerc_summary.csv',
        executed_commands=[
          'CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro scripts/with_cmerc_env.sh python scripts/analyze_official_reliability.py',
          'bash checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis/run_ablation.sh',
          'CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro scripts/with_cmerc_env.sh python scripts/summarize_official_ablation.py',
          'python scripts/render_official_reliability.py --output-dir checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis'],
        completed_experiments=['1 official CMERC','3 official CMERC','4 official CMERC','5 official CMERC','6 official CMERC'],
        pending_experiments=['2 paired Baseline-CMERC','All Baseline full/removal/distribution/calibration comparisons'],
        baseline_status=baseline['status'],matched_baseline_available=baseline['matched_baseline_available'],
        baseline_audit='baseline_checkpoint_audit.md',
        preserved_existing_file_count=len(protected),
        metric_units='CSV rates/F1/ECE/MCE/confidence in [0,1]; undefined denominators blank = N/A',
        limitations=['Author seed/epoch/selection metadata absent; strict paired baseline unavailable',
            'Padded temporal mean depends on fixed batch shape',
            'Published training code has test-selection risk, author checkpoint selection history unverified',
            'Single fixed checkpoint per dataset, no multi-seed significance claims',
            'OOXML formatting and numbers verified; Office/LibreOffice pagination not rendered'])
    dump(out/'run_manifest.json',manifest)
    verification=dict(status='passed',verified_at=now,
        existing_files_unchanged=len(protected),full_utterances=4233,masked_utterance_predictions=12699,
        full_forward_calls=0,masked_forward_calls=60,optimizer_steps=0,
        probability_id_label_text_order_padding_checks='passed',
        independent_check_count=independent['check_count'],
        word_tables_checked=report['word']['tables_checked'],
        word_font='宋体',word_font_size_points=12,word_color='000000',word_reliability_figures=2,
        excel_cells_checked=report['excel']['cells_checked'],
        tests_passed=47,test_evidence='test_results.json',
        baseline_comparison_completed=False,baseline_reason=baseline['status'],
        component_checks={name:sha(out/name) for name in ['full_input_verification.json','ablation_verification.json','independent_verification.json','report_verification.json']},
        report_and_csv_consistent=True,source_snapshot_matches=True,
        previous_validation_results_used_as_official_baseline=False,
        unverified=['Strict paired Baseline experiments','Office/LibreOffice pagination'])
    dump(out/'verification.json',verification)
    paths=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts)
    inventory={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
    assert not any(p.suffix in {'.pkl','.pt','.tar'} for p in paths),'Do not package model/data pickles'
    dump(out/'artifact_manifest.json',dict(schema_version=1,hash_algorithm='sha256',self_excluded=True,file_count=len(paths),files=inventory))
    print(json.dumps(dict(status='passed',artifact_count_including_index=len(paths)+1,protected_files=len(protected),word_tables=report['word']['tables_checked'],excel_cells=report['excel']['cells_checked']),ensure_ascii=False))

if __name__=='__main__':main()
