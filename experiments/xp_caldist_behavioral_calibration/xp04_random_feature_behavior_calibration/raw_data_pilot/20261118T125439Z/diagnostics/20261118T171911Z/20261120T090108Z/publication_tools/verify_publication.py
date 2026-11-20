"""Static publication check only; never trains, infers or opens private tensor files."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-source',type=Path);args=parser.parse_args()
    checks=[]
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    manifest=json.loads((ROOT/'publication_manifest.json').read_text())
    entries=manifest['files']+manifest['generated_files']
    for row in entries:
        p=ROOT/row['file'];check('public_hash:'+row['file'],p.is_file() and sha(p)==row['public_sha256'])
        if args.local_source and row.get('local_source'):
            check('local_source_hash:'+row['file'],sha(args.local_source/row['local_source'])==row['local_source_sha256'])
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    check('complete_export_coverage',actual-{r['file'] for r in entries}==({'publication_manifest.json','publication_verification.json'} & actual))
    def inspect(value):
        if isinstance(value,dict):
            check('no_raw_data_fields',not {'original_text','processed_text','raw_text','sentences','input_ids','targets'}.intersection(value))
            for v in value.values():inspect(v)
        elif isinstance(value,list):
            for v in value:inspect(v)
    for name in sorted(actual):
        p=ROOT/name
        check('no_private_assets:'+name,'local_only' not in p.relative_to(ROOT).parts and p.suffix not in {'.pt','.bin','.pkl','.npz','.npy','.safetensors','.wav','.mp4','.zip','.gz','.hdf5','.bpe'} and 'selected_fit_manifest' not in p.name)
        check('bounded_file_size:'+name,p.stat().st_size<5*1024*1024)
        if p.suffix in {'.pdf','.png'}:continue
        text=p.read_text()
        check('local_identity_redacted:'+name,not re.search(r'/(?:home|data[0-9]+)/|\buser[0-9]+\b|GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}',text))
        check('no_credentials:'+name,not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',text))
        check('no_utterance_ids:'+name,not re.search(r'Ses0[1-5][FM]_(?:impro|script)',text))
        if p.suffix=='.py':ast.parse(text)
        elif p.suffix=='.json':inspect(json.loads(text))
        elif p.suffix=='.csv':
            rows=list(csv.DictReader(io.StringIO(text)))
            check('no_raw_csv_columns:'+name,not rows or not {'text','original_text','processed_text','input_ids','targets'}.intersection(rows[0]))
    original=json.loads((ROOT/'verification.json').read_text())
    check('original_verification_pass',original['status']=='PASS' and original['checks_passed']==2244 and original['original_input_file_count']==467 and original['all_original_allowed_files_unchanged'])
    summary=json.loads((ROOT/'results/extension_summary.json').read_text())
    execution=json.loads((ROOT/'results/execution.json').read_text())
    check('all_jobs_completed',execution['status']=='COMPLETED')
    for ds,steps in [('IEMOCAP',2000),('MELD',5000)]:
        p=ROOT/'training_records/formal'/ds
        complete=json.loads((p/'completion.json').read_text())
        decision=json.loads((p/'decision.json').read_text())
        evals=json.loads((p/'evaluations.json').read_text())
        by_step={r['global_step']:r for r in evals}
        relative=(by_step[1500]['mse']-by_step[2000]['mse'])/by_step[1500]['mse']
        check(ds+':strict_budget_rule',decision['continue_to_5000']==(relative>.05) and steps==(5000 if relative>.05 else 2000))
        check(ds+':actual_updates',complete['global_step']==steps and complete['new_optimizer_steps']==steps-500 and complete['status']=='COMPLETE')
        with (p/'optimizer_steps.csv').open() as f:check(ds+':continuous_updates',[int(r['global_step']) for r in csv.DictReader(f)]==list(range(501,steps+1)))
        check(ds+':feature_condition_failed',not summary[ds]['rmse_necessary_condition_pass'] and all(r>1e-4 for r in complete['final_layer_rmse']))
        check(ds+':all_layer_metrics',all(math.isclose(math.sqrt(m),r,rel_tol=1e-12) for m,r in zip(complete['final_layer_mse'],complete['final_layer_rmse'])))
        for phase in ['smoke','formal']:check(ds+':exit0:'+phase,execution['phases'][phase][ds]['exit_code']==0)
    if args.local_source:
        for row in manifest['files']:
            source=args.local_source/row['local_source'];target=ROOT/row['file']
            if target.suffix=='.csv' and Path(row['file']).parts[0] in {'results','training_records'}:
                check('scientific_csv_unchanged:'+row['file'],list(csv.DictReader(io.StringIO(source.read_text())))==list(csv.DictReader(io.StringIO(target.read_text()))))
        source_summary=json.loads((args.local_source/'results/extension_summary.json').read_text())
        check('all_summary_values_unchanged',source_summary==summary)
    for link in re.findall(r'\]\(([^)]+)\)',(ROOT/'README.md').read_text()):
        check('README_link:'+link,link=='publication_verification.json' or (ROOT/link).exists())
    result=dict(verified=True,created_at=datetime.now(timezone.utc).isoformat(),checks_passed=len(checks),files_published=len(entries)+2,publication_manifest_sha256=sha(ROOT/'publication_manifest.json'),local_source_compared=bool(args.local_source),private_assets_uploaded=False,training_or_inference_rerun=False,feature_compatibility='BOTH_DATASETS_RMSE_CONDITION_FAILED',diagnostic_delivery_pass_is_not_feature_compatibility=True)
    (ROOT/'publication_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
