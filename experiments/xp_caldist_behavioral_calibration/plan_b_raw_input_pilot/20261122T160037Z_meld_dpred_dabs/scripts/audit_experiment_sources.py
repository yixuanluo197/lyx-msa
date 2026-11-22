#!/usr/bin/env python3
"""Read-only independence/source audit and historical probability replay.

No torch/pickle import, checkpoint loading, inference, raw dialogue parsing,
official Validation labels/text, Test, generation, Git or network operation.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def raw_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path,data):
    with Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)


def dump(path,data):
    with Path(path).open('x') as stream:
        json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10)


def softmax(text):
    p=list(map(float,json.loads(text)))
    if len(p)!=7 or not all(math.isfinite(x) and 0<=x<=1 for x in p) or abs(sum(p)-1)>1e-6:
        raise ValueError('Invalid historical full probability vector')
    return p


def replay_family(clean,variants,versions,family):
    index={(r['sample_id'],r['variant_id']):r for r in variants}
    if len(index)!=len(variants):raise ValueError('Duplicate historical version')
    replay=[]
    for c in clean:
        p=softmax(c['original_probabilities']);k=max(range(7),key=p.__getitem__)
        if k!=int(c['original_prediction']):raise ValueError('Original class/vector mismatch')
        deltas=[];reasons=[]
        for version in versions:
            r=index.get((c['sample_id'],version))
            if r is None:
                deltas.append(None);reasons.append('NOT_ATTEMPTED');continue
            if any(str(r[key])!=str(c[key]) for key in ('role','dialogue_id','utterance_id','utterance_index')):
                raise ValueError('Historical sample identity differs')
            q0=softmax(r['original_probabilities'])
            if q0!=p:raise ValueError('Historical clean vector changed')
            if str(r['eligible']).lower() not in ('1','0','true','false'):
                raise ValueError('Unknown recorded eligibility encoding')
            if str(r['eligible']).lower() in ('1','true'):
                q=softmax(r['perturbed_probabilities'])
                if r['inference_status']!='PREDICTED' or not close(q[k],r['perturbed_original_class_probability']):
                    raise ValueError('Historical eligible version invalid')
                d=p[k]-q[k]
                if r.get('signed_response','') and not close(d,r['signed_response']):
                    raise ValueError('Recorded signed response is inconsistent')
                deltas.append(d)
            else:
                if r['perturbed_probabilities']:raise ValueError('Ineligible historical probability unexpectedly present')
                deltas.append(None)
            reasons.append(r['reason'])
        complete=all(d is not None for d in deltas)
        signed=sum(deltas)/2 if complete else None
        absolute=sum(abs(d) for d in deltas)/2 if complete else None
        cancel=max(0.,absolute-abs(signed)) if complete else None
        opposite=deltas[0]*deltas[1]<0 if complete else None
        if complete and not close(cancel,min(map(abs,deltas)) if opposite else 0):
            raise ValueError('Independent cancellation identity failed')
        field='D_pred' if family=='semantic' else 'corruption_d_pred'
        if bool(c[field])!=complete or (complete and not close(c[field],signed)):
            raise ValueError('Historical pair presence/value differs')
        if family=='corruption' and complete and (not close(c['corruption_d_abs'],absolute) or not close(c['corruption_cancellation'],cancel)):
            raise ValueError('Historical corruption D_abs/cancellation mismatch')
        replay.append({'family':family,'sample_id':c['sample_id'],'role':c['role'],'dialogue_id':c['dialogue_id'],
            'original_correct':int(c['original_correct']),'complete_K2':int(complete),
            'delta_a':deltas[0] if deltas[0] is not None else '', 'delta_b':deltas[1] if deltas[1] is not None else '',
            'D_pred':signed if complete else '', 'D_abs':absolute if complete else '',
            'cancellation_absolute':cancel if complete else '',
            'cancellation_relative':cancel/absolute if complete and absolute>0 else '',
            'opposite_sign':int(opposite) if complete else '', 'version_a_reason':reasons[0],'version_b_reason':reasons[1]})
    return replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parent.parent)
    args=parser.parse_args();run=args.run_dir.resolve();pilot=run.parent
    old=pilot/'20261122T041838Z_meld_real_scoring'
    prior=pilot/'20261122T151649Z_meld_dpred_corruption'
    legacy=pilot/'20261120T143131Z'
    workspace=next(p for p in run.parents if (p/'CMERC').is_dir())
    public=workspace/'CMERC/experiments/xp_caldist_behavioral_calibration/plan_b_raw_input_pilot'/prior.name
    out=run/'results/audit';out.mkdir(parents=True,exist_ok=False)
    cache={};checks=[];bindings=[]
    def sha(path):
        key=str(Path(path).resolve())
        if key not in cache:cache[key]=raw_sha(key)
        return cache[key]
    def require(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
        if not condition:raise ValueError(name)
    def bind(path,expected,source):
        path=Path(path).resolve()
        require('hash:'+str(path),sha(path)==expected)
        bindings.append({'path':str(path),'expected_sha256':expected,'actual_sha256':sha(path),
                         'bytes':path.stat().st_size,'binding_source':source,'method':'streaming SHA256 only; not deserialized'})

    publication=read(public/'PUBLICATION_MANIFEST.json')
    for relative,expected in publication['sha256'].items():bind(public/relative,expected,'public publication manifest')
    bind(prior/'SHA256SUMS.csv',publication['local_aggregate_and_private_manifest_sha256'],'public-to-local manifest anchor')
    require('prior_final_verification_binds_same_manifest',read(prior/'verification.json')['sha256_manifest']==sha(prior/'SHA256SUMS.csv'))
    private_manifest={r['path']:r for r in rows(prior/'SHA256SUMS.csv')}
    # Only probability artifacts needed here; do not open historical raw reviews or Validation records.
    relevant_prior=['EXPERIMENT_PROTOCOL.json','provenance/INPUT_AUDIT.json',
        'results/text_comparison/inference_verification.json','results/text_comparison/score_inputs.csv',
        'results/text_comparison/corruption_variants.csv','results/text_comparison/family_response_summary.csv',
        'results/text_comparison/paired_behavior.csv']
    for relative in relevant_prior:
        bind(prior/relative,private_manifest[relative]['sha256'],'prior local SHA256SUMS')
    old_anchors=read(prior/'provenance/INPUT_AUDIT.json')['sha256_relative_to_pilot_parent']
    for relative,expected in old_anchors.items():
        if not relative.startswith(old.name+'/'):
            continue
        suffix=relative[len(old.name)+1:]
        if (suffix.startswith(('local_only/base_data/','local_only/text_model/')) or
            suffix in {'local_only/score_data.pt','local_only/score_audio_features.pt',
                'local_only/score_text_features.pt','local_only/score_text_features.manifest.json',
                'SPLIT_LOCK.json','configs/REAL_SCORING_PROTOCOL.json',
                'results/score_prediction_verification.json','results/score_inputs.csv',
                'results/score_transformations.csv','results/scoring/scoring_report.json',
                'results/scoring/fitted_parameters.json','scripts/downstream_branch.py'}):
            bind(pilot/relative,expected,'immutable previous INPUT_AUDIT chain')

    split_lock=read(old/'SPLIT_LOCK.json')
    dialogue_path=old/'local_only/SPLIT_DIALOGUES.csv'
    utterance_path=old/'local_only/SPLIT_UTTERANCES.csv'
    bind(dialogue_path,split_lock['hashes'][dialogue_path.name],'old frozen role lock')
    bind(utterance_path,split_lock['hashes'][utterance_path.name],'old frozen role lock')
    manifest=rows(dialogue_path);roles={r['dialogue_id']:r['role'] for r in manifest}
    require('1038_unique_train_dialogues',len(roles)==len(manifest)==1038)
    counts=Counter(roles.values())
    require('historical_role_counts',counts==Counter(base_fit=656,base_inner_dev=72,score_fit=155,score_eval=155))
    utterances=rows(utterance_path)
    require('9989_unique_train_utterances',len(utterances)==9989 and len({r['sample_id'] for r in utterances})==9989)

    data_lock=read(old/'provenance/data_lock.json')
    for relative,expected in data_lock['output_hashes'].items():bind(old/relative,expected,'physically isolated old data export')
    training=read(old/'local_only/text_model/training_config.json')
    text_lock=read(old/'local_only/text_model/best_model_lock.json')
    p=training['protocol'];protocol_sha=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    require('text_training_protocol_hash',protocol_sha==training['protocol_sha256']==text_lock['protocol_sha256'])
    fit=set(map(str,p['fit_dialogue_ids']));dev=set(map(str,p['inner_dev_dialogue_ids']))
    require('text_fit_dev_exact_base_roles',fit=={d for d,r in roles.items() if r=='base_fit'} and dev=={d for d,r in roles.items() if r=='base_inner_dev'} and not fit&dev)
    require('text_selection_is_internal_only',text_lock['official_validation_used_for_selection'] is False and text_lock['test_accessed'] is False)
    bind(old/'local_only/text_model/best_classifier/model.safetensors',text_lock['model_sha256'],'completed text encoder lock')
    text_metadata=read(old/'local_only/base_data/features/text_features.manifest.json')
    require('base_text_feature_model_same_encoder',text_metadata['model_sha256']==text_lock['model_sha256'])
    require('base_export_contains_no_official_validation_rows',text_metadata['split_statistics']['validation']['utterances']==0)
    bind(old/'local_only/base_data/features/text_features.pt',text_metadata['feature_file_sha256'],'text extraction manifest')
    for name,expected in text_metadata['tokenizer_hashes'].items():
        bind(old/'local_only/text_model/best_classifier'/name,expected,'same exported text tokenizer')

    for stage in ('m3net','cmerc'):
        stage_root=old/'local_only/base_data/runs'/stage
        completion=read(stage_root/'completion.json');provenance=read(stage_root/'provenance.json')
        require(stage+':complete_and_new_protocol',completion['status']=='COMPLETE' and completion['protocol_sha256']==sha(old/'configs/REAL_SCORING_PROTOCOL.json'))
        branch=provenance['branch']
        require(stage+':no_score_groups_in_model_fit_selection',branch['scoring_groups_loaded'] is False and branch['fit_role']=='base_fit' and branch['selection_role']=='base_inner_dev')
        require(stage+':restricted_fit_dev_counts',provenance['restricted_bundle']['fit_dialogues']==656 and provenance['restricted_bundle']['inner_dev_dialogues']==72)
        bind(stage_root/'best.pt',completion['best_checkpoint_sha256'],stage+' frozen best checkpoint')
        bind(stage_root/'last.pt',completion['last_checkpoint_sha256'],stage+' completed last checkpoint')
        for filename,expected in provenance['inputs'].items():bind(filename,expected,stage+' cached modality/role/protocol provenance')
        if stage=='cmerc':bind(old/'local_only/base_data/new_difficulty.pt',provenance['difficulty_sha256'],'CMERC upstream M3Net-derived difficulty')

    old_inference=read(old/'results/score_prediction_verification.json')
    for name,expected in old_inference['outputs'].items():bind(old/'results'/name,expected,'old semantic probability inference')
    new_inference=read(prior/'results/text_comparison/inference_verification.json')
    for name,expected in new_inference['outputs'].items():bind(prior/'results/text_comparison'/name,expected,'old corruption probability inference')
    require('both_old_forwards_frozen',old_inference['model_state_before']==old_inference['model_state_after'] and new_inference['model_state_before']==new_inference['model_state_after'])
    require('old_corruption_reused_clean_semantics',new_inference['semantic_probabilities_reused'] and new_inference['clean_semantic_rows_bit_exact'])
    semantic_clean=rows(old/'results/score_inputs.csv');corrupt_clean=rows(prior/'results/text_comparison/score_inputs.csv')
    require('old_clean_full_role_counts',len(semantic_clean)==len(corrupt_clean)==2976)
    clean_by_id={r['sample_id']:r for r in semantic_clean}
    require('clean_identity_unchanged_across_old_families',set(clean_by_id)=={r['sample_id'] for r in corrupt_clean})
    for r in corrupt_clean:
        original=clean_by_id[r['sample_id']]
        require('old_clean_control:'+r['sample_id'],all(original[k]==r[k] for k in original))
    score_report=read(old/'results/scoring/scoring_report.json')
    for filename,expected in score_report['input_sha256'].items():bind(filename,expected,'old reliability scorer inputs')
    parameters=read(old/'results/scoring/fitted_parameters.json')
    old_score_fit={r['sample_id'] for r in semantic_clean if r['role']=='score_fit'}
    require('all_old_score_fit_labels_already_used_for_scoring_fit',set(parameters['weak_baseline_all_fit']['fit_sample_ids'])==old_score_fit and len(old_score_fit)==1569)
    prior_evaluated={r['dialogue_id'] for r in semantic_clean}
    require('all310_old_scoring_dialogues_exposed',prior_evaluated=={d for d,r in roles.items() if r in ('score_fit','score_eval')})
    unseen=set(roles)-fit-dev-prior_evaluated
    require('fresh_independent_train_dialogues_zero',len(unseen)==0)
    legacy_report=legacy/'BASELINE_REPORT.md'
    sha(legacy_report);sha(legacy/'verification.json')
    legacy_text=legacy_report.read_text()
    require('official_validation_already_analysed_from_existing_report',bool(re.search(r'\|MELD\|67137\|1109\|',legacy_text)))
    old_checks=read(legacy/'verification.json')['checks']
    known_validation={re.match(r'^MELD:validation:([^:]+):shape$',r.get('check','')).group(1)
        for r in old_checks if re.match(r'^MELD:validation:([^:]+):shape$',r.get('check','')) and r['passed']}
    require('114_validation_dialogues_known_from_prior_metadata',len(known_validation)==114)

    replay=replay_family(semantic_clean,rows(old/'results/score_transformations.csv'),('paraphrase_a','paraphrase_b'),'semantic')
    replay+=replay_family(corrupt_clean,rows(prior/'results/text_comparison/corruption_variants.csv'),('corruption_a','corruption_b'),'corruption')
    by_pair={(r['family'],r['sample_id']):r for r in replay}
    common={sid for sid in clean_by_id if by_pair['semantic',sid]['complete_K2'] and by_pair['corruption',sid]['complete_K2']}
    summaries=[]
    for family in ('semantic','corruption'):
        for role in ('score_fit','score_eval'):
            for correctness in ('all','0','1'):
                selected=[r for r in replay if r['family']==family and r['role']==role and r['complete_K2'] and (correctness=='all' or r['original_correct']==int(correctness))]
                count=len(selected);relative=[r['cancellation_relative'] for r in selected if r['cancellation_relative']!='']
                mass=sum(r['D_abs'] for r in selected)
                summaries.append({'family':family,'role':role,'original_correct':correctness,'complete_K2':count,
                    'opposite_sign_pairs':sum(r['opposite_sign'] for r in selected),'opposite_sign_fraction':sum(r['opposite_sign'] for r in selected)/count if count else '',
                    'mean_D_pred':statistics.mean(r['D_pred'] for r in selected) if count else '',
                    'mean_D_abs':statistics.mean(r['D_abs'] for r in selected) if count else '',
                    'mean_cancellation_absolute':statistics.mean(r['cancellation_absolute'] for r in selected) if count else '',
                    'mean_sample_relative_cancellation':statistics.mean(relative) if relative else '',
                    'cancellation_total_mass_fraction':sum(r['cancellation_absolute'] for r in selected)/mass if mass else '',
                    'relative_denominator_zero_pairs':sum(r['D_abs']==0 for r in selected)})
    aggregate_checks=[]
    for r in rows(prior/'results/text_comparison/family_response_summary.csv'):
        selected=[v for v in replay if v['family']==r['family'] and v['role']==r['role'] and v['complete_K2'] and
                  (r['scope']=='family_available' or v['sample_id'] in common) and
                  (r['original_correct']=='all' or v['original_correct']==int(r['original_correct']))]
        require('old_aggregate_count',len(selected)==int(r['n']))
        for metric in ('D_pred','D_abs'):
            if selected:
                require('old_aggregate_mean:'+metric,close(statistics.mean(v[metric] for v in selected),r['mean_'+metric]))
                require('old_aggregate_median:'+metric,close(statistics.median(v[metric] for v in selected),r['median_'+metric]))
        aggregate_checks.append({'scope':r['scope'],'role':r['role'],'family':r['family'],'correctness':r['original_correct'],'n':len(selected),'status':'PASS'})
    exposure=[{'dialogue_id':d,'historical_role':role,'exposure':'supervised_fit' if role=='base_fit' else 'model_selection' if role=='base_inner_dev' else 'reliability_fit_and_prior_analysis' if role=='score_fit' else 'prior_outcome_evaluation',
               'fresh_independent':0,'allowed_new_generation_scoring':int(role=='score_fit')} for d,role in sorted(roles.items())]
    write_csv(out/'dialogue_exposure.csv',exposure)
    write_csv(out/'hash_bindings.csv',bindings)
    write_csv(out/'historical_pair_replay.csv',replay)
    write_csv(out/'historical_cancellation_summary.csv',summaries)
    write_csv(out/'historical_aggregate_replay_checks.csv',aggregate_checks)
    summary={'status':'PASS_FOR_EXPLORATORY_ONLY','fresh_independent_dialogues':0,'historical_train_dialogues':1038,
        'historical_role_counts':dict(counts),'official_validation_previously_analysed':{'dialogues':114,'utterances':1109,'evidence':'prior aggregate reports and verification metadata only; no Validation text/labels reopened'},
        'new_allowed_pool':'historical score_fit only;155 dialogues; engineering20/pilot_fit75/pilot_eval60',
        'old_score_eval_in_new_generation_scoring':False,'published_delivery_files_verified':len(publication['sha256']),
        'published_source_commit':publication['source_commit'],'requested_publication_commit':'2bf3720',
        'commit_object_verified_by_this_script':False,'commit_note':'No Git operation in this audit lane. Root must separately bind2bf3720 to the checked publication tree.',
        'unique_hashed_assets':len(cache),'hash_binding_checks':len(bindings),'check_count':len(checks),
        'cancellation_summary':[r for r in summaries if r['original_correct']=='all'],
        'corruption_and_semantic_aggregate_groups_verified':len(aggregate_checks),
        'model_checkpoint_deserialization':False,'model_inference':False,'raw_dialogue_parsing':False,
        'official_validation_text_or_labels_opened':False,'test_accessed':False,'checked_at':datetime.now(timezone.utc).isoformat()}
    report=['# 独立性与冻结来源审计','',
        '**结论：没有可用于本轮的新独立Dialogue；只能按探索性Pilot执行。** 这限制研究结论，不表示冻结模型或数值链路损坏。新生成、模型推理及评分须只使用历史score_fit155个Dialogue，按运行前固定规则分成20工程/75 pilot_fit/60 pilot_eval；旧score_eval不进入新流程。','',
        '## 历史暴露核对','',
        '| 历史角色 | Dialogue | 已有用途 | 新独立性 |','|---|---:|---|---|',
        '| base_fit | 656 | 文本监督训练及下游模型梯度 | 无 |',
        '| base_inner_dev | 72 | 文本及下游模型早停/最佳轮选择 | 无 |',
        '| score_fit | 155 | 1569条clean结果已分析，曾用于可靠性逻辑回归拟合 | 无 |',
        '| score_eval | 155 | 1407条结果已评价，后续研究方向已受旧结果影响 | 无，本轮排除 |','',
        '原1038个Train Dialogue完整覆盖上述四组。文本training_config中的Fit/Dev ID与角色清单精确一致，旧2976条clean覆盖全部310个评分Dialogue，原可靠性模型的fit_sample_ids覆盖全部旧score_fit样本。更换本轮角色名称、重新盲化改写者或选择新目标，不会消除既有结果暴露。','',
        '旧Plan B报告已公布MELD官方Validation全部1109条指标；114个Dialogue的既有特征核验记录也已存在。本审计只读取旧报告及核验元数据，未重新打开其文本、情绪标签或执行推理；Test未触及。','',
        '## 来源与哈希链','',
        f'- 已逐文件核验公开交付目录的{len(publication["sha256"])}个SHA-256；公开清单绑定本地SHA256SUMS.csv，且与上一轮verification中的manifest哈希一致。',
        '- 清单source_commit记录的是实验起始源版本，不能混称发布提交2bf3720。本分工未执行Git；2bf3720对象与该公开目录的对应须由root另行核验。',
        '- 核验物理分离的base/score数据及音频缓存、文本四层特征与其model/tokenizer哈希、文本最佳权重、M3Net与CMERC最佳/末轮checkpoint、CMERC难度文件及其输入清单。checkpoint仅流式计算哈希，未加载。',
        '- 两个下游阶段的provenance均声明且绑定base_fit/base_inner_dev，评分组未进入该冻结模型的梯度或模型选择。此“模型训练隔离”与“研究者未见过新评估结果”是不同条件；本轮仅前者成立。',
        '- 上游旧共享Train/Validation数据容器不重新打开；其历史来源只保留已有清单引用。当前实际核验的dataset.pt属于旧物理隔离Train角色缓存。','',
        '## 旧概率的抵消复算','',
        '固定clean argmax c，逐版d_k=p_clean(c)−p_k(c)；D_pred=(d_a+d_b)/2，D_abs=(|d_a|+|d_b|)/2。抵消量=D_abs−|D_pred|；异号时独立核验其等于min(|d_a|,|d_b|)，同号时为0。相对抵消为每对抵消量/D_abs；D_abs=0时保留NA。','',
        '| 家族 | 历史Role | 有效K2 | 异号对 | 平均D_pred | 平均D_abs | 平均绝对抵消 | 平均样本相对抵消 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['original_correct']=='all':
            f=lambda x:'NA' if x=='' else f'{x:.8g}'
            report.append(f"| {r['family']} | {r['role']} | {r['complete_K2']} | {r['opposite_sign_pairs']} | {f(r['mean_D_pred'])} | {f(r['mean_D_abs'])} | {f(r['mean_cancellation_absolute'])} | {f(r['mean_sample_relative_cancellation'])} |")
    report+=['',
        f'两类旧记录共{len(aggregate_checks)}组已有汇总的样本量、D_pred/D_abs均值及中位数均与完整概率复算一致；语义与corruption共同可用子集也逐组核对。所有计划样本和缺失保留在historical_pair_replay.csv。此处旧score_eval仅为历史审计来源，不用于新生成或新评分。','',
        '相对抵消的“逐样本平均”与“总抵消量/总D_abs”不是同一个量，CSV同时保留两者。不以跨样本平均接近0代替样本内部异号抵消；不将抵消现象单独当作D_abs有可靠性收益的证据。','',
        '## 交付与执行边界','',
        '- results/audit/audit_summary.json、checks.json：独立性结论及逐项核对。',
        '- hash_bindings.csv：实际路径、原期望哈希、重新计算哈希及绑定来源。',
        '- dialogue_exposure.csv：1038个Train Dialogue的既有用途与本轮可用池标记。',
        '- historical_pair_replay.csv、historical_cancellation_summary.csv、historical_aggregate_replay_checks.csv：完整旧概率重算、分组统计及与旧汇总对照。',
        '- 审计没有生成文本、读取原始媒体、加载模型、执行训练/推理、接触Test或新的官方Validation文本/标签，也没有Git提交或上传。','']
    (run/'INDEPENDENCE_AUDIT.md').write_text('\n'.join(report))
    # Rehash all actually checked assets after writing only this audit's outputs.
    require('all_original_hashed_assets_unchanged',all(raw_sha(path)==expected for path,expected in cache.items()))
    summary['check_count']=len(checks)
    dump(out/'audit_summary.json',summary);dump(out/'checks.json',checks)
    verification={'status':'PASS_EXPLORATORY_ONLY','fresh_independent_dialogues':0,'check_count':len(checks),
        'checks_passed':all(r['passed'] for r in checks),'all_checked_original_inputs_unchanged':True,
        'source_sha256':cache,'script_sha256':raw_sha(__file__),
        'output_sha256':{str(p.relative_to(run)):raw_sha(p) for p in out.iterdir() if p.is_file()},
        'report_sha256':raw_sha(run/'INDEPENDENCE_AUDIT.md')}
    dump(out/'verification.json',verification)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
