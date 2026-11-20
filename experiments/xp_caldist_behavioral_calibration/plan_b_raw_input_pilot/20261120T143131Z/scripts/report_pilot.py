#!/usr/bin/env python3
"""Report actual frozen-pilot results; no model, selection, fitting or retuning."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'results/analysis'
NAMES = {'neutral_append': '中性追加', 'semantic_paraphrase': '语义改写',
         'pitch_shift': '音高', 'timbre_tilt': '频谱倾斜'}


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def num(value):
    return float(value) if value not in [None, '', 'None'] else None


def fmt(value):
    return 'NA' if value in [None, '', 'None'] else f'{float(value):.6g}'


def yes(value):
    return str(value).lower() in ['true', '1']


def main():
    verification = json.loads((ANALYSIS / 'analysis_verification.json').read_text())
    if verification['status'] != 'COMPLETE' or verification['input_records'] != 2400:
        raise ValueError('Report requires completed fixed two-dataset pilot analysis')
    strict = rows(ANALYSIS / 'condition_summary.csv')
    exploratory = rows(ANALYSIS / 'audio_exploratory_condition_summary.csv')
    main_behavior = rows(ANALYSIS / 'behavior_features_main.csv')
    audio_behavior = rows(ANALYSIS / 'audio_exploratory_behavior_features.csv')
    all_behavior = rows(ANALYSIS / 'behavior_features_all_complete.csv')
    comparisons = rows(ANALYSIS / 'correct_wrong_comparison.csv') + rows(ANALYSIS / 'audio_exploratory_correct_wrong_comparison.csv')
    matched = rows(ANALYSIS / 'within_dialogue_confidence_comparison.csv') + rows(ANALYSIS / 'audio_exploratory_within_dialogue_confidence_comparison.csv')
    effective = [r for r in strict if r['perturbation_type'] in ['neutral_append', 'semantic_paraphrase']] + exploratory
    effective.sort(key=lambda r: (r['dataset'], list(NAMES).index(r['perturbation_type'])))
    lines = ['# 原始文本与音频干扰 Pilot 实测报告', '',
        '两套新单 Seed Baseline 完成并冻结后，在每个数据集固定120条真正 Validation 目标话语上生成4个文本、6个音频版本，共2400个预设干扰条件。每次仅替换目标话语的一种模态，使用完整 Dialogue 推理；新文本和音频编码器及 CMERC 均冻结。所有失败与不确定版本保留，不按响应大小筛选。', '',
        '文本主分析仅使用两次有署名 AI 上下文审核及自动输入质检均通过的完整家族，仍缺少人工语义复核。音频无人类听审结论，因此标签保持主分析为空；下文音频统计是信号质检通过的独立探索分析，只按干净输入的原预测正确性分组，未计算其扰动后准确率、校准误差或 Correct→Wrong/Wrong→Correct。', '',
        '## 样本与质量控制', '',
        '|数据集|固定目标|Dialogue|原正确/错误|文本PASS/UNCERTAIN/FAIL|音频客观QC通过/失败|推理成功/失败|',
        '|---|---:|---:|---:|---|---:|---:|']
    quality = []
    for ds in ['IEMOCAP', 'MELD']:
        p = ROOT / 'results/pilot' / ds
        local = ROOT / 'local_only/pilot' / ds
        samples = rows(p / 'sample_manifest.csv')
        inference = json.loads((p / 'inference_both.json').read_text())
        review = json.loads((ROOT / 'provenance' / ('text_input_review_' + ds + '.json')).read_text())
        transformations = [json.loads(line) for line in (local / 'audio_transformations.jsonl').open() if line.strip()]
        qc = sum(r['signal_qc']['automatic_signal_qc_pass'] for r in transformations)
        correct = sum(int(r['original_correct']) for r in samples)
        counts = review['semantic_status_counts']
        lines.append(f"|{ds}|{len(samples)}|{len({r['dialogue_id'] for r in samples})}|{correct}/{len(samples)-correct}|{counts.get('PASS',0)}/{counts.get('UNCERTAIN',0)}/{counts.get('FAIL',0)}|{qc}/{len(transformations)-qc}|{inference['successful']}/{inference['failed']}|")
        controls = rows(p / 'clean_reextraction_controls_both.csv')
        quality.append(dict(dataset=ds, targets=len(samples), dialogues=len({r['dialogue_id'] for r in samples}),
                            correct=correct, error=len(samples)-correct, text_review_counts=counts,
                            audio_signal_qc_pass=qc, audio_signal_qc_fail=len(transformations)-qc,
                            clean_controls=len(controls),
                            clean_feature_max_abs=max(float(r['clean_feature_max_abs']) for r in controls),
                            clean_probability_max_abs=max(float(r['clean_probability_max_abs']) for r in controls)))
    lines += ['', '干净输入重提取的阴性控制如下。编码器预先提取全部120×2个干净对照；消费者只对存在有效干扰版本的目标模态运行概率对照，因此数量可能小于240。Pilot原定allclose门槛为特征atol/rtol=1e-4/1e-4、概率atol/rtol=1e-4/1e-4，同时要求预测类别不变；先前固定Train工程smoke另使用更严格的概率atol=1e-6。这里报告实际误差，不宣称重提取逐位相等。', '',
              '|数据集|已执行目标模态概率对照|最大特征绝对差|最大概率绝对差|',
              '|---|---:|---:|---:|']
    for q in quality:
        lines.append('|'+ '|'.join([q['dataset'],str(q['clean_controls']),fmt(q['clean_feature_max_abs']),fmt(q['clean_probability_max_abs'])])+'|')
    lines += ['', '这120条为情绪、Dialogue及原预测状态分层样本，整体正确率49.17%不是总体性能。全 Validation 的 Baseline 指标见 BASELINE_REPORT.md；IEMOCAP 527条、MELD 1109条的完整清洁指标与Pilot样本严格分开。各家族质检通过的子集也可能不同，条件均值不应当作完全配对的模态优势比较。', '',
        '音高为±0.5、±1半音，FFmpeg采样时钟变换与反向时长补偿；音色为0.75/1.5 dB每倍频程频谱倾斜、上限3/6 dB。统一单声道16kHz，末端补齐/裁切记录原始长度误差，恢复RMS后限制峰值；门槛为RMS变化≤0.5dB、无输出削波、原始长度误差≤5%，最终采样点数不变。质检失败不更改参数重做。', '',
        '文本追加受原句token数约束，同一类型的两个版本实际Token ID不同，无原文截断；未完成句、叹词、语用强度或否定含义无法确认的内容明确标记，不强行PASS。完整文本、上下文、处理后音频、模型和特征仅存local_only/。', '',
        '## 正式行为定义', '',
        '`μ = mean_k[1(argmax p_k ≠ argmax p_0) × max(p_k)]`；`δ = abs(max(p_0) − mean_k max(p_k))`。翻转率另报，不等同μ；平均绝对置信度变化另报，不等同δ。JS使用自然对数，逐个版本与原分布比较后取均值。中性追加/改写/音高/音色的K分别2/2/4/2，不将不同家族拼成随机重复。', '',
        '## 各家族实测行为', '',
        'C/E表示原正确/原错误；文本为标签保持审核主分析，音频为未确认情绪保持的QC探索。以下均为选定子集的样本均值。', '',
        '|数据集|条件|C/E样本|C/E翻转率|C/E μ|C/E δ|C/E JS|',
        '|---|---|---:|---|---|---|---|']
    for r in effective:
        values = []
        for metric in ['flip_rate', 'mu', 'delta', 'js_mean']:
            values.append(fmt(r.get('correct_' + metric)) + ' / ' + fmt(r.get('error_' + metric)))
        lines.append('|' + '|'.join([r['dataset'], NAMES[r['perturbation_type']],
                      str(r['correct_samples']) + '/' + str(r['error_samples']), *values]) + '|')
    lines += ['', '## 正确与错误的差异及依赖控制', '',
        '下表为错误减正确的均值差及95% Dialogue配对Bootstrap区间（2000次，共享同数据集的Dialogue抽样）。区间只控制同Dialogue内依赖；跨组区间本身不消除Dialogue混杂。全部为多条件探索性关联，未作多重检验后的确认性显著结论。', '',
        '|数据集|条件|指标|E−C|95%区间|C/E样本|', '|---|---|---|---:|---|---:|']
    effective_keys = {(r['dataset'], r['perturbation_type'], r['analysis_scope']) for r in effective}
    for r in comparisons:
        if (r['dataset'], r['perturbation_type'], r['analysis_scope']) not in effective_keys or r['confidence_bin'] != 'all' or r['metric'] not in ['mu', 'delta', 'js_mean']:
            continue
        lines.append('|' + '|'.join([r['dataset'], NAMES[r['perturbation_type']], r['metric'],
                      fmt(r['error_minus_correct']), '['+fmt(r['ci_low'])+', '+fmt(r['ci_high'])+']',
                      r['correct_samples']+'/'+r['error_samples']]) + '|')
    lines += ['', '进一步在相同Dialogue和原置信度0.1宽度区间内匹配正确/错误话语；只有两组均存在的单元参与，单元固定等权，再按Dialogue重复抽样。它是更窄的条件关联，仍不是可靠性预测增量或校准收益的验证。', '',
        '|数据集|条件|匹配Dialogue|匹配单元|匹配样本/候选|JS E−C及95%区间|状态|',
        '|---|---|---:|---:|---:|---|---|']
    for r in matched:
        if (r['dataset'], r['perturbation_type'], r['analysis_scope']) not in effective_keys or r['metric'] != 'js_mean':
            continue
        lines.append('|' + '|'.join([r['dataset'], NAMES[r['perturbation_type']], r['matched_dialogues'],
                      r['eligible_cells'], r['matched_samples']+'/'+r['candidate_samples'],
                      fmt(r['error_minus_correct'])+' ['+fmt(r['ci_low'])+', '+fmt(r['ci_high'])+']', r['status']]) + '|')
    lines += ['', '所有0.1置信度区间及μ/δ/翻转率的匹配结果保留在CSV。MELD为了覆盖114个Dialogue而每个Dialogue大多只有1条目标，必须按实际匹配支持量解释，缺少共同单元时写NA，不能靠bootstrap伪造信息。', '',
        '## 零值与高置信度错误', '',
        '|数据集|条件|分析N|μ为零占比|δ中位数|JS中位数|置信度≥0.9的错误样本|',
        '|---|---|---:|---:|---:|---:|---:|']
    condensed = []
    for r in effective:
        source = audio_behavior if r['perturbation_type'] in ['pitch_shift', 'timbre_tilt'] else main_behavior
        selected = [x for x in source if x['dataset'] == r['dataset'] and x['perturbation_type'] == r['perturbation_type']]
        z = float(np.mean([float(x['mu']) == 0 for x in selected])) if selected else None
        delta = float(np.median([float(x['delta']) for x in selected])) if selected else None
        js = float(np.median([float(x['js_mean']) for x in selected])) if selected else None
        high = sum(int(x['original_correct']) == 0 and float(x['original_confidence']) >= .9 for x in selected)
        lines.append('|'+ '|'.join([r['dataset'],NAMES[r['perturbation_type']],str(len(selected)),fmt(z),fmt(delta),fmt(js),str(high)])+'|')
        condensed.append(dict(dataset=r['dataset'],perturbation_type=r['perturbation_type'],
                              analysis_scope=r['analysis_scope'],n=len(selected),mu_zero_fraction=z,
                              delta_median=delta,js_median=js,high_confidence_error_n=high))
    lines += ['', '高置信度错误的完整分组区间见正确/错误比较CSV中0.9–1.0区间；样本或Dialogue支持不足时不能作强结论。', '',
        '## 保存特征的事后描述', '',
        '以下仅对已保存的输入QC通过版本计算，不再运行模型或拟合。文本包含语义审核未确认版本，音频包含不完整家族中仍通过输入QC的版本；它们不是标签保持主分析的同一分母。相对L2为norm(新−原)/norm(原)，文本将四层展平；不同模态原尺度不同，不比较跨模态绝对RMSE。', '',
        '|数据集|条件|有效提取版本|相对L2中位数|余弦相似度中位数|',
        '|---|---|---:|---:|---:|']
    for r in rows(ANALYSIS / 'feature_response_summary.csv'):
        lines.append('|'+ '|'.join([r['dataset'],NAMES[r['perturbation_type']],r['extracted_variants'],fmt(r['relative_l2_median']),fmt(r['cosine_median'])])+'|')
    lines += ['', '这说明有效音频变换确实改变了编码特征，不能把小概率响应解释为输入未被修改。温和变换、冻结编码器的相对不变性、下游融合的响应程度都可能参与其中；仅靠这些距离无法识别具体原因，也不能推出整种音频模态不重要。', '',
        '## 边界与交付', '',
        '未重训冻结Baseline、未使用Test、未运行五Seed、未训练Reliability Score或CaliDist模块、未继续XP04恢复或蒸馏。完整概率、置信度变化、翻转、正确/错误分组及Dialogue汇总均已保存。', '',
        '新模型及提取器与历史XP04不同，本次结果不能证明原始输入干扰优于历史随机特征噪声。模态响应也不等于错误识别能力。下一阶段判断见FINAL_PILOT_DECISION.md，不自动扩大实验。', '']
    (ROOT / 'RAW_DATA_PERTURBATION_REPORT.md').write_text('\n'.join(lines))
    (ROOT / 'results/pilot_report_summary.json').write_text(json.dumps(dict(quality=quality,behavior=condensed),indent=2)+'\n')


if __name__ == '__main__':
    main()
