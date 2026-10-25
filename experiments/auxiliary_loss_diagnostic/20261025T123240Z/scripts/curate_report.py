"""Add evidence-grounded Chinese interpretation to the frozen analysis output."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((ROOT / 'results/metrics_summary.json').read_text())
    assert summary['completed'] and summary['verification_status'] == 'passed'
    generated = ROOT / 'reports/GENERATED_SNAPSHOT_REPORT.md'
    original_report = ROOT / 'reports/CMERC_AUXILIARY_LOSS_DIAGNOSTIC.md'
    assert not generated.exists(), 'Preserve the original generated report once; do not overwrite.'
    generated.write_bytes(original_report.read_bytes())
    curves = PROJECT / 'experiments/hscl_ablation_validation/20261025T080937Z/results/curves.csv'
    rows = list(csv.DictReader(curves.open()))
    (ROOT / 'results/historical_validation_curves.csv').write_bytes(curves.read_bytes())
    history = []
    for variant in ('XP04_full', 'XP06_wo_cc', 'XP12_wo_hscl'):
        part = sorted((r for r in rows if r['variant'] == variant), key=lambda r: int(r['epoch']))
        assert [int(r['epoch']) for r in part] == list(range(1, 41))
        best = max(part, key=lambda r: float(r['validation_weighted_f1']))
        history.append({'variant': variant, 'best_validation_weighted_f1': float(best['validation_weighted_f1']),
                        'best_epoch': int(best['epoch']),
                        'last10_validation_weighted_f1_mean': statistics.mean(float(r['validation_weighted_f1']) for r in part[-10:]),
                        'last10_validation_classification_loss_mean': statistics.mean(float(r['validation_loss']) for r in part[-10:])})
    assert [r['best_validation_weighted_f1'] for r in history] == [65.24, 65.77, 65.69]
    with (ROOT / 'results/historical_validation_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    findings = {
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'historical_curves_source': str(curves.relative_to(PROJECT)), 'historical_curves_sha256': sha(curves),
        'generated_report_sha256': sha(generated), 'metrics_summary_sha256': sha(ROOT / 'results/metrics_summary.json'),
        'history_validation_only': history, 'preregistered_priority': summary['priority'],
        'next_single_mechanism_to_verify': '原始分类Loss与CC在不同真实类置信度下的局部目标方向关系',
        'new_coefficient_selected': None, 'new_training_authorized_or_started': False,
        'reason_for_no_unique_loss_coefficient': 'CC冲突频率跨快照最高，但late MSCL有效梯度尺度更高，原唯一筛选规则未通过。',
        'classification_source': {'file': 'code_xp04/model_cm.py', 'lines': '41-49', 'sha256': sha(ROOT / 'code_xp04/model_cm.py'),
                                  'fact': '非真实类别仍贡献-alpha*log_p，最终对N*C取mean；本组保留原公式。'},
        'model_forwards_added_by_editorial_analysis': 0, 'autograd_calls_added_by_editorial_analysis': 0,
    }
    (ROOT / 'results/interpretation_summary.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2) + '\n')
    lines = ['# CMERC辅助损失梯度冲突与伪标签质量诊断（XP13）', '',
             '## 主要结论与下一步', '',
             '**本组已完成并通过验收；未开展任何新训练、参数更新或测试集前向。** 四个真实Checkpoint在同一组32个训练Batch上完成128个诊断单元，768次模型前向、512次独立autograd.grad。观察到局部梯度竞争及二值关系标签的语义混杂/阶段偏移，尚不能据此确定论文F1差距的根因，也不支持直接宣称HSCL无效或CC有害。', '',
             '**下一次最值得验证的单一机制：原始分类Loss与CC在不同真实类置信度下的目标方向关系。** 这是对实测冲突的机制核查建议，不是预登记筛选产生的唯一降权胜者。CC在XP04两个快照的共享主干均有28/32负余弦，但后期MSCL的加权范数比最高，原筛选规则明确返回`insufficient`。因此当前不选择新的最优系数、不启动五Seed，也不自动新增训练。', '',
             '若安排后续最小训练，应先完成上述方向核查，再单独预登记一个损失系数与现有控制的验证集对照；数据、Seed、CL、模型、其他系数保持，测试仍隔离。现有证据不足以决定应降低CC还是MSCL，不把“范数大”当作选系数依据。不建议同时改伪标签和系数。', '',
             '## 历史验证集背景及XP12现象', '',
             '以下直接重算XP12归档中只含Train/Validation的40轮曲线；没有重新评估历史模型，也没有读取历史测试分数来作选择。', '',
             '| 历史实验 | 最佳验证W-F1 | 最佳Epoch | 末10轮验证W-F1 | 末10轮验证分类Loss |',
             '|---|---:|---:|---:|---:|']
    for r in history:
        lines.append(f"| {r['variant']} | {r['best_validation_weighted_f1']:.2f}% | {r['best_epoch']} | {r['last10_validation_weighted_f1_mean']:.3f}% | {r['last10_validation_classification_loss_mean']:.4f} |")
    lines += ['', 'XP12最佳值比XP04高0.45个百分点，但末10轮低0.115个百分点。关闭HSCL不保证后期持续改善：在XP12第40轮，实际辅助目标只剩CC，其共享主干仍有28/32负余弦；原分类目标、过拟合及其他泛化因素也没有被去除。这是与现象相容的推断，不是关闭HSCL造成/解决梯度冲突的因果证明。XP12最佳轮e5与XP04最佳轮e8并非等时点；XP12离线HSCL梯度只是反事实参考。', '',
              '## 关键定量证据', '',
              '共同有效参数支持上的共享主干统计（每项均32个有效Batch；范数比已经乘原系数，负余弦频率不能外推整个历史训练）：', '',
              '| XP04快照 | 项 | 负余弦Batch | cos中位数 | 加权梯度/分类梯度范数比中位数 |',
              '|---|---|---:|---:|---:|']
    for cp in ('XP04_best', 'XP04_latest'):
        for loss in ('cc', 'mscl', 'cscl', 'aux_actual_sum'):
            row = next(r for r in summary['gradient_summary'] if r['checkpoint_id'] == cp and r['loss_b'] == loss and r['group'] == 'shared_backbone')
            lines.append(f"| {cp} | {loss} | {row['conflict_n']}/{row['valid_n']}（{100*row['conflict_frequency']:.2f}%） | {row['cosine_median']:.4f} | {row['norm_ratio_actual_weighted_median']:.4f} |")
    lines += ['', 'CC的冲突不只出现在主干：XP04 e40的分类头、音频投影、视觉投影与共享图模块各为30/32，文本GRU为23/32。e8共享图模块为28/32、文本GRU17/32。不能把不同组相加当独立样本。MSCL/CSCL对分类头没有共同梯度支持，记N/A；其冲突主要在表示层测量。完整逐层结果及梯度内积、raw/reference/actual范数见底层CSV。', '',
              'XP04 e8五路both-wrong默认0占全部关系观测的16.96%–20.71%；其中移除文本关系的837个默认0里，682个真实类概率实际下降、155个上升。该默认0显然不统一表示“置信度上升”。e40文本关系77个默认0均伴随真实类概率下降；这只是分支语义事实，不将其称为情绪伪标签错误率。', '',
              '文本关系label1从e8的65.70%变为e40的92.10%，音频从60.98%变为80.27%；上下文intra从19.37%变为36.14%。e40文本label0的Batch逆频率权重最大60.33，XP12 e40反事实路径最大181。类别比例与权重均会变化，但没有证据证明所有关系均缺正负对。', '',
              'XP04 e40文本关系在3个Batch各有1个anchor缺真实同类正对；XP12 e40有4个，其他checkpoint/关系未见该缺对现象；640条关系数值审计均未见非有限logit，256次wrapper没有NaN/Inf回退。XP12 e40文本关系有65个anchor的neg加权和小于0，但pos+neg分母仍为正；不能把它误报为负分母或数值崩溃。', '',
              '不同情绪类别的关系分布也不同：XP04 e40音频label1在情绪ID0为约0.95、ID2为约0.22；它们不是情绪准确率。真实类别置信度移除差值、主模型正确/错误条件下的分布、分支覆盖及关系Loss贡献均在聚合CSV中保留，避免只看总体比例。', '',
              '## 分类目标的补充源码核查', '',
              '[model_cm.py](../code_xp04/model_cm.py)第41–49行先令`pt=onehot*log_p`，但`fl`末尾仍乘完整`log_p`，最后对全部样本和类别求mean。对单话语的七类输出，其真实实现为：', '',
              '$$L_{cls}=\\frac{\\alpha}{7}\\left[-(1-\\log p_y)^\\gamma\\log p_y-\\sum_{j\\ne y}\\log p_j\\right].$$', '',
              '这里包含非真实类别的负对数概率惩罚，不能只按名称理解为仅目标类的Focal项。另一方面CC在满足移除后真实类log概率不低于学生的分支中（原代码包含相等），对学生施加提高真实类log概率的梯度；teacher分支no_grad。两者局部目标可以不同。该实现被Baseline及CMERC共同沿用，本组没有改公式；它本身不能单独证明CMERC相对Baseline收益缺失或论文差距来源。', '',
              '待验证假设：高置信样本上原分类目标与CC的竞争可能解释部分负余弦。下一次先只读核查现有权重中按预登记真实类概率区间分层的目标方向，并区分单样本logit方向与共享参数聚合方向；不能由本批Batch余弦直接宣称该条件机制已成立。建议使用原公式、训练样本、无参数更新，不直接换成另一版FocalLoss。', '',
              '## 完成范围、隐私及复核', '',
              '选取512个不同训练对话、4936条话语，覆盖原训练池1038对话/9989话语的49.33%/49.41%，七类话语数依次为2315/607/134/347/850/135/548。只使用训练ID；未构建验证或测试诊断loader。固定输入分层重组影响padding、正负对和信息移除基准，eval模式及同步teacher也不同于历史在线训练。只有四个离散快照，不重建不存在的早期权重。', '',
              '验收验证2302个历史文件和四Checkpoint未变，每个case的模型参数、BN等buffer、RNG及.grad状态前后一致。52项合成CPU检查覆盖采样、统计、运行守卫与分析；本组无新训练、无optimizer、无step、无Tensor.backward、无测试F1/ECE/CEL。独立审查及哈希记录可追溯。', '',
              '原始冻结历史参考内嵌的划分ID列表仅保留服务器；公开副本删除ID并保留数量、原SHA和验证指标。实际采样ID、逐话语关系、预测、置信度、原权重和日志均不公开。公开哈希可审计来源，但克隆GitHub后缺外部权重/数据/私有采样清单不能直接重跑全部诊断。', '',
              '9组PNG/SVG已人工检查；第7图原长图例被裁切，另存原图后通过独立纯排版脚本修复，输入CSV和冻结绘图脚本均未变，前后SHA单列。没有修改统计结果。', '',
              '事实：局部梯度竞争、标签比例/默认0语义、现有数值及缺对计数。推断：辅助目标与分类目标可能在部分阶段竞争，标签覆盖变化可能影响HSCL。未证明：持续训练冲突的频率、模块的净因果损害、其他Seed/测试集收益、最优损失系数或论文差距的唯一根因。', '',
              '[解释汇总](../results/interpretation_summary.json)、[历史验证对照](../results/historical_validation_comparison.csv)、[源码审计](../reports/SOURCE_AUDIT.md)、[独立验收](../provenance/final_verification.json)、[纯排版修复记录](../provenance/figure7_layout_revision.json)。', '',
              '## 实测图表', '']
    for name, title in [('01_cosine_distributions', '共享主干余弦分布'), ('02_reference_norm_ratios', '原系数加权梯度尺度'),
                        ('03_group_conflict_frequency', '参数组冲突比例及有效分母'), ('04_pseudo_label_proportions', '五路二值关系标签分布'),
                        ('05_emotion_relationship_heatmap', '情绪类别条件下的关系标签分布'), ('06_paired_checkpoint_cosine_changes', '同Batch快照差值'),
                        ('07_pseudo_branch_coverage', '标签条件分支覆盖'), ('08_relationship_pair_coverage', '真实同类/异类关系对比例'),
                        ('09_actual_reference_gradient_scale', '实际与反事实参考梯度尺度')]:
        lines += [f'### {title}', '', f'![{title}](../results/figures/{name}.png)', '', f'[SVG](../results/figures/{name}.svg)', '']
    lines += ['## 冻结分析脚本生成的完整统计', '', generated.read_text().replace('# CMERC辅助Loss快照诊断报告', '### 快照统计明细', 1)]
    original_report.write_text('\n'.join(lines))
    root_report = PROJECT / 'reports/CMERC_AUXILIARY_LOSS_DIAGNOSTIC.md'
    prefix = '../' + str(ROOT.relative_to(PROJECT)) + '/'
    root_report.write_text(original_report.read_text().replace('(../', '(' + prefix))
    (ROOT / 'provenance/report_curation.json').write_text(json.dumps({
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'original_generated_sha256': sha(generated),
        'reviewed_batch_report_sha256': sha(original_report), 'public_root_report_sha256': sha(root_report),
        'editorial_script_sha256': sha(Path(__file__).resolve()), 'frozen_analyzer_changed': False,
        'statistics_or_measurements_changed': False, 'additional_model_forwards': 0, 'additional_autograd_calls': 0,
    }, indent=2) + '\n')
    print(json.dumps({'report': str(root_report), 'history': history}, ensure_ascii=False))


if __name__ == '__main__':
    main()
