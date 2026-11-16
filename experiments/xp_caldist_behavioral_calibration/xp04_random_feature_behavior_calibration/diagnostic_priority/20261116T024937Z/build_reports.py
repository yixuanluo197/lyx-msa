from pathlib import Path
import json,pandas as pd,numpy as np
r=Path(__file__).resolve().parent;o=r/'results'
st=pd.read_csv(o/'stability_summary.csv');dist=pd.read_csv(o/'distribution_seed_summary.csv');boot=pd.read_csv(o/'matched_bootstrap.csv');cov=pd.read_csv(o/'matched_coverage.csv');auc=pd.read_csv(o/'unconditional_risk_auc_summary.csv');noise=pd.read_csv(o/'measurement_noise_summary.csv');seed=pd.read_csv(o/'matched_seed_statistics.csv')
a=json.loads((o/'independent_verification.json').read_text());assert a['status']=='passed' and a['failed_checks']==0

def table(f,precision=6):
 lines=['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']
 for row in f.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.{precision}f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
 return '\n'.join(lines)
summary=[]
for ds in ['IEMOCAP','MELD']:
 for cor in ['gaussian','dropout']:
  z=dist[(dist.dataset==ds)&(dist.corruption==cor)&(dist.measure=='mu')&(dist.class_name=='all')].iloc[0]
  s=st[(st.dataset==ds)&(st.corruption==cor)].set_index('measure');n=noise[(noise.dataset==ds)&(noise.corruption==cor)&(noise.class_name=='all')]
  summary.append(dict(dataset=ds,corruption=cor,mu_zero_fraction=z.zero_fraction_mean,mu_split_rho=s.loc['mu','spearman_mean'],delta_split_rho=s.loc['delta','spearman_mean'],delta_split_seed_sd=s.loc['delta','spearman_sample_sd'],mean_seed_median_delta_over_se=n.delta_over_signed_mean_se_median.mean()))
summary=pd.DataFrame(summary);summary.to_csv(o/'diagnostic_summary.csv',index=False)
report='# 特征扰动行为：诊断统计报告\n\n已有joint .05强度、10次扰动，两个数据集完整Validation、五Seed8180条模型记录。相同话语的不同Seed不是独立数据。本轮没有推理、扰动生成、校准拟合或Test使用。\n\n## 稀疏性与重复一致性\n\n'+table(summary)+'\n\n'
report+='μ是带扰动置信度权重的翻转均值，与翻转率零模式相同、数值不同。MELD Gaussian μ约99.13%为0，Dropout约96.12%为0；高度相等值使μ的秩相关必须谨慎解释。拆分统计先平均每Seed的三个5/5拆分，再汇总五Seed均值/样本标准差；不是15份独立数据。常量向量记NA。\n\n'
report+='源标准差为10次重复的ddof0：sample_sd=std_pop×sqrt(10/9)，signed_mean_SE=std_pop/sqrt9。δ=abs(mean signed change)，δ/SE仅是幅度/噪声描述，不是δ置信区间或p值。MELD Gaussian的逐Seed比值中位数均值0.745，Dropout2.556；IEMOCAP分别0.694/1.767。Gaussian δ的典型幅度与估计噪声同量级，结合低一致性，支持检查测量噪声限制；不能据此断言δ完全由噪声组成。零均值采样波动经绝对值也会产生正δ。\n\n## 未控制原C的错误风险AUC\n\n'+table(auc[['dataset','corruption','measure','auc_larger_is_error_mean','auc_larger_is_error_sample_sd']])+'\n\n'
report+='方向固定为数值越大越容易错误，不事后翻转。AUC接近.5表示该单一方向区分弱；小于.5不等于没有信息。上述未控制C，不代表相对E1增量。\n\n## 同Dialogue、近似原C配对\n\n'+table(boot[['dataset','corruption','measure','point_estimate','ci_low','ci_high','min_seed_pairs','min_seed_dialogues']],8)+'\n\n'
report+='差值为wrong−correct。8个探索性95%区间全部跨0：没有清晰方向证据，不是证明组差为零。2000次Dialogue共享抽样，各Seed单独计算再平均；每组2000次均有效，没有丢弃Seed。配对只控制同一Dialogue和|ΔC|≤.02，仍有C残差、类别及语境混杂；覆盖限制见配对报告。\n\n'
report+='## 对已有校准结果的解释\n\nE1已修正MELD大部分整体欠自信。当前稀疏性、Gaussian δ不稳定及近似同C下组差不明确，与行为模型未超过E1的结果相一致，尚未定位唯一原因。Dropout δ较稳定却也未出现明确配对组差，说明“测得稳定”和“能改善校准”必须分别检验。\n\n'
report+='## 验证与边界\n\n276项历史hash通过、正式输入/源码前后hash一致、11项工程测试通过。独立V2审计329项通过，复算16000条抽样统计及全部配对最优目标。V1误将μ等同翻转率，已版本化勘误并保留失败记录；15个原统计CSV未改，详见provenance/verifier_erratum.json。所有区间未做多重比较修正，数据为反复分析的Validation。本轮没有新增ECE/Brier/NLL结果，不能把诊断关联当校准收益。\n'
(r/'STATISTICAL_DIAGNOSTIC_REPORT.md').write_text(report)
cc='# 原C相近时的正确/错误样本诊断\n\n原C固定区间[0,.4)、[.4,.6)、[.6,.8)、[.8,1]，160行分箱记录均保存，包含两组均值/分位数/原C均值。IEMOCAP4/20个Dataset/Seed区间支持不足、MELD7/20，按每腐蚀/指标重复同一样本结构；不删除、不挑选有利区间。10条/3Dialogue标记是工程提醒，不代表统计功效保证。\n\n## 配对覆盖\n\n'+table(cov[['dataset','seed','n_pairs','n_dialogues','matched_correct_fraction','matched_wrong_fraction','mean_abs_confidence_distance','max_confidence_distance']])+'\n\n'
cc+='同Dialogue、不放回、|ΔC|≤.02，先最大匹配数再最小总C距离，匹配不读取行为。1279对跨Seed重复底层话语，不能当1279份独立数据。MELD仅覆盖约17.7–19.4%的正确样本及36.9–40.3%的错误样本，不能外推全部话语。\n\n## 配对差及相等值\n\n'+table(seed.groupby(['dataset','corruption','measure'])[['mean_wrong_minus_correct','tie_fraction','tie_win_fraction']].mean().reset_index(),8)+'\n\n'
cc+='tie_win=P(wrong>correct)+.5P(tie)，不翻转方向。MELD Gaussian μ约97.97%的配对相等、Dropout约91.76%；极少非零值可影响均值，不能仅凭正均值认定广泛区分能力。\n\n'+table(boot[['dataset','corruption','measure','point_estimate','ci_low','ci_high','n_valid','n_invalid']],8)+'\n\n全部区间跨0不证明严格条件独立或完全无信息。本设计是描述性诊断，没有用配对记录训练模型。\n'
(r/'CONDITIONAL_CONFIDENCE_REPORT.md').write_text(cc)
rows=[['P1','Seed1475固定方法，10→30重复技术Pilot','排除测量噪声限制；性能收益未知','两个数据集完整Validation、.05 joint、两种扰动',65440,'proposed_not_executed'],['P2','测量检验后再确认模态线索','MELD视觉当前只有单Seed微改善且区间跨0','另行固定协议，暂不扩Seed',None,'deferred'],['low','继续增加评分复杂度','已测加性/冻结基线/联合/交互，缺少继续扩展依据','不执行',0,'paused'],['defer','完整原始数据遮挡/音色实验','小机制Pilot可独立登记，不能直接完整投入','本轮不处理原始数据',0,'not_started']]
pd.DataFrame(rows,columns=['priority','experiment','purpose_and_uncertainty','scope','new_target_perturbation_records','status']).to_csv(o/'experiment_priorities.csv',index=False)
(r/'EXPERIMENT_PRIORITY_REPORT.md').write_text('''# 后续优先级

## P1：一次固定方法的测量排错Pilot

先Seed1475、两个数据集完整Validation；.05强度及评分结构不变，保留既有10次为前缀，追加20次至30。新增目标扰动记录为(527+1109)×2×20=65440，不是GPU时间估计，本轮未执行。

依据是μ高度稀疏和Gaussian δ一致性差。增加重复不会增强真实扰动强度，也不保证产生额外信息。独立同分布且方差不变时，均值标准误的理论比例sqrt(10/30)≈.577；这是数学预期，不是已测性能。

固定30次为主结果，不事后挑次数。先检查稳定性，再沿用原Dialogue OOF检验相对E1的ECE/Brier/NLL。如果稳定性改善而校准仍无增量，应记录“测量更准但未获得额外收益”，不继续扩大公式。

## P2：模态线索确认后置

此前MELD视觉仅Seed1475三项微小改善，区间全跨0；本轮joint诊断不能宣布视觉有效。先完成测量排错，再决定是否扩模态Seed，不为两个数据集事后选择不同方法。

## 低优先级：继续增加校准结构

加性μδ、冻结E1修正、联合学习和C交互已做过；近似同C配对的8个区间全跨0，多数折关闭分支，缺少“只需更复杂模型就有效”的证据。E1继续作为行为增量的必要对照。

## 暂缓：完整原始数据实验

面部遮挡或音色变化是不同机制，可以另行登记小规模技术Pilot，当前结果不能证明它会改善校准。本轮不自动开展任何原始数据任务。

优先级按能否排除具体原因排序，不代表高概率性能收益。高重复性、组间差和校准收益是不同证据；全部仍为重复Validation上的探索。
''')
(r/'FINAL_DIAGNOSTIC_DECISION.md').write_text('''# 诊断结论

- MELD μ覆盖很少：Gaussian约99.13%为0、Dropout约96.12%为0。
- Gaussian δ测量不稳定：MELD拆分ρ=.0717、典型δ/SE约.745；Dropout δ较稳定(ρ=.5807)，但稳定不等于有额外信息。
- 同Dialogue近似同C配对的8个μδ差值区间全部跨0，未得到清晰额外组差证据；不能证明完全无效，也不能外推到未匹配样本。
- 下一优先级是固定结构检验10→30重复，先Seed1475完整Validation；校准收益未知。模态扩Seed后置、评分扩展低优先级、完整原始数据处理暂缓。
- 本轮只完成统计和排序，没有新GPU推理、扰动、校准拟合、CMERC训练或Test使用，30重复尚未执行。

11项工程测试和329项独立检查通过，原统计数据未因核验器勘误而改变。上传后停止，等待研究者审核。
''')
(r/'README.md').write_text('''# XP04 特征扰动：优先级和诊断统计

只用已有10次joint扰动、两个数据集完整Validation、五Seed。

- [统计报告](STATISTICAL_DIAGNOSTIC_REPORT.md)
- [相近置信度诊断](CONDITIONAL_CONFIDENCE_REPORT.md)
- [实验优先级](EXPERIMENT_PRIORITY_REPORT.md) / [最终判断](FINAL_DIAGNOSTIC_DECISION.md)
- results/：15个原统计CSV及diagnostic_summary.csv、experiment_priorities.csv；保留1279个配对和16000条bootstrap统计。
- inputs/：8180条话语模型记录、49080条既有拆分记录、历史参考、冻结2000次抽样。

只读交付检查：python verify_delivery.py（标准库）。环境/data2/yb/conda_envs/cmerc-xp-repro/bin/python，三个BLAS线程变量均1。测试：python scripts/test_diagnostics.py。独立审计有效入口为scripts/verify_diagnostics_v2.py；V1错误等同μ与翻转率，原源码/失败JSON/日志保留，勘误见provenance/verifier_erratum.json，统计数据未改。

正式统计命令python scripts/diagnose.py --formal拒绝覆盖。重现应复制代码、协议、inputs到新目录，保持results为空；深度审计会写报告，应在副本执行，历史绝对来源路径需换机映射。V2在正式统计后新增，原before/after快照仍对应V1，未回写历史。

本轮是探索性诊断，没有新增校准性能结果；增加重复和原始数据处理均未执行。
''')
s=(r/'verify_delivery.py').read_text().replace("assert json.loads((r/'results/independent_verification.json').read_text())['all_passed']","audit=json.loads((r/'results/independent_verification.json').read_text())\nassert audit['status']=='passed' and audit['failed_checks']==0")
(r/'verify_delivery.py').write_text(s)
print(summary.to_string(index=False))
