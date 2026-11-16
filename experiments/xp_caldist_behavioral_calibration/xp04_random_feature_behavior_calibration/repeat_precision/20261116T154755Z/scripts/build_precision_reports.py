"""Build concise Chinese reports without selecting a repeat count by OOF metrics."""
import json
from pathlib import Path
import pandas as pd
from mainline_core import METRICS, markdown_table


LIMITATIONS='''本轮只有Seed1475、同一批已探索的真正Validation。K30由三个K10块合并而来，条件相互嵌套，不能视为独立重复。仅有一组30次，15+15分半不是独立30次复现实验。Bootstrap复用2000次Dialogue抽样，不重新拟合，未涵盖校准器拟合、模型Seed或随机扰动的不确定性；多重比较未校正。Validation也用于原checkpoint选择和候选探索。本轮不使用Test、不重跑CMERC、不生成新噪声、不更改强度、不训练E3新结构。'''


def read(root,name):
    return pd.read_csv(root/'results'/f'{name}.csv',float_precision='round_trip',dtype={'dialogue_id':str})


def write(root,name,text):
    (root/name).write_text(text.strip()+'\n')


def build_reports(root):
    root=Path(root);m=read(root,'calibration_metrics');b=read(root,'bootstrap_results');measurement=read(root,'measurement_summary');s=read(root,'split_half_stability');shift=read(root,'confidence_shift');influence=read(root,'dialogue_concentration_summary')
    config=json.loads((root/'preregistration.json').read_text())
    display=m[['dataset','condition','method']+METRICS].copy()
    primary=m[m.method.isin(['E1','M_gaussian'])][['dataset','condition','method']+METRICS]
    cis=b[(b.candidate_condition=='K30_pooled')&((b.reference=='E1')|b.scope.isin(['primary_repeat','descriptive_repeat']))]
    decisions=[];differences=[]
    for ds in config['datasets']:
        f=m[(m.dataset==ds)&(m.condition=='K30_pooled')].set_index('method')
        delta=f.loc['M_gaussian',METRICS]-f.loc['E1',METRICS]
        old=m[(m.dataset==ds)&(m.condition=='K10_block0')&(m.method=='M_gaussian')].iloc[0]
        for metric in METRICS:
            differences.append(dict(dataset=ds,metric=metric,K30_vs_E1=float(delta[metric]),K30_vs_historical_K10=float(f.loc['M_gaussian',metric]-old[metric])))
        stable_ci=cis[(cis.dataset==ds)&(cis.reference=='E1')&(cis.candidate=='M_gaussian')]
        any_clear=bool((stable_ci.ci_high<0).any())
        ece=delta['ece15']<0;proper=(delta[['binary_brier','binary_nll']]<0).any()
        adverse=(delta[['binary_brier','binary_nll']]>0).any()
        statement=('存在ECE与至少一项proper loss同步改善的点估计' if ece and proper else '未达到ECE与至少一项proper loss同步改善的点估计')
        if adverse:statement+='；另有proper loss变差，需保留取舍'
        statement+=('；至少一项Dialogue区间完全小于0' if any_clear else '；相对E1没有任何一项Dialogue区间完全小于0')
        decisions.append(f'- **{ds}：**{statement}。仅作为本单Seed的新重复次数敏感性证据，不据此自动扩展其他Seed或原始数据。')
    diff=pd.DataFrame(differences)
    write(root,'PRECISION_METHOD.md',f'''# 随机扰动重复次数精度实验方法

固定IEMOCAP文本、MELD音频，Gaussian σ=0.05，Seed1475。保留历史K10_block0及两个新独立K10块，新增K30_pooled：每条话语先汇总全部30次，再形成一条训练记录。完整Validation为IEMOCAP527条/12Dialogue和MELD1109条/114Dialogue。

μ=mean(1[扰动预测≠原预测]×扰动最大置信度)；δ=abs(C−mean(扰动最大置信度))；JS只作诊断。δ不能平均三个10次δ，因绝对值非线性。K30的δ变小也可能来自正负置信度变化抵消，不能自动解释为可靠性提高。

校准器沿用冻结mainline_core：raw C +训练折标准化行为特征的双向Logistic，固定内层GroupKFold5、Brier选择off/1/.1/.01；off精确复用E1。IEMOCAP历史LODO12、MELD历史GroupKFold5保持原话语顺序。每个折保存完整训练/留出Dialogue、索引、标准化参数、全部内层候选与外层权重。没有选择最好块或搜索20次。

历史block0仅保存完整概率，未保存模型log-probabilities；inputs保留其缺失状态，以log_probabilities_available=False标识，不从概率伪造原始模型输出。blocks1/2保存的log-probabilities原样保留。

{LIMITATIONS}''')
    write(root,'MEASUREMENT_PRECISION_REPORT.md',f'''# 行为估计精度

## 全部条件汇总

{markdown_table(measurement)}

## 固定分半一致性

K10为本块前5/后5，K30为全局0–14/15–29。Spearman缺失表示常量，不能当作0或强稳定性；μ稀疏、并列过多时相关系数解释受限。

{markdown_table(s)}

confidence_mean_standard_error=逐次最大置信度样本标准差(ddof=1)/sqrt(K)，它是置信度均值的Monte Carlo标准误，**不是δ的标准误**。理论分母增大造成的数值减小本身不是独立复现。signed confidence_change_mean见behavior_features.csv；δ绝对化后的缩小可能只是随机正负变化抵消。

{LIMITATIONS}''')
    write(root,'OOF_CALIBRATION_REPORT.md',f'''# 全部OOF结果

ECE15为0–1单位；0.04=4%。Brier、NLL为Top-label二元正确性口径，越低越好。Accuracy/F1、AUC及置信度上/下调记录于calibration_metrics.csv；分类结果固定不变。

{markdown_table(display)}

## K30与E1、历史K10的点差

负值表示改善。

{markdown_table(diff)}

## 配对Dialogue Bootstrap

{markdown_table(cis[['dataset','candidate_condition','candidate','reference_condition','reference','metric','difference','ci_low','ci_high']])}

## 正确/错误样本置信度

{markdown_table(shift[(shift.condition=='K30_pooled')&shift.method.isin(['E1','M_gaussian'])])}

## 固定OOF逐Dialogue省略影响

不重新拟合；ECE非加性，不将某Dialogue的局部差直接当作总体贡献。

{markdown_table(influence[influence.candidate_condition=='K30_pooled'])}

{LIMITATIONS}''')
    write(root,'FINAL_PRECISION_DECISION.md',f'''# 重复次数精度实验判断

## 预登记主模型结果

{markdown_table(primary)}

## 研究判断

{chr(10).join(decisions)}

本实验目的为区分“估计不精确”和“增量可靠性信息不足”。分半相关/SE只说明测量性质，是否有额外校准价值必须由M_gaussian相对E1的OOF指标共同判断。不能只凭μ/δ非零、扰动更多或ECE单项下降宣布成功；同样不要求额外AUROC硬门槛。

本轮完成后停止，等待研究者审核。无自动补齐30次五Seed，无原始文本、面部遮挡或音色处理。

{LIMITATIONS}''')
    write(root,'RESULTS_SUMMARY.md',f'''# 10次与30次随机扰动：结果摘要

固定强度Gaussian σ=0.05，只改变汇总次数。CPU复用49,080条现有扰动预测，1,636条原始话语，4个条件共6,544条行为记录、32,720条OOF预测。三个K10条件全部保留，未挑最好块。

{markdown_table(primary)}

{markdown_table(diff)}

{chr(10).join(decisions)}

方法与边界见PRECISION_METHOD.md；测量诊断见MEASUREMENT_PRECISION_REPORT.md；所有方法、Bootstrap及Dialogue影响见OOF_CALIBRATION_REPORT.md；结论见FINAL_PRECISION_DECISION.md。

{LIMITATIONS}''')


if __name__=='__main__':build_reports(Path(__file__).resolve().parents[1])
