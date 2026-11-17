"""Summarize locked results; no fitting, model inference, or source pickle access."""
from pathlib import Path
import csv
import json

OUT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((OUT / 'provenance' / name).read_text())


def main():
    datasets = {ds: read(f'{ds}_clean_feature_metrics.json') for ds in ('IEMOCAP', 'MELD')}
    assert all(not d['feature_gate']['passed'] for d in datasets.values())
    rows = []
    for ds, result in datasets.items():
        for key, stage in [('selected_feature_metrics', '1582' if ds == 'IEMOCAP' else '1611'),
                           ('context300_feature_metrics', '300')]:
            if key not in result:
                continue
            m = result[key]
            rows.append(dict(dataset=ds, stage=stage, expected_rows=m['shape'][0],
                             finite_rows=m['fully_finite_rows'], width=m['shape'][1],
                             max_abs=m['max_absolute_error'], MAE=m['mean_absolute_error'],
                             RMSE=m['rmse'], train_scaled_RMSE=m['global_train_scale_nrmse'],
                             dialogue_equal_NRMSE=m['dialogue_equal_nrmse'],
                             mean_cosine=m['mean_cosine_similarity'], min_cosine=m['min_cosine_similarity'],
                             correlation_p05=m['required_target_dimension_correlation_p05'], passed=m['passed']))
    with (OUT / 'clean_parity_summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ('| 数据集/阶段 | 有效/固定条数 | Max abs | MAE | RMSE | Train尺度RMSE | 平均/最小余弦 | 相关性p05 |\n'
             '|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in rows:
        table += (f"| {r['dataset']} {r['stage']}维 | {r['finite_rows']}/{r['expected_rows']} | "
                  f"{r['max_abs']:.6g} | {r['MAE']:.6g} | {r['RMSE']:.6g} | {r['train_scaled_RMSE']:.6g} | "
                  f"{r['mean_cosine']:.8f} / {r['min_cosine']:.8f} | {r['correlation_p05']:.6g} |\n")
    final = f"""# XP04 双数据集音频转换器重建：最终决定

本轮结果：**IEMOCAP 与 MELD 均为 C（转换器重建未通过）**。未找到作者原始转换器，Train 配对拟合的候选也未通过锁定方案的独立 Validation Clean Feature Parity。没有原始算法恢复 A 或功能等价通过 B 的结果。

## 实际完成的工作

- 沿用先前已确认的本地官方资产、XP04 effective split 和原始 checkpoint，没有重新下载资产。
- IEMOCAP：仅 Seed1475 effective Train 5283条/108对话；比较 IS10、IS10 compat、emobase2010 三个官方 v3.0.1 候选配置。最终选 IS10 + 无行变换 + 逐列仿射。固定 Validation 527条/12对话全部提取成功。
- MELD：Seed67137 effective Train 固定9989条/1038对话，其中9988条成功配对。先用256个不同 Train 对话的一条话语比较8个前端，再锁定 ffmpeg mono/16kHz/无波形zscore，完整 Train 比较转换器，最终选最佳列匹配加仿射。固定 Validation1109条/114对话，1108条提取成功。
- 两个数据集均完成按对话分组的3折 Train 内模型选择；精确列匹配、顺序/仿射、行标准化、PCR、低秩 Ridge 与全秩 Ridge 的结果全部保留。逐列关系的高相关性不能代替逐维兼容验证。
- 两份最终模型锁定后，各仅打开一次本轮 Validation 目标。MELD 额外执行原始官方 BiGRU 的完整对话前向，保留顺序、真实长度、33步右侧padding和Masking(0)，6组权重逐元素核验一致。

## 独立 Validation 数值

{table}

误差只在双方均有限的元素上计算；缺失行仍留在固定形状中，并使完整覆盖门槛失败。表中的有效子集指标不能宣称为全量通过。逐维相关性、均值/标准差差异及 Wasserstein 距离见两个 Clean Parity 报告及其本地逐维CSV。

## 为什么停止

锁定协议要求全部元素 allclose(atol=rtol=1e-4)、MAE≤1e-5、Train尺度RMSE≤1e-3、平均/最小非零行余弦≥0.99999/0.9999、所有目标非恒定维相关性可定义且p05≥0.999，并要求固定Validation完整覆盖。两数据集均有数值门槛失败。MELD1611维与上下文300维必须同时通过，本次均未通过。

MELD Train raw125/utt3 的原包和同划分候选副本均损坏；Validation rawdev110/utt7 不存在。没有借用其他划分或官方目标特征填补。缺失 Validation 话语所在XP04对话1149共有10条话语，完整BiGRU上下文不可恢复，因此300维只生成113个完整对话/1099条真实话语。33步padding的差异独立报告，不混入真实话语统计。

由于特征门槛失败，**XP04 Seed1475/67137 的概率验证为 NOT_RUN**；概率误差与类别一致率未测量，不能写成零误差或不一致。没有执行CMERC前向、训练或原始音频干扰。

## 已知信息、推断与未知环节

- 已知：当前候选官方配置可执行，已得到 Train/Validation 候选特征及真实配对误差；官方 MELD1611→BiGRU→300 的权重和真实上下文计算可执行。此前官方1611直接输入的24条Clean Parity仍属于历史证据，不能当成本轮WAV生产链通过。
- 支持的有限推断：IEMOCAP候选IS10与XP04有较强列级线性关系；MELD也存在可利用的列相关性。它们不足以恢复完整数值空间。
- 仍未知：作者实际音频前端版本、完整解码/预处理、外部归一化记录，以及 MELD 原6373→1611选择器/索引/统计量。此次拟合对象均明确是重建模型，不能冠名为作者原SVM、PCA或原始配置。
- 本轮失败只约束已测试的配置及映射族。它不能证明所有可能转换器都不成立，也不能证明模型不响应真实音频干扰。

## 验证独立性与失败记录

历史审计已经看过部分Validation摘要和24条GRU检查，所以这里的“独立”指本轮Train拟合/选择后，Validation只做一次检验，无结果反馈调参；不声称该数据此前从未被研究者接触。全秩Ridge菜单是在Train阶段补充的协议v2，早于本轮任何Validation候选提取/目标访问；原菜单、阈值、折分和历史结果均留档，不能称全部菜单最初完全预注册。

MELD首次probe启动时代码哈希缺口保留；随后冻结副本独立重算相同256×8候选，全部逐位、文件哈希一致。第一次frontend CV在完整数值输出后的日志关闭阶段退出120，失败源码与记录保留；修复只涉及日志flush，后续运行退出0。IEMOCAP早期包装接口失败和音频接口控制也保留，未删除不理想条件。

## 下一阶段建议与边界

本轮不把这两份近似模型作为XP04兼容音频接口，不进入扰动实验或扩大到其他Seed。若后续获得作者原配置/选择器/统计量，可在新的协议和独立输出目录重验；若要继续探索更复杂的重建，须使用Train内分组选择，并为已经看过本轮结果的Validation明确重用性质。不能继续调本次Validation后再声称一次独立通过。

未修改原始特征或checkpoint，未使用Test槽位或标签，未训练CMERC；未Git commit/push。完整媒体、特征、模型参数与逐话语清单保留本地local_only目录。共享pickle容器会整体反序列化，但只索引许可的Train/Validation字段，Test记录未被用于拟合或评估。

## 证据入口

- [方法比较与全部候选](CONVERTER_METHOD_COMPARISON.md)
- [IEMOCAP重建](IEMOCAP_FEATURE_RECONSTRUCTION.md) / [独立Parity](IEMOCAP_CLEAN_PARITY_REPORT.md)
- [MELD重建](MELD_FEATURE_RECONSTRUCTION.md) / [独立Parity](MELD_CLEAN_PARITY_REPORT.md)
- [概率门控报告](XP04_PROBABILITY_PARITY_REPORT.md)
- [汇总CSV](clean_parity_summary.csv) / [验证清单](verification.json)
"""
    (OUT / 'FINAL_RECONSTRUCTION_DECISION.md').write_text(final)
    (OUT / 'README.md').write_text('''# XP04 音频转换器反向重建

两个数据集均未通过本轮独立Validation兼容门槛，分类C；不运行CMERC概率比较或原始音频干扰。

入口：[最终决定](FINAL_RECONSTRUCTION_DECISION.md)。八项交付由七份主题报告及verification.json组成，另有方法比较与Clean Parity汇总CSV。

- 配置与协议：configs/reconstruction_protocol.json（v2），v1与修订记录保留。
- 实验实现：scripts/，含提取、分组CV、转换器、受锁保护的单次Validation评价、门控概率适配器及合成测试。
- 过程证据：provenance/，含配置/源码/输入hash、训练记录、fold边界检查、模型锁、一次Validation访问标记、异常与失败结果。
- 本地私有产物：local_only/，含清单、原始目标/候选/重建特征、转换器参数、OOF及逐维CSV。不要直接公开此目录或未经审查上传整目录。

本次未Git commit/push。原始媒体、完整特征、权重不作公开交付。
''')


if __name__ == '__main__':
    main()
