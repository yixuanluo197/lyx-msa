# 实验六：概率校准分析

结果解释与全部10箱明细见 SUMMARY.md。输入为实验一至五同一批本地validation权重的完整输入预测，仅CPU统计，不训练、不重新推理、不换checkpoint。

## 文件

- exp6_overall_summary.csv：两个数据集、两模型的ECE/MCE/Brier/NLL及平均置信度/准确率，4行。
- exp6_confidence_bins.csv：每模型10个等宽区间，共40行，含数量、平均置信度、实际准确率、有符号/绝对差与ECE贡献。
- IEMOCAP_confidence_bins.csv、MELD_confidence_bins.csv：每个数据集20行。
- figures/reliability_IEMOCAP、reliability_MELD：PNG嵌入Word，另保存PDF和SVG矢量版本。
- CMERC_高置信度错误诊断报告_实验一至六.docx：前五实验Word内容完整保留，后附实验六三线表、图及解释。
- CMERC_实验六_概率校准结果.xlsx：本轮指标和10箱完整明细。
- run_manifest.json、render_manifest.json、verification.json、artifact_manifest.json：来源、绘图及表格记录、独立核验、文件哈希。
- source_code/：本轮分析、渲染、核验脚本及测试的完整快照。

## 复现

工作目录为仓库根目录。本次使用现有Python/matplotlib/python-docx/xlsxwriter，无新增依赖。重复运行统计时须指定新的空输出目录，保护既有结果。

```bash
/home/yangbin/miniconda3/bin/python scripts/run_calibration_experiment.py \
  --output-dir checkpoints/analysis/high_confidence_errors/experiment_6_calibration_rerun
```

随后对新的输出目录调用渲染器，母版路径保持前五实验Word：

```bash
/home/yangbin/miniconda3/bin/python scripts/render_calibration_experiment.py \
  --output-dir checkpoints/analysis/high_confidence_errors/experiment_6_calibration_rerun \
  --mother-docx /data2/yb/paper/lyx-msa/CMERC/checkpoints/analysis/high_confidence_errors/experiments_2_to_5/CMERC_高置信度错误诊断报告_完整版.docx
```

只读核验现有结果：

```bash
/home/yangbin/miniconda3/bin/python scripts/verify_calibration_experiment.py \
  --output-dir checkpoints/analysis/high_confidence_errors/experiment_6_calibration
```

所有分箱和分母事先固定；空箱为N/A，不作0准确率解释。ECE/MCE是top-label校准指标；Brier和NLL评价整个概率预测。只描述本批固定checkpoint，不声明多seed显著性或严格重现论文校准分数。

## 完成记录

`run_manifest.json`为统计阶段的不可变输入清单，`render_manifest.json`引用其原SHA-256；最终完整执行状态见`completion_manifest.json`。`verification.json`包含独立NumPy复算、Word三线表及正文保留、Excel逐格一致和图坐标验证。`source_code/`完整保存本轮代码及测试快照。原实验一至五的66份既有文件在统计和交付后均通过哈希核验。
