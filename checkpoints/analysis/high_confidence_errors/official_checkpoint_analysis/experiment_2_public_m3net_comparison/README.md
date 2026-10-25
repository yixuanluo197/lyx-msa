# IEMOCAP 官方公开权重实验二

完整结论见 `SUMMARY.md`，正式报告为 `CMERC_官方权重_实验二分类纠错与校准对比.docx`。

本次只读取两份固定作者权重的既有CSV，不调用PyTorch模型，不执行GPU推理、训练、重新选模或MELD分析。它是公开权重行为对照，不是严格训练配对的模块消融。

## 固定输入

- M³Net：`checkpoints/analysis/m3net_official_checkpoint_verification/all_predictions.csv`，预先固定作者 `Roberta_IEMOCAP3.pth.tar`，权重SHA `fb939de1f88855fda46a9d491e5cce4400c88c7187f3b792bb40f61c4ed03faa`。
- CMERC：`checkpoints/analysis/official_checkpoint_verification/IEMOCAP_predictions.csv`，固定作者 `IEMOCAP_checkpoint.pkl`，权重SHA `f3098fed2734c18171c02a59d014ee27320d845bc1ad5dfa17162b8dcfb49925`。
- 样本key是(dataset, dialogue_id, utterance_index)，保持原始1623条有效话语顺序，并逐字段核对原文、真实标签、source utterance ID、类别名称与概率映射。
- `run_manifest.json`记录原始CSV SHA、权重审计来源、统计口径和代码版本；`preserved_inputs.json`记录321个旧实验文件SHA，全部保持不变。

## 复现

从仓库根目录运行，使用已安装NumPy、matplotlib、python-docx、pytest及仓库既有报告工具的本地Python环境，无新增依赖。

```bash
# 已完成的统计命令。入口拒绝覆盖已完成目录；独立复算请指定新目录。
python scripts/analyze_public_m3net_comparison.py \
  --output-dir /path/to/new_output

# 对新的统计目录生成Word和对比图，默认拒绝覆盖已有报告。
python scripts/render_public_m3net_comparison.py \
  --output-dir /path/to/new_output

# 本目录原始CSV驱动的独立审查脚本；--root明确本地仓库根。
python checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis/experiment_2_public_m3net_comparison/source_code/independent_recompute.py \
  --root /data2/yb/paper/lyx-msa/CMERC

python -m pytest -q tests/test_public_m3net_comparison.py \
  tests/test_render_public_m3net_comparison.py tests/test_high_confidence_calibration.py
```

如果在新目录运行独立脚本，需复制 `source_code/independent_recompute.py` 到该目录的source_code，并将本目录的 `preserved_inputs.json` 复制到新目录（清单内全部旧文件SHA仍须匹配）；独立脚本审查其父目录中的产物。这里的`/path/to/new_output`是新目录示例，不是已执行的实验路径。实际目录、代码快照与命令见运行清单。CPU统计运行于本地主机；产物同时同步到biggpu同项目根，不在服务器启动任何GPU作业。

## CSV与单位

- `paired_predictions.csv`：全部1623条样本，两模型真实/预测标签、正确性、全部12个类别概率。最大置信度差、真实类别概率差、原Baseline预测类别的概率差分别保存。每个阈值的高置信错误与新增错误标记也保留，便于完全复算。
- `classification_transitions.csv`：四种互斥分类转移，合计1623；ratio的分母为1623。
- `high_confidence_transitions.csv`：三个阈值使用`>=`。各模型高置信错误率分母为各自全部达到阈值的预测。corrected/lower/not_lower互斥合计原Baseline高置信错误；`still_wrong_above_threshold`是独立重叠统计。新增CMERC高置信错误指原Baseline不是高置信错误、CMERC变为高置信错误；比例分母为全部非Baseline高置信错误话语，来源进一步拆为原正确和原低置信错误。
- `classwise_transitions.csv`：六类真实情绪支持数与分类变化；net_error_change=CMERC errors−Baseline errors，正值为错误增加。
- `calibration_comparison.csv`：全部1623话语的Accuracy、Weighted/Macro F1、Mean Confidence、ECE、MCE、Brier、NLL，及各模型自己的错误数和错误均值置信度。
- `calibration_bins.csv`：两模型各10等宽箱，左闭右开，最后包含1，空箱观测值N/A；Signed Gap=mean confidence−accuracy。ECE加权非空箱绝对差，MCE取最大绝对差；Brier按所有类别平方误差求和再样本平均，不除类别数或2；NLL为原真实类别概率自然对数负均值，不裁剪、不重新归一化。沿用实验六定义。
- `typical_cases.csv`：四种类型、三个阈值的全部匹配案例，共599行。类型和阈值可以重叠，同一句可出现多次，不能把案例行数当作去重样本数。原文及两模型全部概率完整保留。

CSV中的Accuracy、F1、ECE、MCE、置信度、比率和差值都是0–1分数；Word表头明确按百分数或原分数展示。未定义统计值写N/A，不解释为0。

## 文档与验证

Word按宋体12pt、黑色正文、黑色粗体标题、三线表、无页眉页脚生成，含两模型同图的可靠性比较。正文典型样本统一按0.90阈值、每类型selection_score降序前2条，稳定ID破同分，四种正反类型使用同一规则；这不替代599条完整CSV案例。

`report_verification.json`记录34张Word表、627个单元格与CSV的逐项复核、字体/三线表/图片检查。`independent_verification.json`记录独立CPU重算的39287项检查。`verification.json`汇总所有检查和用户参考值的计算后交叉核验。`artifact_manifest.json`记录全部新产物SHA，排除索引自身。Word的OOXML格式与数字已核验，尚未在Office/LibreOffice做分页预览。

这两份公开权重的实际训练seed、epoch、选模方式和训练配对关系未知，不得将行为差异直接归因为CL、HSCL、CC，也不提供跨训练重复的显著性结论。
