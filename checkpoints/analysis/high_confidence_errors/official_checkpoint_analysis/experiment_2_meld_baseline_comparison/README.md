# MELD 实验二：本地 README Baseline 与官方 CMERC

- `SUMMARY.md`：实际统计和解释边界。
- `MELD_baseline_checkpoint_audit.md`：先行身份/配置/来源及选模风险审计。
- `CMERC_MELD_实验二_分类纠错与校准对比.docx`：本轮独立报告。
- `CMERC_官方Checkpoint_实验一至六综合报告_含MELD实验二.docx`：保留原官方综合报告内容，再追加 IEMOCAP 实验二和本轮 MELD 实验二。

## 固定 Baseline 与评估

本次Baseline是**本地按官方README复现**的M³Net结构，不是作者公开的MELD权重：

`experiments/official_readme_reproduction/checkpoints/MELD_baseline_seed67137/best.pt`

SHA-256：`a6ab85cb6c0c12054dade9c6bcdf01cfaa6abc417625059a927de5ba386d3738`。训练seed67137、选中epoch5、历史`official_test_selection`，存在测试集选模偏差。三个附加训练开关都关闭。预先依据用户顺序固定首选，其他候选只盘点身份/哈希，没有通过其测试F1更换模型。

完整历史训练状态包含NumPy随机状态，当前PyTorch旧版本的weights_only读取失败；在核验已知可信本地SHA后CPU审计，从`model`导出未改变数值的纯state。导出位置`checkpoints/baseline_candidates/meld_readme_fixed_20261025/model_state.pt`，SHA `1d446d95fc21d14fd550e20bed939e0510d7ac343284507dad9f40a4e2217c6d`，83个张量的key/shape/dtype/content与原model逐值一致。本轮推理仅安全读取该固定纯state，不更新参数。

训练服务器为本地user23/TITAN Xp。本次按用户要求，在`ssh biggpu`（user22）的已有官方CMERC验证环境执行统一评估，明确记录服务器变化；宿主机GPU1 UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`，逻辑cuda:0；禁用biggpu GPU4。没有因为资源繁忙而迁移训练或重新训练。

原始Baseline及导出权重不随报告上传。原始CSV已公开，因此完整统计和文档可在不持有权重、不执行GPU的情况下复核；重复Baseline前向则需上述已有本地权重或其已核验纯state。

## 已执行命令与复现

从项目根`/data2/yb/paper/lyx-msa/CMERC`执行。远端根为`/data2/yb/paper/cmerc_reproduction_20261013`。

```bash
# 已执行的唯一Baseline推理（在biggpu，现有产物无需再跑）：
# bash checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis/experiment_2_meld_baseline_comparison/run_evaluation.sh

# 已执行的CPU统计；该入口拒绝覆盖已有完成清单：
# python scripts/analyze_meld_baseline_comparison.py

# 独立CPU复核现有结果，无模型或GPU调用：
python checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis/experiment_2_meld_baseline_comparison/source_code/independent_recompute_meld.py \
 --root /data2/yb/paper/lyx-msa/CMERC

# Word和图的生成入口，拒绝覆盖已有报告：
# python scripts/render_meld_comparison.py --output-dir checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis/experiment_2_meld_baseline_comparison

python -m pytest -q tests/test_verify_meld_baseline.py tests/test_meld_baseline_comparison.py \
 tests/test_render_meld_comparison.py tests/test_high_confidence_calibration.py
```

如需在全新目录重新统计，先把本轮`baseline_predictions.csv`、`baseline_inference/`、`baseline_checkpoint_audit.json`、`MELD_baseline_checkpoint_audit.md`复制至新目录，再给统计脚本传`--output-dir`。独立脚本还需要同目录`preserved_inputs.json`（原清单的旧文件SHA须保持一致），并在新目录source_code下放脚本副本，明确传项目根。报告合并需现有官方母版和IEMOCAP实验二Word，二者由源哈希验证后读取，均不修改。

依赖沿用现有环境：GPU推理用`/data2/yb/conda_envs/cmerc-repro`；本地CPU统计与报告用已有NumPy/matplotlib/python-docx/pytest，未新增依赖。源码及必要工具快照保存在`source_code/`，审计源码在`audit/`。

## CSV与口径

- `baseline_predictions.csv/.npz`：本轮新推理的全部2610话语及七类概率；NPZ含原始log_softmax、exp概率、labels、predictions，不含object/pickle数组。它们分别与`baseline_inference/`中的同内容文件字节一致。
- `baseline_evaluation_results.csv`：Baseline新实测与既有官方CMERC的Accuracy、Weighted/Macro F1，单位为百分数0–100。
- `baseline_classification_report.csv`：七类Precision/Recall/F1（百分数）与support；`baseline_confusion_matrix.csv`行真实类别、列预测类别，原始计数。
- `paired_predictions.csv`：全部2610条稳定ID配对，原文、真实标签、两模型完整14个概率、正确性、最大置信度变化、真实类别概率变化、原Baseline预测类别概率变化，及全部七阈值标志。
- `classification_transitions.csv`：四类互斥转移，合计2610。
- `high_confidence_transitions.csv`：原始0.80/0.90/0.95，使用>=；`auxiliary_threshold_transitions.csv`另存预先固定的0.40/0.50/0.60/0.70，不能替代或混入原表。
- 各模型高置信错误率的分母为该模型所有达到阈值的预测；零分母写N/A。corrected/lower/not_lower的分母是对应原Baseline高置信错误数；remain-over-threshold是独立重叠统计。新增CMERC高置信错误从原非Baseline高置信错误集合筛选，分母及原正确/原低置信错来源分开保存。
- `classwise_transitions.csv`：七类支持数与分类变化；net_error_change=CMERC errors−Baseline errors，正值为错误增加。
- `calibration_comparison.csv`：全2610话语的accuracy/F1、mean confidence/error confidence、ECE/MCE/Brier/NLL。除性能百分数表及分类报告外，统计CSV的accuracy、F1、ECE/MCE、概率与比率统一为0–1。
- `calibration_bins.csv`：两模型各10等宽左闭右开箱，最后包含1；空箱观测N/A，图不画虚假点。ECE按样本权重加权绝对差，MCE取非空箱最大绝对差。Brier各类别平方误差求和后样本平均，不除类别数或2。NLL用原真实类别概率自然对数，不裁剪、不温度缩放、不重新归一化。
- `typical_cases.csv`：全部7阈值四种案例匹配，共470条记录，`threshold_group`明确primary/auxiliary，案例类型与阈值可重叠，不能把行数当去重样本数。Word展示预先固定辅助0.60阈值每类型selection_score最高前2条，稳定ID破同分；空类型如实无案例，不改阈值挑选有利样本。

## 文档与质量核验

MELD独立Word包含26张三线表、632个与CSV核对的单元格及可靠性图。综合Word保持母版全部body前缀、17张旧表、2张原图，完整追加IEMOCAP和MELD实验二，合计4张图，图片关系重映射且SHA一致。原正式报告未覆盖，原先“Baseline待确认”等历史文字保留，在新增导读注明补充范围。

`independent_verification.json`包含109509项独立检查，`report_verification.json`与`combined_report_verification.json`记录文档数字、宋体12pt黑色、三线表、无页眉页脚与原文/图片保留。Office/LibreOffice分页尚未实际预览。

`verification.json`汇总样本、概率、性能、原权重和352个旧文件SHA、43项测试、运行模式检查；`run_manifest.json`记录先行Baseline 18次GPU批次前向与本轮CPU统计零前向的区别。`artifact_manifest.json`记录新产物哈希并排除自身。

## 解释边界

Baseline历史测试选模可能影响对照，不将其写成作者公开权重或Validation选模。IEMOCAP实验二用作者公开IEMOCAP3，MELD用本地README Baseline，两者来源不同。当前只支持固定权重的描述性比较，不构成严格训练配对，不能将差异直接归因于CL、HSCL、CC。

MELD的ECE、MCE、Brier、NLL改善与错误均值置信度上升、辅助阈值条件错误率上升同时存在。两模型主要呈置信度不足，不能用原高阈值全零错误推断完美校准，也不能将最大置信度变化与真实类别概率变化混同。
