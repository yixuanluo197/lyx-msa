# M³Net 官方 IEMOCAP3 权重验证

主报告为 [SUMMARY.md](SUMMARY.md)，可比性结论为 [pairing_feasibility.md](pairing_feasibility.md)。本轮仅一次固定公开权重的完整输入评估，CMERC 原预测直接复用，没有模态删除、校准统计、完整纠错分析或训练。

## 路径与执行

- 服务器：`ssh biggpu`（user22），项目根 `/data2/yb/paper/cmerc_reproduction_20261013`。
- 本地镜像：`/data2/yb/paper/lyx-msa/CMERC`。
- 固定权重：`checkpoints/baseline_candidates/m3net_official_20261025/Roberta_IEMOCAP3.pth.tar`。
- SHA-256：`fb939de1f88855fda46a9d491e5cce4400c88c7187f3b792bb40f61c4ed03faa`。
- 有效模型代码与兼容补丁：`checkpoints/analysis/official_checkpoint_verification/code_runtime/`，由原官方验证冻结；本轮不添加模型补丁。
- CMERC 已验证完整预测：`checkpoints/analysis/official_checkpoint_verification/IEMOCAP_predictions.csv` / `.npz`。
- 实际启动命令：`bash checkpoints/analysis/m3net_official_checkpoint_verification/run_evaluation.sh`。启动器绑定宿主机 GPU1（逻辑 cuda:0），禁用 biggpu GPU4；用 `/data2/yb/conda_envs/cmerc-repro` 和既有环境包装器执行。

`run_evaluation.sh` 和 `launch.json` 记录持久运行、PID 和设备映射。当前已完成，不需要重复推理；推理脚本拒绝覆盖已有完成清单。脚本 `verify_m3net_official_checkpoint.py` 的权重来源及哈希在代码中固定，不能传入其他候选；仅将官方测试入口的默认路径请求定向到固定 M³Net 文件，审计同时记录实际载入路径与原请求路径。

从项目根复核已有预测（CPU，不调用模型）：

```bash
python scripts/summarize_m3net_verification.py
python -m pytest -q tests/test_m3net_official_checkpoint.py tests/test_verify_author_checkpoints.py
```

上述 CPU 校验用本地已有 Python/NumPy；GPU 推理使用原 PyTorch 环境。未新增依赖。所有运行及校验脚本副本见 `source_code/`，不要在源码快照目录内直接运行依赖默认根路径的脚本；CPU 复算可通过 `--project-root` 明确项目根。

## 文件口径

- `evaluation_results.csv`：两份公开权重的真实 Accuracy / Weighted F1 / Macro F1，单位为百分数（0–100），有效样本数为整数。
- `classification_report.csv`：仅本次 M³Net 六类别 Precision/Recall/F1（0–100）与支持数。
- `confusion_matrix.csv`：行是真实类别，列是预测类别，均为原始计数。
- `all_predictions.csv`：全部 1623 条原始文本、稳定样本 ID、真实/预测类别、最大概率、真实类别概率、全部六类概率及派生正确性。话语索引从 0 开始。
- `predictions.npz`：原始 log_softmax、其 exp 概率、labels、predictions，只读安全加载，不含 pickle/object 数组。概率未归一化或温度变换。
- `checkpoint_audit.json`：纯 state_dict 的85个参数张量、完整strict匹配及实际来源。缺少训练seed/epoch/选模元数据，保持null。
- `run_manifest.json` / `evaluation.log`：本轮实际参数、环境、模型/后端、严格加载/无更新/两次前向记录。
- `inference/`：官方评估观察器的原始输出，顶层预测/类别报告/混淆矩阵/权重审计是字节一致的命名副本。
- `verification.json` / `protocol_audit/`：概率、原始特征标签/文本/ID/顺序、批次、独立分类指标与模块可比性证据。
- `preserved_inputs.json`：此前291个结果文件的SHA，本轮全部保持不变。
- `artifact_manifest.json`：本轮产物校验值，排除索引文件自身。

类别顺序：happy、sad、neutral、angry、excited、frustrated。两模型概率无需彼此相等；对齐检查确保对应同一话语及类别。公开权重行为比较可继续，严格模块因果对照仍缺训练与配对证据。此前审计报告作为历史记录保留，不回写或伪造已知训练历史。
