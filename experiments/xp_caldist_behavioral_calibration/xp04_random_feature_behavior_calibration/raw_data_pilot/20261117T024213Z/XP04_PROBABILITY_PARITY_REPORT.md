# XP04 冻结预测概率 Parity 报告

两个数据集均为 **NOT_RUN**：一次性最终 Validation 的 Clean Feature Gate 都未通过，按预先冻结协议禁止加载 CMERC 模型或进行 Softmax 比较。本报告的概率误差和类别一致率均为 `null`，不是0，也不能解释为预测不一致。

## 冻结 Checkpoint 与配置

本次仅重新计算下列原始文件的 SHA-256；未反序列化 Checkpoint、未读取模型参数。

| 数据集 | Seed | 原始 Checkpoint SHA-256 | Effective config SHA-256 |
|---|---:|---|---|
| IEMOCAP | 1475 | `2c0d1a7a39627e12d1244d23a225ad3da03bbce071a118909921a547a2bfd106` | `59e40ae7b7d63bb5f56806eedefb2f70c6797095455f63c083917b1141275369` |
| MELD | 67137 | `3f79f9eb3547e35ac15ad7ff4345d9b873d0c884a6f3a1ce60a7a7e47077e701` | `5d6271460c882dc34bf55bbcdec96afd09bb6b963c238745e04a54fb20515f68` |

Checkpoint 对应冻结 XP04 的 `runs/cmerc/<DATASET>_validation_seed<SEED>/best.pt`，配置对应 `configs/effective/<DATASET>_cmerc_validation_seed<SEED>.json`；完整本地路径及哈希证据保存在 [final_probability_gate_verification.json](provenance/final_probability_gate_verification.json)。

## Feature Gate 阻断证据

- IEMOCAP：固定527条 Validation 全覆盖，1582维全部有限，但最大特征绝对误差 `17.9104099936`、MAE `0.0135472936581`；allclose、MAE、标准化RMSE、平均/最小余弦及相关性p05门槛均失败。`passed=false`，不能进入模型概率验证。
- MELD：1611维只有 `1108/1109` 条有限，1条原始音轨缺失；最大特征绝对误差 `2.31946767988`、MAE `0.0683247657352`。保持完整双向GRU上下文后，只有 `1099/1109` 条300维结果有效；缺失话语所在完整Dialogue的10条话语均保留为缺失，不以官方特征补齐。113个Dialogue上下文完整、1个不完整。300维最大特征绝对误差 `1.50055301189`、MAE `0.0221887707975`。1611与300维Gate均失败，`full_validation_coverage=false`、`full_dialogue_bigru_context=false`。

上述数值来自已锁定转换器的一次性最终特征评估；没有根据这些 Validation 结果修改转换器，也未执行新的拟合。MELD原始GRU前向属于上游特征生成，不是CMERC前向。相关结果详见 [IEMOCAP Feature Gate](provenance/IEMOCAP_feature_gate.json)、[MELD Feature Gate](provenance/MELD_feature_gate.json)及各自 `clean_feature_metrics.json`。

## 非推理守卫验证

使用两个全新 Python 子进程调用真实适配器 `run()`，输入分别为本轮实际失败的 IEMOCAP/MELD Gate。每个子进程同时安装：

1. 框架/模型导入拦截器：一旦尝试导入torch、TensorFlow、Keras、CMERC模型或官方dataloader即失败。
2. 文件打开审计钩子：一旦尝试打开Checkpoint、Pickle、HDF5或NumPy特征/概率数组即失败。
3. `import_helper()`哨兵：一旦到达模型助手入口即失败。

两次均首先收到预期 `ValueError: Feature gate required: <DATASET>.passed=true`。所有框架导入、参数/数组打开及模型助手调用次数为0，torch/numpy均未进入这些子进程的 `sys.modules`。这证明当前失败Gate在模型导入、Checkpoint加载、任何CMERC前向之前已阻断；这些守卫测试也没有加载重建特征数组。

## 概率结果

| 数据集 | 状态 | 最大概率绝对误差 | 平均概率绝对误差 | 预测类别一致率 | 最大置信度绝对差 |
|---|---|---|---|---|---|
| IEMOCAP Seed1475 | NOT_RUN | null | null | null | null |
| MELD Seed67137 | NOT_RUN | null | null | null | null |

预先冻结的概率门槛仍为最大绝对误差≤`1e-4`、平均绝对误差≤`1e-5`、类别一致率100%、最大置信度差≤`1e-4`。本轮没有产生Softmax向量或逐话语概率比较CSV，亦没有打开历史概率文件进行比较。

概率适配代码已准备好，但不代表兼容性验证通过。当前不能把重建特征作为已验证的XP04兼容输入，也不进入原始音频干扰。本轮未训练CMERC、未使用Test、未修改冻结输入、未Commit或Push。
