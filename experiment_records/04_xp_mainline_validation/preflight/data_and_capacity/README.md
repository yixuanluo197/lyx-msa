# CPU input and batch capacity preflight

本批独立实跑通过。仅 CPU；CUDA_VISIBLE_DEVICES 为空；OpenMP/MKL/OpenBLAS/NumExpr 单线程。不是训练或模型分数。

## 输入与划分

四份特征与两份 fixed_author 难度文件 SHA-256 均匹配 V2 全部 20 份 run_manifest。全对话逐项检查特征维度、有限值、标签整数/范围、utterance 数量和 ID；raw/roberta 标签、说话人与句子逐行完全一致。所有数值特征（含未使用的 legacy_text）均检查。

| 数据集 | train 对话/utterance | validation 对话/utterance | test 对话/utterance | split SHA-256 |
|---|---|---|---|---|
| IEMOCAP | 108/5283 | 12/527 | 31/1623 | d1ae10a27d20d9ee0bf81b9c884073d037a4362e2ba8b9f3f6720902d2502d62 |
| MELD | 1038/9989 | 114/1109 | 280/2610 | 5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8 |

split 使用冻结 dataloader.py 的真实函数、split_seed=1475，匹配 V2 全部模型/种子；三组互斥且 union 覆盖所有对话。IEMOCAP 原始 roberta split 元数据不是本实验使用的 split。

| 输入 | SHA-256 |
|---|---|
| caldiff/IEMOCAP_diff.pkl | db4c7b96f169948a3615a62b29d2a793bd64cccd8a0387b1de9d7ceab348a150 |
| caldiff/MELD_diff.pkl | db005ceb0e4847caffc8571eaa94ab7c5b4a310abd615b12c995c9a22ff060e0 |
| data/IEMOCAP_features.pkl | ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3 |
| data/MELD_features_raw1.pkl | d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0 |
| data/iemocap_features_roberta.pkl | 4bd60c581a5b7c70e217e18439f741b191605fbd6f2cf4dc1a6dcb7fcaa18f06 |
| data/meld_features_roberta.pkl | 58b2bdadc2eaec6abb40ac0cb3ffa3f31e929d636ece8197aecf1ca373a92951 |

## 完整采样与容量

本批运行冻结脚本 audit_batch_capacity.py：五种子 × 两变体 × 两数据集 = 20 runs，IEMOCAP 80 epochs、MELD 40 epochs，共 1200 epoch permutations；逐 epoch 结果与旧 V2 归档完全一致。每个 eligible ID 恰好一次，无删样本/替换/重复；CMERC 的预定 curriculum 子集不等于容量删样本。保持 batch_size=16 以及最后不足 16 的正常尾 batch。

| 数据集 | train 最大 edges/nodes | 拒绝首候选 epoch 数 | 最大 retries | 固定 validation edges/nodes | 固定 test edges/nodes |
|---|---|---|---|---|---|
| IEMOCAP | 998/2850 | 21 | 2 | 563/1581 | 909/2583 |
| MELD | 315/801 | 0 | 0 | 248/600 | 255/621 |

全部满足官方 1000 edges / 5200 nodes 容量。validation/test 保留 split 源顺序，不 shuffle、不删样本。

## 来源限制与环境记录

作者 difficulty 原文件未改。IEMOCAP 120 entries 包含当前 train 108 + validation 12；MELD 1152 entries 包含 train 1038 + validation 114，均无当前 test ID。训练选择只限当前 train ID，但难度生成过程未知，不能断言 train-only 生成，也不能仅凭 test ID 缺失证明无上游信息泄漏。

首次 capacity CPU 调用未用 with_cmerc_env.sh，torch_geometric 报 torch-sparse 因 libcusparse.so.11 未在搜索路径而禁用；CPU sampler 未调用该扩展，审计退出 0。已保留警告和命令于 verification_summary.json。input audit 使用项目环境包装器，退出 0。GPU 完整扩展由 leader 使用包装器另行真实 probe；本报告不据 CPU 成功断言 GPU 可用。

结构、标签/说话人/句子元数据对齐不等于重新验证上游特征提取语义。SHA 证明本批沿用 V2 输入字节，不能恢复未知的预处理来源。

详细证据：input_audit.json、batch_capacity_audit.json、verification_summary.json。审计入口：../../tools/audit_inputs.py。
