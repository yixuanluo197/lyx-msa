# XP04 baseline 与数据审计

审计通过：10个原始checkpoint存在，配置与真实Validation身份一致；跨模型Seed的Dialogue ID、Utterance Index、真实标签及顺序一致。

| Dataset | Seed | Validation话语 | Dialogue | checkpoint SHA256 |
|---|---:|---:|---:|---|
| IEMOCAP | 1475 | 527 | 12 | `2c0d1a7a39627e12d1244d23a225ad3da03bbce071a118909921a547a2bfd106` |
| IEMOCAP | 67137 | 527 | 12 | `1b4b91bc683de99264a7b446b84187bae3b982f1ff7c73b552e3c8e3f82eedf3` |
| IEMOCAP | 2024 | 527 | 12 | `80ee239060ae1afa3f8c49d7c99d38db8e91fb1d706c7f95087b9e6c025ebf96` |
| IEMOCAP | 3407 | 527 | 12 | `e4978c68e4119309d371152aeec469149c3fdd8989078182fb348c6db7e4f0ed` |
| IEMOCAP | 42 | 527 | 12 | `f896b657e62110dcb7b5de033df8f81f485ec037e922cca8ad8ebc48d74b43fa` |
| MELD | 1475 | 1109 | 114 | `263145b6a2057624e55791eb8c68c6d3de8a353af46a2b2f234e977588655601` |
| MELD | 67137 | 1109 | 114 | `3f79f9eb3547e35ac15ad7ff4345d9b873d0c884a6f3a1ce60a7a7e47077e701` |
| MELD | 2024 | 1109 | 114 | `59c6f3fa544b59bcdc79d14bbe8ee17772a4900b15f4d32ef088971e59c9d0df` |
| MELD | 3407 | 1109 | 114 | `0c0ccae216958f22d6c58d8d4d5a566ef61e22a81a29c0f815d7b7e180404562` |
| MELD | 42 | 1109 | 114 | `e14f4c855e277cca5d256f0e2a3e4048b0a407035be9c28a5de0a3451721026b` |

Training统计量与已审计历史统计量文件哈希完全相同。IEMOCAP使用5283条Training话语，MELD使用9989条；不从Validation/Test重新计算参考统计量。Validation和Training/Test的Dialogue ID互斥。

本次审计仅对原始特征文件计算哈希，核对有效配置中已有数据哈希；没有加载Test标签。原始特征容器可能包含全划分存储，执行推理只选择配置指定的真正Validation IDs。

模型在运行时还必须完成fresh Validation概率与已存原始预测的容差复现，以及权重、mask、上下文和源文件不变检查；此静态审计不代替运行时验证。

本次候选IEMOCAP文本同时处理实际输入的四个T1–T4数组，不将其解释成特定RoBERTa层；MELD只改音频。视觉缺失样本保留（IEMOCAP1，MELD95），视觉完全不改。

全部来源哈希：`provenance/source_hashes_before.json`。
