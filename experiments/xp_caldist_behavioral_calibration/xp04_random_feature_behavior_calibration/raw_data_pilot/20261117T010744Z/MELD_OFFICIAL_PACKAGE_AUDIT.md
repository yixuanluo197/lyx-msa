# MELD 官方特征与模型包审计

审计日期：2026-11-17 UTC。本轮仅下载指定约 923 MB 特征模型包、核验归档、对齐既有音频表示，并验证少量真正 Validation 的官方网络前向。没有下载 10.9 GB 原始数据包，没有训练或修改 XP04，没有原始音频干扰。

## 来源、空间与下载验证

来源：[declare-lab/MELD 官方包页面](https://huggingface.co/datasets/declare-lab/MELD/blob/main/MELD.Features.Models.tar.gz)。本轮固定 Hugging Face repository revision 为 `9abc51ee7903424ffb971297608aa6d3d0de3bfa`，下载该 revision 的 `MELD.Features.Models.tar.gz`。

下载前 审计文件系统 剩余 **76,247,830,528 字节**，超过下载与隔离检查所需空间。发现本地历史包后，先独立计算其哈希；与用户给出的官方 SHA-256 完全一致。另行下载独立网络副本，最终必须满足：

- 压缩大小：**922,694,594 字节**。
- SHA-256：`59131c6904ab1272912c6ebec7873524e6f90bce842b333bc27f02411718b2c3`。

首次 HTTP 200 下载仅得到 167,314,834 字节，短于响应声明长度，被完整性检查拒绝。这个截断文件和失败记录保留，没有视为源文件损坏。随后核验 HTTP 206 字节范围请求，检查每段 Content-Range 与长度，使用四路 32 MiB 分段恢复；最终完整体重新计算官方 SHA-256。最终成功状态以 `evidence/package_inventory.json` 与 `verification.json` 为准；首次失败保留为 `provenance/official_download_attempt1_failed.json`。

没有覆盖历史包或旧解包目录。完整目录来自流式扫描已核验归档：**23 个目录项，3 个目录、20 个普通文件，未压缩普通文件合计 1,170,662,066 字节**。没有链接或特殊类型成员。逐成员 SHA-256 与现有解包文件全部核对一致，再只读使用这些副本；新包字节哈希一致也证明两份归档的成员身份相同。

## 完整归档目录

下列文件都位于 `MELD.Features.Models/` 下，完整大小和成员哈希分别见 `evidence/package_inventory.json`、`evidence/package_inventory.json`。

```text
MELD.Features.Models/
  README.txt
  models/
    audio_weights_emotion.hdf5
    audio_weights_sentiment.hdf5
    text_weights_emotion.hdf5
    text_weights_sentiment.hdf5
    bimodal_weights_emotion.hdf5
    bimodal_weights_sentiment.hdf5
  features/
    audio_emotion.pkl
    audio_sentiment.pkl
    audio_embeddings_feature_selection_emotion.pkl
    audio_embeddings_feature_selection_sentiment.pkl
    data_emotion.p
    data_sentiment.p
    text_emotion.pkl
    text_sentiment.pkl
    text_glove_CNN_emotion.pkl
    text_glove_CNN_sentiment.pkl
    text_glove_average_emotion.pkl
    text_glove_average_sentiment.pkl
    bimodal_sentiment.pkl
```

## 目标特征与权重

| 特征成员 | 大小（字节） | 已核实用途 / 形状 | 对应权重 |
|---|---:|---|---|
| `audio_emotion.pkl` | 56,810,678 | train/val/test 三个字典，每对话 float32 `(33,300)`；真实行与 XP04 全部精确一致 | `audio_weights_emotion.hdf5`，19,622,496 字节 |
| `audio_sentiment.pkl` | 113,517,878 | 官方 sentiment 上下文音频表示；不能替代 emotion 300 维 | `audio_weights_sentiment.hdf5`，75,633,624 字节 |
| `audio_embeddings_feature_selection_emotion.pkl` | 177,669,409 | 已完成选择的 1611 维声学输入，已用于官方 emotion 网络的少量 clean 检查 | emotion HDF5 |
| `audio_embeddings_feature_selection_sentiment.pkl` | 156,942,913 | 已完成选择的 1422 维声学输入 | sentiment HDF5 |

HDF5 自带模型结构及权重，能据实判断网络，不再依赖当前 GitHub baseline 代码猜测。**emotion 官方权重的实际架构是双向 GRU 150×2，输出 300 维，不是两层双向 LSTM 300×2。** sentiment 权重是另一条 1422→600 的双向 LSTM 路径。详细配置、版本、哈希及 CPU 复现见 [网络恢复报告](BC_LSTM_RECOVERY_REPORT.md)。

完整目录没有 OpenSMILE 配置、原始 6373 维声学数据包、SVM 特征选择器模型/字段索引或外部标准化参数文件。`feature_selection` 文件名表示已经选择后的数组，不能解释为选择器可执行对象。`.p` 元数据文件的无标签字段支持 ID 与话语长度映射，不足以恢复原始声学提取与选择步骤。

## 本轮结论和范围

归档核验、发布特征的全覆盖数值比对、真实官方网络的部分 clean 恢复三类证据分别保存，不能互相替代：

1. 完整归档身份：官方 SHA-256 和成员哈希。
2. 发布表示身份：13708×300 个真实话语特征逐维相同，详见 [300 维对齐](AUDIO_300D_PARITY_REPORT.md)。
3. 1611→300 可执行性：3 个完整 Validation 对话 / 24 条真实话语使用原 HDF5，误差约 1e-6；不含原始音频提取，详见恢复报告。

Test 仅参与无标签特征身份比对；未使用 Test 标签、未计算预测性能或校准指标。原归档、HDF5、pickle、完整原文及逐话语结果仅留本地，公开交付只包含报告与脱敏统计。原始 XP04 文件哈希前后复核，不对历史文件作补丁。
