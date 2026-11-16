# XP04 特征提取链路审计

结论：**两数据集的文本与音频真实输入重提取均不可按当前已核实资产启动。** CMERC checkpoint 完整；缺口位于原始输入到发布特征的上游。

## 精确 CMERC 锚点

XP04 根目录：`experiments/xp_mainline_validation/20261016T050823Z/`。

| 数据集 | checkpoint（相对 XP04 根目录） | SHA-256 |
|---|---|---|
| IEMOCAP | `runs/cmerc/IEMOCAP_validation_seed1475/best.pt` | `2c0d1a7a39627e12d1244d23a225ad3da03bbce071a118909921a547a2bfd106` |
| MELD | `runs/cmerc/MELD_validation_seed67137/best.pt` | `3f79f9eb3547e35ac15ad7ff4345d9b873d0c884a6f3a1ce60a7a7e47077e701` |

对应配置为 `configs/effective/IEMOCAP_cmerc_validation_seed1475.json`（SHA-256 `59e40ae7b7d63bb5f56806eedefb2f70c6797095455f63c083917b1141275369`）和 `configs/effective/MELD_cmerc_validation_seed67137.json`（`5d6271460c882dc34bf55bbcdec96afd09bb6b963c238745e04a54fb20515f68`）。本轮逐项复核配置登记的全部源码、数据和 curriculum 文件哈希，无偏差。只读取 checkpoint 字节计算哈希，没有加载模型执行推理。

模型既有归属为本地 **user23 / TITAN Xp**，依据 XP04 `reports/EXECUTION_PLAN.md:11`、`batch_manifest.json:62`。本轮无 GPU 作业，没有迁移到 biggpu。

## 文本

冻结 `code/dataloader.py:106–111,157–163` 只加载四组离线 1024-D 数组，不调用 tokenizer 或原始编码器。

结构匹配的参考生产脚本位于：

`$REFERENCE_FEATURE_EXTRACTION/`

`roberta_feature_extract_iemocap.py:8–12` 和 `roberta_feature_extract_meld.py:8–12` 分别要求 dataset fine-tuned `checkpoints/{iemocap,meld}/checkpoint_best.pt` 及 `{iemocap,meld}-bin`。本次检查这些路径均不存在，训练初始化要求的 `roberta.large/model.pt` 也不存在。目录中有 `encoder.json`、`vocab.bpe`、`dict.txt`，但没有它们与 XP04 发布 pickle 相联系的历史 manifest/哈希，不能认定为原 tokenizer。

参考脚本显示：Fairseq RoBERTa-large；整条话语编码；`pad_idx=1`；最后四个 hidden states 的 BOS 向量各 1024 维；MELD 先 ASCII-ignore。其 fine-tuning 脚本包含 512 位置和截断设置，但历史 extraction 的实际超长处理仍未确认。参考实现和维度一致不等于发布特征来源已经确认。不能换成 Hugging Face 预训练 RoBERTa 或重新微调一个编码器后继续声称是原 XP04 流程。

参考脚本默认遍历 train + validation + test，不能直接执行。本轮只读源码，没有运行它。恢复后也须限制实际重提取的输入范围，只输出目标话语 T1–T4，并先验证 clean parity、token 长度、padding 及其他话语向量不变。

## 音频

冻结 `code/train_cm.py:999–1000` 只有 `IS10=1582` 和 `MELD_audio=300` 的名称/宽度；`code/dataloader.py` 直接读取已提取向量。原始音频 extractor、模型或工具版本、特征名与顺序、采样率/声道处理、切段规则、帧参数、池化、归一化、缺失策略及生产命令未找到可追溯版本。不能仅凭 IS10 名称自行选择 openSMILE 配置，也不能用其他项目的音频编码器代替 MELD 300-D 提取器。

证据与搜索边界：20261110 `AUDIO_FEATURE_EXTRACTION_AUDIT.md:12–18`、冻结 XP04 源码、外部原 CMERC/M3NET 来源，以及当前工程与 COSMIC 参考目录。上述已审范围没有发现解除缺口的新资产；不声称机器任意位置绝对不存在权重，没有擅自扫描远程服务器。

## 目标替换的可复用部分

已有 `evidence_sensitivity/20261108T095843Z/scripts/run_evidence.py:77–79,97–100,115–148` 支持保持原 batch16、padding、speaker/mask 和对话图，仅替换一个话语的目标槽位。文本槽位 0–3、视觉 4、音频 5；退出时恢复并核对全部输入、冻结参数及 buffer 不变、无梯度。此控制思路可复用，但尚不能生成真实干扰对应的 XP04 特征。本轮未执行该 runner。

## 解除阻断所需材料

1. 每数据集生成发布文本特征的原 fine-tuned RoBERTa checkpoint、tokenizer/data dictionary、Fairseq 版本、代码版本、清洗/截断命令及可追溯哈希。
2. 两套原音频 extractor 的代码、配置、所需权重和环境，以及原始片段 ID/边界、采样率、声道、归一化与输出顺序 manifest。
3. 同版本 MELD dev 缺失片段及来源；先检查本地 dev 压缩包目录，若仍缺失，从原数据源补齐精确版本，不能使用同名 Test 文件。
4. 在少量不修改的 Validation 话语上重提取 clean 特征，与冻结特征逐维比较。记录精确相等/误差、cosine、是否影响原概率及约定容差；仅维度相同不算通过。特征 parity 未确认时不能用额外漂移冒充干扰效应。
5. 通过小规模目标替换 smoke check，核验其他模态、其他话语、上下文、speaker/label、CMERC state 和原文件哈希均不变，然后才启动登记的 120 条。

资产现场检查见 `provenance/producer_asset_inventory.json`；本轮没有下载、安装依赖、训练或构造替代提取器。
