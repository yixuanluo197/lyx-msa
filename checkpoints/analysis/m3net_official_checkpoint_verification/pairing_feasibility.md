# M³Net 与官方 CMERC 的公开权重对照可行性

本轮结论：**IEMOCAP 公开权重行为对照可行；严格训练配对尚未确认。** 两份权重在完全相同的测试配置、特征和 1623 条话语上可以逐样本测量预测行为。缺少训练种子、训练轮次、选模记录和两者的训练对应关系，不能将两份权重间的差异直接归因于 CMERC 的课程学习、对比学习或置信度约束。

本文件只评估对照可行性。未执行实验二的完整纠错分析、模态删除或概率校准诊断。

## 1. 预先固定的权重与真实推理结果

M³Net 使用作者公开分享目录中预先固定的 `Roberta_IEMOCAP3.pth.tar`，没有测试另外两份候选，也没有根据本次成绩更换权重。

- 服务器路径：`biggpu:/data2/yb/paper/cmerc_reproduction_20261013/checkpoints/baseline_candidates/m3net_official_20261025/Roberta_IEMOCAP3.pth.tar`。
- SHA-256：`fb939de1f88855fda46a9d491e5cce4400c88c7187f3b792bb40f61c4ed03faa`；26,271,973 字节。
- 来源：[M³Net 固定版本 README](https://github.com/feiyuchen7/M3NET/blob/4d0fa88cd4f92c72b053527e877763f54e29ccb2/README.md) 中的预训练权重分享链接；原下载与 ZIP 成员证据保留在 `../high_confidence_errors/official_checkpoint_analysis/baseline_audit/m3net_download_manifest.json`。
- 安全 CPU 加载为纯 `OrderedDict`，85 个张量；严格加载全部匹配，不使用 `strict=False`。纯权重没有训练 epoch、seed 或 checkpoint selection 元数据。

本次 M³Net 推理于 2026-10-25 07:40:26–07:40:37 UTC 在 `biggpu`/`user22` 执行。CMERC 一行直接复用此前已验证的官方预测，没有再次运行 CMERC 推理。

| Dataset | Model | Accuracy (%) | Weighted F1 (%) | Macro F1 (%) | 有效话语 |
|---|---|---:|---:|---:|---:|
| IEMOCAP | Official M³Net | 72.45841035120148 | 72.66013116562415 | 71.5532217841169 | 1623 |
| IEMOCAP | Official CMERC | 71.90388170055454 | 71.98379358245155 | 71.37466315596619 | 1623 |

本轮固定 M³Net 的 W-F1 比现有官方 CMERC 高 **0.6763375831725966 个百分点**。该事实描述这两份已固定公开权重的测试表现，不用于选择新的 checkpoint，也不能单独证明或否定 CMERC 各模块的训练因果效果。两个模型的完整概率分别保留，不要求不同权重产生相同概率。

## 2. 结构与数据协议的实证比较

| 比较维度 | 结论 | 实际证据与限制 |
|---|---|---|
| 模型结构 | 兼容 | M³Net 固定版本 `4d0fa88cd4f92c72b053527e877763f54e29ccb2` 与 CMERC 固定版本 `815c7c060914f5b55db816c0ba3662b11d08552f` 的 `model.py` 逐字节相同；85 个权重张量完整严格加载。 |
| 超图模型默认路径 | 相同 | CMERC `model_hyper.py` 新增默认 `mask_context=None` 的可选上下文删除分支；本次默认不触发。只排除这些默认关闭的分支及其参数后，两份完整模块 AST 相同。 |
| 超图卷积原文件 | 相同 | 两仓库的 `HypergraphConv.py` 官方 git blob 都是 `bdfde7ddf96172c407baba93d4784a645ccb2c60`。 |
| DataLoader | 默认测试路径相同 | 两份 `get_IEMOCAP_loaders` AST 完全相同。dataloader 的差异为数据路径和默认 `epoch_ratio=-1` 关闭的课程筛选；归一这两类差异后完整模块 AST 相同。 |
| 数据集、特征文件 | 相同 | 当前重新计算本地 M³Net 与 CMERC 两套 IEMOCAP 特征文件哈希，原特征和 RoBERTa 文件均逐字节同 SHA；本次远程运行也验证固定输入哈希。 |
| 六分类映射 | 相同 | `happy, sad, neutral, angry, excited, frustrated` 对应整数 `0,1,2,3,4,5`。 |
| 测试样本 | 完全对齐 | 两个 CSV 均为 1623 条唯一话语；dataset、dialogue_id、utterance_index、source_utterance_id、原文、标签、真实情绪、输入条件逐行完全一致。 |
| Batch 与 Padding | 相同 | batch size 16、num_workers 2；31 个测试对话按原 testVid 顺序分成 16 与 15 个对话的两批，分别 861 与 762 条有效话语；padding 长度分别 91 与 83。两次运行记录的 batch IDs、长度及张量形状完全相同。 |
| 实际评估配置 | 相同 | 新旧 `official_argv` 和全部 `effective_args` 字典完全相同：GRU、hyper、concat_DHT、avl、BN、dropout 0.5、num_L=3、num_K=4、epochs=0、testing。 |
| RNN 与 CUDA 环境 | 相同 | Python 3.10.13、PyTorch 1.13.1+cu117、cuDNN 8500；CUDA/cuDNN 启用、deterministic=true；没有 ATen RNN 替换。 |
| 评估随机种子 | 相同 | 本次有效评估 seed 1475；M³Net 官方评估函数默认全局 1475，CMERC 传入 args.seed=1475，种子函数体 AST 相同。该数字不能被写成已知训练 seed。 |
| 训练种子 | 未知 | 两份纯 state_dict 均不能提供实际训练 seed。README 示例不构成该文件训练记录。 |
| 训练 Epoch | 未知 | 不从文件名、张量数量或成绩推断训练轮次。 |
| Checkpoint 选择方式 | 未知，源码存在风险 | 两个公开训练脚本都有按测试 F1 更新最优结果的逻辑；无法据此确定作者公开的具体文件是否按该流程选出。 |
| 与 CMERC 的训练配对关系 | 未确认 | 没有证据证明该 M³Net 文件是本次 CMERC 作者权重使用的对应基础模型、共享种子或同一训练链。 |

当前本地重新核验的 IEMOCAP 特征：

| 文件 | 大小（字节） | SHA-256 |
|---|---:|---|
| `IEMOCAP_features.pkl` | 72216498 | `ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3` |
| `iemocap_features_roberta.pkl` | 124438886 | `4bd60c581a5b7c70e217e18439f741b191605fbd6f2cf4dc1a6dcb7fcaa18f06` |

## 3. 共用执行代码及兼容补丁的边界

本次在与官方 CMERC 成功复现相同的冻结 CMERC testing 入口及模型中加载固定 M³Net 权重；只读加载包装将官方入口请求的 IEMOCAP 权重明确映射到预先固定的 M³Net 文件，不改写任何原权重。由于上述源码默认路径核验及 strict load 通过，这支持相同评估协议下的公开权重行为比较。它不等于恢复 M³Net 作者当时的完整软件环境。

两个模型共用已经记录的两处 PyG 兼容处理：删除未使用、当前版本不可导入的 `topk` 导入；修正 `HypergraphConv.py` 第二次反向消息传播的 `size=(num_nodes, num_edges)`。补丁没有增加或移除权重参数。不能声称已经在旧版作者环境中证明与未补丁实现完全数值等价。

双向 GRU 在裁去 padding 前先运行于 padded 张量，因此 batch 和对话顺序可能影响输出，本次没有修改它们。官方评估返回的 IDs 只含最后一个 batch；本次只读观察器直接读取每批 ID、长度与裁剪标签，未用这个不完整返回值关联全量样本。

本次记录为一次 strict load、两次完整输入 forward、零 optimizer step、零模态删除 forward、零 CMERC forward；参数及 BN buffers 前后哈希相同，梯度关闭且模型处于 eval 状态。

## 4. A：公开权重行为对照

**可行。** 本次已独立检查全部 1623 条话语的稳定 ID、原文、标签、原始顺序、六类概率列及同批协议。两模型每条概率有限、位于 [0,1]、概率和约为 1，argmax 与预测类一致，最大概率及真实类概率字段一致；分类指标由 CSV 独立复算与运行记录一致。

如果后续授权实验二，可以用 `(dataset, dialogue_id, utterance_index)` 对齐，分析固定两份公开权重的纠错、保持错误和置信度变化。结论须表述为“公开权重的行为差异”，不得命名为严格训练配对的模块增益。本轮只为下一步保存预测数据，没有提前执行完整实验二。

## 5. B：严格配对的模块效果对照

**尚不成立。** 需要补齐两份权重真实训练种子、训练/验证/测试拆分、完整超参数、训练 epoch、checkpoint 选择准则，以及该 M³Net 与官方 CMERC 的对应训练链或作者确认。若原流程使用测试集选模，应明确披露，不能补写成 Validation 选择。

仅凭结构兼容、特征相同、公开来源或本次分数，均无法排除训练随机性、不同训练时长、不同选模方式等解释。课程学习、对比学习和置信度约束的单独或联合因果收益，需要控制这些因素的配对训练实验；本任务没有授权重新训练。

## 6. 可复核证据

- `protocol_audit/source_protocol_comparison.json`：源码哈希、精确 AST 对比与默认关闭分支的归一说明、重新计算的特征哈希。
- `protocol_audit/independent_alignment_verification.json`：两份 CSV 的全部稳定字段逐行对齐、配置/batch 一致性、概率合法性和独立分类指标。
- `inference/IEMOCAP_run_manifest.json`：本轮实际运行环境、两次 full forward、冻结参数及 runtime 文件哈希。
- `../official_checkpoint_verification/IEMOCAP_run_manifest.json` 与 `IEMOCAP_predictions.csv`：复用的官方 CMERC 基准；没有重选或重跑。
- `checkpoint_audit.json`、`run_manifest.json`、`verification.json`：主实验清单与最终独立核验。
