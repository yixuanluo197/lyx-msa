# M³Net Baseline 配对权重审计

审计日期：2026-10-25。结论：**Baseline comparison pending: matched checkpoint unavailable**。

IEMOCAP、MELD 均未找到来源、训练/选模记录与作者官方 CMERC 可验证匹配的 Baseline checkpoint。因此不执行实验二及严格 Baseline–CMERC 对照，不训练新 Baseline，不用已有 Validation Baseline 代替。这里的结论是“未核验到符合本任务要求的权重”，不是“世界上不存在公开 M³Net 权重”。

## 1. 外部官方来源核验

实际读取并保存了两个官方仓库的 README、完整 recursive git tree、releases、tags、train.py；小型原始证据与来源版本保存在 `baseline_audit/`，SHA-256 见 `baseline_checkpoint_inventory.json`。

| 来源 | 固定代码版本 | 核验结果 |
|---|---|---|
| [M³Net 官方仓库](https://github.com/feiyuchen7/M3NET) | `4d0fa88cd4f92c72b053527e877763f54e29ccb2` | README 声明有 RoBERTa-IEMOCAP 预训练权重目录；git tree 只有 8 个源码/文档文件，无权重；releases、tags 均为空。 |
| [CMERC 官方仓库](https://github.com/HITSZ-HLT/CMERC) | `815c7c060914f5b55db816c0ba3662b11d08552f` | README 发布 IEMOCAP/MELD 两个 CMERC 权重，Baseline 阶段要求训练 M³Net，并将 `YOUR_M3NET_MODEL` 传给难度计算；没有单独、明确标注的配对 Baseline 发布链接或选模清单。 |

M³Net 的 [固定版本 README](https://github.com/feiyuchen7/M3NET/blob/4d0fa88cd4f92c72b053527e877763f54e29ccb2/README.md) 提供 [Dropbox checkpoints 分享目录](https://www.dropbox.com/sh/gd32s36v7l3c3u9/AACOipUURd7gEbEcdYSrmP-0a?dl=0)，当前跳转到 [新分享 URL](https://www.dropbox.com/scl/fo/nrgfqwuo4sa08gzxzu2vk/AJGRsP0l_bIwPoAlmd363t8?rlkey=zw9mh8uiuq5nmk7ihxtsj8lse&dl=0)。普通 HTML 仅为 JavaScript 外壳，随后通过 `dl=1` HEAD 和限量 GET 确认 ZIP，最终在 60 秒下载上限内取得完整官方分享 ZIP。biggpu 直连失败（curl exit 7），因此从本地主机下载后通过 SSH 复制到 biggpu 独立候选目录；没有将权重放入诊断报告目录或 Git。

ZIP 完整 SHA-256 为 `92c887f02fba3e74843b2b292259aa295123e8d5e293e6eaa05dd869afa17cab`，78,816,511 字节。其完整成员列表确认只有一个根目录和 `Roberta_IEMOCAP1.pth.tar`、`Roberta_IEMOCAP2.pth.tar`、`Roberta_IEMOCAP3.pth.tar` 三份 IEMOCAP 候选，没有 MELD 权重。未比较三份测试表现或挑最优，预先固定为头部首先识别的 `Roberta_IEMOCAP3.pth.tar`，仅对该精确文件名安全提取，不执行不受控 `extractall`。

实际候选路径：`biggpu:/data2/yb/paper/cmerc_reproduction_20261013/checkpoints/baseline_candidates/m3net_official_20261025/Roberta_IEMOCAP3.pth.tar`。

- 大小：26,271,973 字节；SHA-256：`fb939de1f88855fda46a9d491e5cce4400c88c7187f3b792bb40f61c4ed03faa`。
- 在已验证环境 PyTorch `1.13.1+cu117`，用 `torch.load(weights_only=True, map_location="cpu")` 实际加载成功；未用 unsafe fallback。
- 结果为纯 `OrderedDict`，85 个值全部为张量；无训练 epoch、seed、配置、checkpoint-selection 或对应 CMERC run 的元数据。
- 与冻结官方 CMERC IEMOCAP state_dict 对照：无缺失 key、无额外 key、无 shape 不兼容，85/85 完整结构兼容；其中只有 23 个张量逐值相等，因此不是 CMERC checkpoint 的相同副本。该逐值统计不用于判断训练身份。
- 与 MELD state_dict 不兼容：多出 2 个 key，29 个共同 key 的形状不同，不能用作 MELD Baseline。

来源与结构证据足以将其称为**M³Net 作者官方发布的 IEMOCAP 预训练候选**，而不是本地训练 Baseline；但仍不能称为与本次官方 CMERC 严格匹配的配对对照，因为缺少实际训练/选模记录和对应关系。完整下载 UTC、URL、ZIP 成员、哈希、安全加载方式和逐 key 形状检查结果已保存于 `baseline_audit/m3net_download_manifest.json`、`m3net_candidate_cpu_inspection.json`。未执行候选 GPU/CPU 模型前向或测试集选权重。

另外已逐文件比较 M³Net 固定版本源码与冻结官方 CMERC：`model.py` 完全同 SHA；`model_hyper.py` 的差异为默认关闭的上下文 mask 扩展；`dataloader.py` 的差异为数据路径和默认关闭的课程筛选分支，测试 keys、类别注释、张量返回/拼接方式一致。两仓库 README 使用相同多模态特征分享链接；本地 `/data2/yb/paper/M3NET` 四个特征文件与本次 CMERC 四个特征文件实际 SHA 均一致。以上验证说明结构与可用数据协议兼容，但不能证明某份纯 state_dict 的真实训练历史。源码 SHA、diff、特征 SHA 详见 `m3net_cmerc_source_comparison.json`、`m3net_to_cmerc_*.diff`、`local_m3net_feature_comparison.json`。

在本次实际核验的 M³Net README/tree/releases/tags 与取得的权重 ZIP 中，没有找到 MELD checkpoint 发布；未穷尽所有外部托管位置。

## 2. 现有本地与服务器资源

本轮只读枚举范围：

- 本地 `/data2/yb/paper/lyx-msa/CMERC`：273 个 `.pt/.pth/.ckpt/.pkl` 文件（排除 data、官方冻结代码等目录；并非逐个加载）。
- 本地 `/data2/yb/paper/M3NET`：实际检查文件树，只有源码、特征文件与 `record_*.pk` 等结果记录，没有 `.pt/.pth/.ckpt/.tar` 模型权重；不能将特征 pickle 或分数记录视为 checkpoint。
- `ssh biggpu` 的 `/data2/yb/paper/cmerc_reproduction_20261013`、`/data2/yb/paper/cmerc_seed_v2_20261014`：枚举 84 个上述候选文件，其中包括作者 CMERC 权重副本、本地训练状态、难度文件和 smoke 权重，文件名本身不构成身份依据。
- `ssh biggpu` 的 `/data2/yb/paper/cmerc_seed_20261014`：额外枚举 28 个 `.pt/.pth/.ckpt` 文件，属于历史多 seed、smoke 或恢复验证目录；相关 Baseline run manifest 单独留存。
- 复用此前 `../checkpoint_inventory.json` 中已在 CPU 检查真实训练状态得到的身份记录；本轮重新计算 8 个 Baseline 文件的 SHA-256，均与原记录一致。
- 实际读取 V100 multiseed_validation_v2 的 10 份 Baseline run manifest；训练 flags 均关闭校准/对比/课程模块，selection_protocol 为 validation、RNN backend 为 aten、训练 batch_policy 为 capacity_safe。这些是本地复现 run，不是 CMERC 作者公开权重配对记录。

审计不是整个服务器所有用户目录的全盘搜索；枚举之外的其他种子、latest 状态、失败任务不逐个反序列化，不根据名称赋予身份。原始路径清单、文件字节数及 10 份 manifest 的相关字段见 `baseline_audit/*inventory.json`、`v100_multiseed_baseline_manifests.json`。

## 3. 已验证身份的本地 Baseline 为什么不纳入

下表的身份来自**此前对可信本地训练状态内部 args 的实际检查**：`calibrate=false`、`contrastlearning=false`、`courselearning=false`；本轮只读复核文件哈希，没有为了找更优测试分数重新选 checkpoint。

| 文件/族 | 数据集 | 训练来源与选择 | SHA-256 | 本次处理 |
|---|---|---|---|---|
| `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/IEMOCAP_validation_seed1475/best.pt` | IEMOCAP | 本地 xp_mainline_validation；validation；epoch 70 | `bef27e03c3e72ba32036de0ae425b03a5826b27181d0812f4028c8cb062b0c31` | 不作为作者配对 Baseline |
| `/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z/runs/baseline/MELD_validation_seed67137/best.pt` | MELD | 本地 xp_mainline_validation；validation；epoch 5 | `22a44516e58fc6f613879085d902cfb7e572c05fb2b50c06d16f4f1cf20bb96e` | 不作为作者配对 Baseline |
| `/data2/yb/paper/lyx-msa/CMERC/runs/formal/baseline/IEMOCAP_official_test_seed1475/best.pt` | IEMOCAP | 本地 historical_formal；official_test；epoch 48 | `df67ac80ae51be37e0e36d1fac6e5cc71af21033191275d727a51ba821a6988a` | 不作为作者配对 Baseline |
| `/data2/yb/paper/lyx-msa/CMERC/runs/formal/baseline/IEMOCAP_validation_seed1475/best.pt` | IEMOCAP | 本地 historical_formal；validation；epoch 77 | `6d83f492dbcc0a2063d7eb146da865c9d155afe0dbcd053d86b31b61ed1decfe` | 不作为作者配对 Baseline |
| `/data2/yb/paper/lyx-msa/CMERC/runs/formal/baseline/MELD_official_test_seed67137/best.pt` | MELD | 本地 historical_formal；official_test；epoch 4 | `c458dbb7b51d9c9294c2abb0b1c7a1dbea117375b04acbaaf360c5b3ca9e7db0` | 不作为作者配对 Baseline |
| `/data2/yb/paper/lyx-msa/CMERC/runs/formal/baseline/MELD_validation_seed67137/best.pt` | MELD | 本地 historical_formal；validation；epoch 6 | `c9cc2eb7578b38e279d6e73d391494eb498edb2ddff9cba3d5a3886ab48af86e` | 不作为作者配对 Baseline |
| `/data2/yb/paper/lyx-msa/CMERC/experiments/official_readme_reproduction/checkpoints/IEMOCAP_baseline_seed1475/best.pt` | IEMOCAP | 本地 official_readme_reproduction；official_test_selection；epoch 72 | `39d2338cb074786405f9b82fabf29cc10fdebdd2342ff66ae21afc8773ea58d8` | 不作为作者配对 Baseline |
| `/data2/yb/paper/lyx-msa/CMERC/experiments/official_readme_reproduction/checkpoints/MELD_baseline_seed67137/best.pt` | MELD | 本地 official_readme_reproduction；official_test_selection；epoch 5 | `a6ab85cb6c0c12054dade9c6bcdf01cfaa6abc417625059a927de5ba386d3738` | 不作为作者配对 Baseline |

XP Validation Baseline 的数据哈希、测试 ID、关闭附加模块、种子、Validation 选模、ATen 后端有实际证据；它们可以支持此前的本地 Validation 对照。它们与本次作者权重存在训练/选模来源及 RNN 协议差异，作者 CMERC 的实际训练 seed、epoch、拆分、选模记录未发布，不能把这两个来源混为严格同条件配对。

历史 `official_test` 与 `official_readme_reproduction` Baseline 另有测试集选模问题，即使模型结构和评估输入兼容，也不能通过选择其中测试分数更好的一份来充当匹配对照。V100 训练产物仅因服务器相同也不能自动取得配对资格。

## 4. 官方评估与选模风险

[M³Net 固定版本 train.py](https://github.com/feiyuchen7/M3NET/blob/4d0fa88cd4f92c72b053527e877763f54e29ccb2/train.py#L492) 每个 epoch 计算测试集 F1 并追加到 all_fscore，507 行按 test_fscore 更新 best_fscore，532 行报告 max(all_fscore)。[CMERC 固定版本 train.py](https://github.com/HITSZ-HLT/CMERC/blob/815c7c060914f5b55db816c0ba3662b11d08552f/train.py#L498) 也有对应逻辑。这证明公开训练/结果汇总代码存在测试集选模风险，**不证明作者发布的某个纯 state_dict 一定由该风险流程选出**。

作者 CMERC 的 85/83 个参数张量为纯 state_dict；没有内部 epoch、seed、selection 或配对 Baseline 身份元数据。配置示例、成功复现论文四舍五入 F1、参数维度相同，均不能恢复缺失的历史记录。本次仅使用预先固定的两个作者权重，复用完整输入预测，没有用测试集搜索权重或配置。

## 5. 完成严格对照所需资源

1. IEMOCAP 与 MELD 的作者 M³Net Baseline 原始权重，明确来源、不可变 SHA-256 及对应作者 CMERC 的配对关系，或作者确认可使用的对照权重。
2. 对应代码版本、完整模型配置、文本/音频/视觉特征哈希、类别映射、训练/验证/测试拆分与逐话语顺序记录。
3. 训练 seed、训练 epoch、checkpoint 选择准则及所用划分；如果含测试集选模应明确披露，不能补写成 Validation 选择。
4. 使用本次固定 full/均值替换协议、batch 16、Padding/Mask、官方 cuDNN 后端与已验证兼容补丁，可严格加载并在相同 1623/2610 条测试话语上评估的证据。

满足资源与来源/协议验证后才能补充实验二、Baseline 完整输入与模态删除、错误分布及校准对比。本轮这些对照保留为 pending；不会为了补齐表格重新训练或用旧 Validation 结果填空。
