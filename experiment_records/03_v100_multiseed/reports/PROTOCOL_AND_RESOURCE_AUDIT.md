# 多 seed 协议与资源审计

检查日期：2026-10-14 UTC。原工程 `/data2/yb/paper/lyx-msa/CMERC` 保持原状；新工程位于 `CMERC（seed）`。V1诊断目录 `/data2/yb/paper/cmerc_seed_20261014` 保留；正式V2使用 `/data2/yb/paper/cmerc_seed_v2_20261014`。

## 非因果路径

快照来自 `91a1bffa763d57015896dee1014f0d11c5d09029`，上游官方版本 `815c7c060914f5b55db816c0ba3662b11d08552f`。该快照包含可选因果分支，但本批入口明确禁止 `--causal`，所有模型使用原BN双向非因果路径；原模型文件、分类损失、CC/HSCL、模态移除和课程公式均不修改。MELD沿用官方非标准FocalLoss，不改成新的损失。

真实官方checkpoint复核（同特征、batch16、原始信息移除）已通过：

| 数据集 | 新快照W-F1 | 新快照CEL | 论文CMERC W-F1 / CEL |
|---|---:|---:|---:|
| IEMOCAP | 71.98379358 | 1143.49284046 | 71.98 / 1143.50 |
| MELD | 66.84839220 | 605.05028273 | 66.85 / 605.05 |

新结果保存在 `results/preflight_official`，不计入多seed训练统计。

## 随机过程与划分

旧代码将训练seed用于IEMOCAP验证划分，且每次train/eval调用重新执行seed_everything。其结果是每轮dropout重用相同RNG起点；validation loader每轮重建同seed generator也会重复排列。旧单seed结果不修改。

新多seed明确使用 `rng_policy=continuous`：模型初始化时播种一次，训练推进全局RNG；评估保存恢复Python/NumPy/Torch/CUDA状态；训练DataLoader用training_seed+零基epoch派生generator，worker用torch.initial_seed派生Python/NumPy状态。采样可按epoch重建，checkpoint记录全局RNG和epoch，恢复后可重建下一epoch采样。

IEMOCAP split-seed恒定1475，108训练/12验证/31测试对话，两模型所有seed共享同集合；MELD使用官方1038训练/114验证/280测试对话。保留官方难度文件并固定哈希/排序；难度文件生成过程未完全公开，因此无法证明该外部课程评分的生成完全排除本轮holdout，仅能保证本轮训练与选模不混入validation/test。此边界须随最终结果保留。

连续RNG改变了随机训练过程而未改变模型数学定义。新旧单seed结果差异含RNG策略影响；IEMOCAP除1475外，旧seed绑定split的结果也含划分影响，不能解释为纯seed波动。CUDA scatter类原子算子仍可能有浮点非确定性，不能承诺跨GPU逐位一致。

真实GPU恢复测试进一步发现：PyTorch1.13.1的cuDNN GRU首次训练会从CPU RNG抽取一次seed，并在C++进程内缓存/推进dropout状态，该buffer不进入普通模型/RNG checkpoint。因此仅保存全局RNG不足以严格恢复。原cuDNN恢复预检失败并保留在 `runs/preflight_resume`；没有把这一失败计作完成。

本组continuous模式只在GRU/LSTM前向使用ATen执行同一数学操作，BN、图运算、GRU结构和dropout概率保持原设置；checkpoint明确记录 `rnn_backend=aten` 并在恢复/独立评估时检查一致性。legacy模式仍保留cuDNN。依据为 [PyTorch1.13.1官方RNN实现](https://raw.githubusercontent.com/pytorch/pytorch/v1.13.1/aten/src/ATen/native/cudnn/RNN.cpp) 的 `get_dropout_state`。

修复后的真实IEMOCAP完整CMERC两epoch对照，在GPU1连续训练与GPU2首轮checkpoint后中断/恢复之间，CPU/CUDA RNG完全一致；最大参数差 `3.3285468816757202e-06`，在预先设置的rtol1e-3/atol1e-4以内。验证产物 `runs/preflight_resume_aten/verification.json`。最小GRU单卡三进程恢复对照还通过逐位参数一致检查。ATen执行后端是明确的工程协议变化，须与旧单seed分数区分解释。

## 预注册矩阵与预算

| 阶段 | seeds | IEMOCAP Baseline / CMERC | MELD Baseline / CMERC | 完整run数 |
|---|---|---|---|---:|
| 第一轮 | 1475,67137,2024 | 每seed各80epoch | 每seed各40epoch | 12 |
| 稳定性门控后扩展 | 3407,42 | 每seed各80epoch | 每seed各40epoch | 8 |

阶段门控检查完整epoch、有限损失/预测、相同split、同seed配对、causal=false、validation选模及产物完整性，不设test分数门槛。最终20runs共1200epoch，报告所有5seed、样本SD(ddof=1)、成对增益。

初查GPU1/2/3/5空闲且uncorrected ECC=0、无pending retired pages；GPU0运行其他实验，GPU6/7有既有满载进程，GPU4明令禁用。每次派发前重新查询，一卡一个train/eval任务，显式CUDA_VISIBLE_DEVICES=宿主整数卡号并记录UUID。占用变化时排队，不终止他人任务。

按历史预算IEMOCAP peak6–9GiB、MELD约1GiB；白名单V100各32GiB，启动要求至少12GiB空闲。首轮保守2–4小时、全部4–6小时墙钟（待真实新调度更新）；checkpoint约3.8GiB，含源码、日志、预测、smoke预算6GiB，启动前biggpu /data2剩余23GiB。

## 启动门控

必须通过CPU回归测试、原checkpoint校验、真实短训练/恢复和调度smoke后再启动正式队列。所有smoke使用独立输出路径；正式阶段保存run_manifest、source/data/split哈希、GPU/PID/命令、逐epoch损失指标和full checkpoint。未经完整训练的任务不得进入正式统计。

## 已执行的启动验证

- 专用torch1.13.1环境CPU suite：182 passed、1 skipped（该GPU-only单测由实际GPU预检另行覆盖）。
- 原官方两个checkpoint的W-F1/CEL对齐；所有shell/Python编译通过。
- `--dry-run`生成20项可执行任务矩阵；冻结30个源码/配置文件，snapshot SHA256为 `98db3b561e9d65bcf54a4390322d99e0dabc01430f78f6da040d2ce7f1c70ada`。
- 真实调度smoke：seed1475、两数据集×两模型共4run，各1完整epoch与独立评估；状态为completed并通过paired/source/data/split/finite门控。宿主GPU映射为1/2/3/5，一卡一任务。
- 2026-10-14 17:25:50 UTC首轮正式队列已启动，调度PID1510957。初始四个train PID分别1511539、1512177、1512415、1512591，固定宿主GPU1、2、3、5；实际进度以 `state/multiseed_validation_queue.json` 为准。
- 本地tmux会话 `cmerc_seed_artifact_sync_20261014` 每60秒只读同步正式状态、日志、预测/CSV/JSON/报告；终态再同步完整checkpoint。此同步流程不修改冻结训练源码。

首轮12项与后续8项目前仍需完整训练；不把已通过的smoke或运行中的best checkpoint视为五seed统计完成。最终中文报告由调度器根据真实完整产物自动更新。

## V1诊断退出与V2统一容量协议

V1第一轮实际触发原模型固定参数容量错误：IEMOCAP baseline seed67137第6epoch第6batch含1001 utterances，超边为1001+3×16=1049，超过 `model_hyper.py` 固定1000个hyperedge_weight；产生CUDA device-side assert。用相同真实DataLoader顺序在CPU重放已确认。不是标签/划分错误，不通过扩容模型、减batch或删样本修复。

V1调度器已停止；仅向本工程自己的剩余训练PID发SIGTERM，保留可恢复latest、日志和全部已完成run。IEMOCAP baseline1475完成80epoch、MELD baseline1475完成40epoch；另有失败与中断任务。具体退出状态在旧 `state/v1_retirement.json`，诊断报告在 `attempts/v1`。它们均不进入V2五seed均值；不会因为分数低/高删选。

V2所有seed与两个模型统一使用capacity_safe批次协议：候选为原seed+epoch完整随机排列，每批仍16个对话，末批大小不变，每个训练对话仅出现一次；若任一批超出固定1000边或5200节点，拒绝整轮排列后重抽。仅使用训练对话长度与独立generator，不用标签、验证/测试分数或模型RNG。上限10000次，耗尽则报错，不静默drop。checkpoint、manifest、每epoch日志及评估均记录/校验策略、容量和实际重排次数。

真实5seed×80/40epoch×两模型的1200个训练排列扫描通过：IEMOCAP800个排列中779次无需重排、19次重排1次、2次重排2次；最大998超边。MELD400个排列全部无需重排。原失败的seed67137第6epoch重排一次后为915超边/2601节点。固定划分及batch16全样本覆盖均通过，valid/test原固定顺序也需验证后启动。V2是明确不同的条件批次随机协议，必须与旧单seed/V1分开解释，但所有新seed彼此同协议。

容量修复后本地CPU suite为196 passed、1 skipped；专用环境与真实GPU恢复/调度短测仍须在V2新源码冻结后重新验证，不能以V1通过替代。实际ATen运行中IEMOCAP allocated峰值约5–9GiB但CUDA缓存可占用接近30GiB；严格整卡独占，无并卡实验。此时biggpu /data2实际余量约19GiB；保留V1同时预算V2约6GiB，启动与每次派发均检查磁盘余量。

## V2实际启动验收

- 专用torch1.13.1环境重新运行CPU suite：196 passed、1 skipped。真实1200排列审计保存 `results/preflight_v2/batch_capacity_audit.json`；原validation/test固定批次容量均安全。
- 同一V2完整CMERC真实两epoch中断/恢复：RNG完全相等，最大参数差8.778180927038193e-06，通过预设rtol1e-3/atol1e-4。证据 `runs/preflight_resume_capacity_v2/verification.json`，不是正式结果。
- V2四项两数据集×两模型真实1epoch调度短测全部完成训练和独立评估；2026-10-14 17:54:53 UTC通过配对/source/data/split/finite门控。实际GPU为1/2/3/5，未用GPU4。
- 冻结31个执行源码/配置文件，snapshot SHA256 `4f8cd5e6c9fb4f54d7280ed3989942f4a01018f3d496452226c4757dba3b0e33`，每次派发重新校验。旧V1冻结快照保留于 `attempts/v1/snapshots`。
- V2首轮正式队列已启动，调度PID1638305，状态 `state/multiseed_validation_v2_queue.json`。运行/评估/统计独立采用 `multiseed_validation_v2` 路径。首轮12项全部完整通过后，才扩展预注册后两seed；门控不使用测试分数阈值。

当前状态为正式训练进行中，不是五seed复现完成；有效正式指标以自动报告与CSV/JSON为准。

### 首个完整正式产物与容量修复实证

V2 MELD baseline seed1475已完整40epoch并完成独立checkpoint评估：validation-selected epoch=6，Acc=66.3601532567、W-F1=65.4929824167、Macro-F1=49.3872317864、CEL=601.9534370452、ECE=25.8668826355%。调度器自动汇总退出码0，机器汇总有效run=1/20，三张真实SVG与CSV/JSON均成功产生。这里只是一个正式seed，不能视为三seed/五seed均值。

原容量失败的IEMOCAP baseline seed67137已在V2真实GPU训练通过第6epoch：sampler_retries=1、max_batch_edges=915、max_batch_nodes=2601，total_loss=0.9274960416；后续第7/8epoch也正常完成。与CPU预注册扫描一致，未扩展模型参数或删样本。当前队列没有failed任务，其余run仍训练/排队中。
