# 官方 README 原始训练协议与 XP04 对照审计

审计对象为官方固定提交 `815c7c060914f5b55db816c0ba3662b11d08552f` 与 XP04 真正冻结源码及运行档案。本文件聚焦源码协议及XP04核验；本次正式结果、终态和归因边界见报告04–06。下文“官方”路径相对于 `experiments/official_readme_reproduction/official_code/`；“XP04 代码”路径相对于 `experiment_records/04_xp_mainline_validation/code/`。行号来自实际文件读取。

## 1. README 实际命令与入口

官方 `README.md:34–36,49–54,62–64,78–83` 的 baseline 与完整 CMERC **均执行 `train_cm.py`**。baseline 不传 `--calibrate --contrastlearning --courselearning`；完整 CMERC开启三项。`train.py` 是另一个应审计的官方入口，README主要在评估段使用它（`README.md:112,117`），不能以其默认值代替README训练命令。

| 参数 | IEMOCAP | MELD | 核验依据 |
|---|---|---|---|
| Seed / Epoch / Batch Size | 1475 / 80 / 16 | 67137 / 40 / 16 | README实际训练命令 |
| Learning Rate / Dropout | 0.0001 / 0.5 | 0.0001 / 0.4 | 同上 |
| Backbone / Graph / graph_construct | GRU / hyper / direct | GRU / hyper / direct | 同上 |
| Fusion / modals / Norm | concat_DHT / avl / BN | concat_DHT / avl / BN | 同上 |
| num_L / num_K | 3 / 4 | 3 / 3 | 同上 |
| 完整CMERC rank_coff | 0.005 | 0.002 | README:52,81 |
| 完整CMERC mscl_coff / cscl_coff | 0.05 / 0.05 | 0.15 / 0.15 | README:53,82 |
| 完整CMERC epoch_ratio / scheduler_steps | 0.15 / 1 | 0.4 / 1 | README:54,83 |
| 分类loss | 固定类别权重NLL | 官方FocalLoss实现 | 官方train_cm.py:690–711 |
| 命令未覆盖但实际生效：优化器 / l2 | Adam / 0.00003 | Adam / 0.00003 | 官方train_cm.py:402,713 |
| 命令未覆盖但实际生效：use_speaker / use_modal / use_residue / av_using_lstm | True / False / False / False | 同左 | 官方train_cm.py:432,440,446–448 |

四份 XP04 `run_manifest.json` 已逐一读取，以上网络/主要超参数一致；差异在下列实际执行协议，不能把“超参数一致”表述成“全部条件一致”。

## 2. 随机数初始化、重置及评估隔离

官方 `train_cm.py:37–44` 初始化Python/NumPy/Torch/CUDA RNG并设置cuDNN deterministic；`:533` 模型创建前调用；图训练/评估函数每次入口又在 `:192` 重置相同Seed。`train.py:18–25,167,379` 存在同样行为。因而官方每轮训练及每次评估从相同RNG状态开始；这不会自动证明cuDNN内部RNN dropout缓存也每次重置。

XP04 `train_cm.py:29–65` 在continuous模式评估前保存、评估后恢复RNG；`:615–616` 只在legacy模式重置。实际四份manifest均为 `rng_policy=continuous`。`:307–313` 为loader建立独立generator，continuous模式种子为 `seed+epoch`。潜在影响：dropout序列、训练排列及评估对后续训练随机流的影响改变；尚未隔离验证其分类性能作用。

## 3. GRU后端与非因果网络路径

官方直接使用 `nn.GRU`（`model_cm.py:615–631`），没有强制禁用cuDNN。XP04 `train_cm.py:69–92,1190–1191` 对RNN forward使用 `torch.backends.cudnn.flags(enabled=False)`，实际manifest为 `rnn_backend=aten`。GRU维度、双向与层数在非因果路径保留，但执行kernel、dropout随机轨迹及浮点累计可能不同，不能称逐位等价。

XP04源码包含causal扩展，但四次实际训练 `causal=false, effective_norm=BN`。`model_cm.py` 中条件causal分支不构成这四次运行使用了因果模型的证据。对照发现 XP04 `model.py` 与官方文件一致；`model_cm.py:41` one-hot构造从官方固定`.cuda()`改为跟随logits设备，未改分类公式。

共享工程差异另含：官方 `model_hyper.py:16` 的旧topk import在当前PyG不可导入；官方 `HypergraphConv.py:151` 第二次 `propagate` 的size从 `(num_edges,num_nodes)` 改为XP04的 `(num_nodes,num_edges)`。本次预检已分别复现import失败与第二次propagate维度失败，独立兼容补丁见 `provenance/compatibility.patch`。这些应标记为兼容性修改，不能将有补丁运行写成官方源码完全未修改。

## 4. 划分、过滤与训练数据量

官方 `train_cm.py:745–754` 实际传 `valid=0.0`，所以验证loader为空，训练使用全部原始trainVid。`train.py:468–475`同样。官方 `dataloader.py:19,70` 按原始feature文件trainVid/testVid取keys；MELD虽然在 `:66–68` 读取roberta的validIds，却没有采用其验证划分。

XP04 `dataloader.py:9–22` 使用固定RandomState(split_seed)对IEMOCAP对话划分，`train_cm.py:466–493` 先划分再建课程训练集；`dataloader.py:25–34,155–171` 使用MELD roberta trainIds/validIds/testIds。四份manifest记录split_seed均1475。

| 数据集 | 官方训练/验证/测试对话数 | XP04训练/验证/测试对话数 | XP04测试utterances |
|---|---|---|---:|
| IEMOCAP | 120 / 0 / 31 | 108 / 12 / 31 | 1623 |
| MELD | 1152 / 0 / 280 | 1038 / 114 / 280 | 2610 |

上述数量已读取XP04真实manifest与运行目录pickle核验。MELD raw trainVid的1152项对应roberta train1038与valid114；ID60在两者训练集合中均不存在，不能说XP04额外删除ID60。未发现为容量安全删除困难对话的实现。潜在影响：新旧训练量和具体可训练IDs不同，构成实质混杂变量。

## 5. Batch组织、容量采样与worker数

官方IEMOCAP `train_cm.py:88–92` 没有shuffle或train_sampler，按原始keys固定顺序组batch；MELD `:60–65` 使用SubsetRandomSampler，而函数入口每次reset RNG。`train.py:41–46,69–73`同样。官方正式循环使用 `num_workers=2`（`train_cm.py:746–754`）。

XP04 `train_cm.py:316–383` 的 CapacitySafeBatchSampler 从私有generator生成整个排列；若任何batch的 `sum(lengths)+3*B>1000` 或 `3*sum(lengths)>5200`，拒绝整条排列重抽。batch_size不变且每个合格训练对话每epoch都出现一次；不会通过截断、丢弃或减小batch来适配。实际 `batch_policy=capacity_safe, num_workers=0`。各epoch真实retries及最大边/节点数在 `runs/.../metrics.jsonl`。

潜在影响：IEMOCAP固定batch变为随机batch，以及MELD每轮私有排列均会改变BN统计、padding均值替换、HSCL正负样本集合。拒绝采样如发生还改变排列分布。这些均需独立对照，不能预先归因。

## 6. 分类、CC、MSCL与CSCL数学

官方 `train_cm.py:232–271` 的五种反事实移除与rank损失，对应XP04 `train_cm.py:656–707` 非因果分支；官方 `:272–346` 伪标签和HSCL调用，对应XP04 `:708–771`。XP04 `:777–787` 将总loss组合提取为helper，官方 `:348–358` 是原地组合，系数和加法顺序保留。CC为五种移除下true-class log-probability正差的**和**，不是样本均值。

`clloss.py` 的公式对照仅发现设备处理`.cuda()`改为`.to(features.device)`（官方 `:30–34,118–122,208–212,299–309,344–348`；XP04 `:30–32,116–118,204–206,293–305,338–342`），没有发现MSCL/CSCL数学被改写。两边 `ft_scl=hidden` 都在no_grad代码块内赋值；赋值本身不detach既有张量，不能据此声称HSCL没有梯度。

IEMOCAP分类loss在官方 `train_cm.py:690–711` 与XP04 `:1198–1219` 均为固定权重NLL；MELD为FocalLoss。该官方FocalLoss（官方/XP04 `model_cm.py:43–50`）使用log_p计算 `pt=onehot*log_p, sub_pt=1-pt`，再求 `-alpha*(sub_pt)**gamma*log_p`；其实现必须忠实保留，不能静默换成标准概率版focal公式。潜在影响主要来自输入、batch和优化轨迹，不应无证据归咎loss公式改动。

## 7. 课程难度来源、覆盖与README重算缺陷

已实际读取原运行目录 `experiments/xp_mainline_validation/20261016T050823Z/code/caldiff/` 的两个pickle，与固定官方commit的git blob逐字节比较完全一致：

| 文件 | SHA256 | 条目 | 覆盖XP04 train / validation / test |
|---|---|---:|---|
| IEMOCAP_diff.pkl | db4c7b96f169948a3615a62b29d2a793bd64cccd8a0387b1de9d7ceab348a150 | 120 | 108 / 12 / 0 |
| MELD_diff.pkl | db005ceb0e4847caffc8571eaa94ab7c5b4a310abd615b12c995c9a22ff060e0 | 1152 | 1038 / 114 / 0 |

两文件值均有限。归档 `code/` 不包含这些大文件，哈希也记录在XP04 `provenance/immutable_inputs.json:49–50`。未发现测试IDs，但作者真实生成checkpoint、生成命令和模型选择历史未知；不能由“无测试key”证明完整生成过程无测试信息依赖。

官方 `dataloader.py:23–36,73–86` 在全difficulty字典排序取比例，再恢复源顺序；XP04 `dataloader.py:37–46,119–130,174–185` 先限制当前训练IDs，再选最容易比例，验证IDs不会进入训练课程。

**README难度计算命令实际为batch-size=16，不是1**（`README.md:41,69`）。官方 `caldiff/cal_diff.py:358–360` 却把整个batch的rank差和只写到 `data[-1][0]`，speaker项也只取首个dialogue，因此照README16重算会只覆盖每batch首ID，不能生成所有对话难度。改成1会改变公开命令，应单独标记协议；不能擅自替换并称完全原始。MELD README `:69–72` 还指向train_cm.py并传未定义的ckpt_path；实际难度脚本入口是caldiff/cal_diff.py。这些问题与作者现成难度文件的真实生成史应分开陈述。

官方难度脚本 `:669` 加载args.ckpt_path，`:704–717` 仅对valid=0的train_loader计算；其test_loader虽构建但未用于难度调用。若输入checkpoint通过test选epoch，仍可能存在间接测试依赖。README Quick Start `:86–87`明确允许跳过难度生成，使用作者现成文件；本次来源选择应在独立provenance记录。

课程比例公式在官方 `train_cm.py:737–742` 与XP04 `:1265–1269` 一致，但基数不同。XP04实际IEMOCAP前7轮比例 `[0.15,0.30,0.45,0.60,0.75,0.90,1]`，训练对话 `[16,32,48,64,81,97,108]`；MELD前3轮 `[0.4,0.8,1]`，对话 `[415,830,1038]`。官方按全部train基数推导的对应数量为IEMOCAP `[18,36,54,72,90,108,120]`、MELD `[460,921,1152]`；训练完成后已逐轮核验真实Batch的ID覆盖与上述数量一致，见 `results/metrics_summary.json`。

## 8. 测试集使用、选模与Checkpoint保存

官方 `train_cm.py:757–773` 每epoch评test并以test W-F1选择最佳；指标先在 `:376–378` 四舍五入两位，更新条件为严格提高，因此相同两位分数保留首次epoch。官方 `train.py:493–512`同样，并存在test-loss更新best_label/pred的分支。该结果须标记 `official_test_selection`，不能称无测试泄漏独立评估。

XP04 `train_cm.py:1275,1305–1344` validation模式每epoch只train/validation，test字段为null；以validation W-F1选best。`:1412–1439` 训练结束后加载best并评test。四份真实final_test.json与独立评估metrics.json的选epoch和四舍五入后测试指标一致。

官方train.py与train_cm.py全文无torch.save；train.py `:535–550`只写max(test_fscore)及报告到日期record pk，train_cm.py在786行结束。XP04 `train_cm.py:152–173,1394–1402` 保存模型、优化器、RNG等完整best/latest。为满足本次保存最佳/最终checkpoint，需要独立记录instrumentation补丁，保持官方测试选择语义。正常训练没有预训练权重初始化：官方只在testing分支读取历史模型，四份XP04 manifest均resume=null/testing=false；课程来源checkpoint与正式模型初始化应分别记录。

## 9. 真实结果核验、度量尺度与可比较范围

以下为XP04同Seed现有结果，不是本次README实训结果。Accuracy、W-F1、Macro F1、ECE均为百分数；ECE使用15个等宽置信度bins；CEL为五种移除的正true-class log-probability差总和，不是百分数，也不是CC训练batch均值。

| 数据集 | 模型 | Seed | W-F1 % | Accuracy % | Macro F1 % | CEL总和 | ECE % | 最佳Epoch（1起始） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IEMOCAP | baseline | 1475 | 66.93611646 | 66.85150955 | 66.40313089 | 1520.02346293 | 16.45600689 | 70 |
| IEMOCAP | cmerc | 1475 | 68.28015294 | 68.08379544 | 67.54733567 | 683.02970313 | 3.97746635 | 43 |
| MELD | baseline | 67137 | 65.43860835 | 66.51340996 | 48.91235681 | 503.96788976 | 26.31207958 | 5 |
| MELD | cmerc | 67137 | 65.74612650 | 66.70498084 | 49.26783905 | 644.53838345 | 19.06660751 | 8 |

每行实际文件均已读取并对最终测试结果交叉核验：

- IEMOCAP baseline：`experiment_records/04_xp_mainline_validation/results/baseline/IEMOCAP/validation/seed1475/metrics.json`；训练最终测试 `experiment_records/04_xp_mainline_validation/runs/baseline/IEMOCAP_validation_seed1475/final_test.json`；参数/划分见同目录 `run_manifest.json`；loss及曲线见 `metrics.jsonl` 第70条epoch记录。
- IEMOCAP cmerc：`experiment_records/04_xp_mainline_validation/results/cmerc/IEMOCAP/validation/seed1475/metrics.json`；训练最终测试 `experiment_records/04_xp_mainline_validation/runs/cmerc/IEMOCAP_validation_seed1475/final_test.json`；参数/划分见同目录 `run_manifest.json`；loss及曲线见 `metrics.jsonl` 第43条epoch记录。
- MELD baseline：`experiment_records/04_xp_mainline_validation/results/baseline/MELD/validation/seed67137/metrics.json`；训练最终测试 `experiment_records/04_xp_mainline_validation/runs/baseline/MELD_validation_seed67137/final_test.json`；参数/划分见同目录 `run_manifest.json`；loss及曲线见 `metrics.jsonl` 第5条epoch记录。
- MELD cmerc：`experiment_records/04_xp_mainline_validation/results/cmerc/MELD/validation/seed67137/metrics.json`；训练最终测试 `experiment_records/04_xp_mainline_validation/runs/cmerc/MELD_validation_seed67137/final_test.json`；参数/划分见同目录 `run_manifest.json`；loss及曲线见 `metrics.jsonl` 第8条epoch记录。

| 数据集/模型 | 所选Epoch | classification loss | CC | MSCL | CSCL | total loss |
|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP / baseline | 70 | 0.2641454509 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.2641454509 |
| IEMOCAP / cmerc | 43 | 0.5217724442 | 280.8637084961 | 3.8165691239 | 2.3783957958 | 2.2358391966 |
| MELD / baseline | 5 | 3.4549507288 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 3.4549507288 |
| MELD / cmerc | 8 | 3.3452713013 | 60.8713400914 | 4.0387307424 | 3.2040927410 | 4.5534375337 |

上述训练loss为epoch内batch平均、分项尚未乘系数；total已含系数，训练日志的train_loss为分类loss。独立评估脚本 XP04 `scripts/evaluate_checkpoint.py:451–460,508–525` 匹配记录的ATen后端并导出分类指标、ECE、CEL。完整原始精度、文件SHA256、参数、split hash、课程比例及选epoch行保存于 `provenance/xp04_verified_metrics.json`。

**已证实事实**：训练IDs与规模、batch组织、RNG、GRU后端和模型选择存在上述差异；作者难度pickle与官方commit一致。**有证据但未隔离的推断**：这些变化可能影响优化和观察到的最佳分数。**待实验假设**：哪一项解释F1差距，须本次正式结果及后续隔离变量实验检验。本次两个指定单Seed的结果不能代表XP04五Seed均值；官方test选模与XP04validation选模也不能直接解释成同一协议的真实性能差异。

## 实际Batch运行证据补充

`provenance/batch_runtime_comparison.json` 从四份XP04正式metrics.jsonl逐轮读取并验证字段存在：对应IEMOCAP seed1475和MELD seed67137的Baseline/CMERC，`sampler_retries` 全部为0。因此这些配对不能归因为容量拒绝重采样；batch随机排列及RNG策略仍与官方不同。已完成官方运行的真实batches.jsonl显示：IEMOCAP Baseline全量80轮、MELD Baseline全量40轮、MELD CMERC全量38轮均仅有一种完整训练对话顺序。IEMOCAP CMERC全量74轮也已纳入终态核验，完整训练顺序指纹同样只有一种。该证据不改变前述源码对照，仅限制归因范围。
