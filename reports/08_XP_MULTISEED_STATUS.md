# XP 多 seed 批次 20261014T1627Z

服务器：本地 user23；设备：TITAN Xp 候选 GPU 0、1、3。

MELD 每卡最多 2 个本批次任务；前三个正式训练使用不同 GPU。
仅当预检实测双任务总吞吐改善至少 10% 且显存有余量时启用第二个 MELD；不与 IEMOCAP 混跑。

公开 official_test 协议每 epoch 用测试集 W-F1 选 best checkpoint，存在测试集选模偏差。
首个 seed 来自作者 README，其余四个为预先固定的复现种子；作者完整五 seed 列表未知。
固定作者难度文件，统计仅反映该 curriculum 条件下的训练随机性。
单 seed baseline 仅作训练链诊断，不支持五 seed 配对提升或显著性结论。

| 数据集 | 完成数 | XP 五 seed Mean ± 样本 Std（ddof=1） | 最小 / 最大 | 与论文差 | ±0.5 点 |
| --- | ---: | --- | --- | --- | --- |
| IEMOCAP | 5/5 | 69.6150 ± 0.8254 | 68.2061 / 70.3589 | -2.3650 | 未达到 |
| MELD | 5/5 | 65.5253 ± 0.4451 | 65.1065 / 66.2072 | -1.3247 | 未达到 |

| 数据集 | 论文五次均值 | 官方 checkpoint 历史评测 | V100 单 seed 从头训练 | XP 官方 seed 单次 |
| --- | ---: | ---: | ---: | ---: |
| IEMOCAP | 71.98 | 71.9838 | 70.3753 | 70.3589 |
| MELD | 66.85 | 66.8484 | 66.1767 | 66.2072 |

历史 checkpoint / V100 指标未纳入 XP 均值；论文使用 RTX 4090，本批为 XP，不能将差距直接归因于 GPU。
XP 环境、代码与输入哈希见 batch_manifest.json；独立官方 checkpoint 验证和峰值 reserved 测量见批次审计产物。
逐任务状态、失败原因、配置、选中 epoch、耗时与显存见 state.json、status.csv 和 results.csv。

| 任务 | 状态 | 退出码 | 错误 |
| --- | --- | --- | --- |
| IEMOCAP_cmerc_official_test_seed1475 | completed | 0 |  |
| MELD_cmerc_official_test_seed67137 | completed | 0 |  |
| IEMOCAP_baseline_official_test_seed1475 | completed | 0 |  |
| MELD_baseline_official_test_seed67137 | completed | 0 |  |
| IEMOCAP_cmerc_official_test_seed42 | completed | 0 |  |
| MELD_cmerc_official_test_seed42 | completed | 0 |  |
| IEMOCAP_cmerc_official_test_seed3407 | completed | 0 |  |
| MELD_cmerc_official_test_seed3407 | completed | 0 |  |
| IEMOCAP_cmerc_official_test_seed2025 | completed | 0 |  |
| MELD_cmerc_official_test_seed2025 | completed | 0 |  |
| IEMOCAP_cmerc_official_test_seed2026 | completed | 0 |  |
| MELD_cmerc_official_test_seed2026 | completed | 0 |  |

尚未满足两个数据集完整五 seed 均值均在论文 ±0.5 点范围内的条件；不追加 seed 或调参。

复用命令：

```bash
python scripts/xp_multiseed.py run --batch-root /data2/yb/paper/lyx-msa/CMERC/experiments/xp_multiseed/20261014T1627Z
python scripts/xp_multiseed.py summarize --batch-root /data2/yb/paper/lyx-msa/CMERC/experiments/xp_multiseed/20261014T1627Z
```

## 实验链诊断与限制

| 数据集 | XP baseline 官方 seed | V100 baseline 官方 seed | XP−V100 | XP同 seed CMERC−baseline |
| --- | ---: | ---: | ---: | ---: |
| IEMOCAP | 70.1808 | 69.6256 | +0.5552 | +0.1781 |
| MELD | 66.1365 | 66.0644 | +0.0721 | +0.0707 |

两个官方 checkpoint 的 XP W-F1/CEL 与历史评估一致，支持输入、推理和指标链对齐；它不能保证从头训练重现论文均值。
默认非因果路径的模型初始化、CPU RNG、辅助同步、课程顺序和损失定义已与上游审计；未发现因果扩展污染当前路径的证据。见代码快照 reports/XP_PROTOCOL_AUDIT.md。
全部 seed 使用固定作者难度文件。MELD 的公开训练集合为 train+valid 共1152对话，IEMOCAP为120对话；official_test 每轮按测试分数选模。论文的hold-out调参与完整五seed仍未公开，数值接近不能消除这些协议差异。
本批为torch1.13.1+cu117/PyG2.4.0匹配环境。为保留UUID绑定，修复了PyTorch1.13的NVML设备计数问题，仅选择既有CUDA runtime回退；原源码、补丁、哈希及加载失败/通过证据已保存。
默认缓存和max_split128的工程运行reserved峰值分别11736/11836MiB，未通过余量门槛。最终统一GC0.8，并以memory_fraction1.0启用torch1.13缓存回收，保持原显存容量、batch16、FP32和所有训练超参数；完整课程探测与同卡吞吐证据均保留。
GPU散射数值差异、作者实际训练配置与初始化/seed集合仍未被完全排除。单seed的XP/V100差异和baseline诊断不足以确定因果，不直接归咎于显卡或随机种子。
失败项进入清单，不替换seed、不挑最高seed，不因未达±0.5点自动追加实验。样本Std为ddof=1；n=5只能粗估当前固定curriculum条件下的波动，不作复现等价或显著提升声明。

验证结果见 verification.json，环境与资源见 preflight/，运行状态、配置和日志分别见 state.json、runs/ 和 logs/。

## 最终复核与差距结论

全部12项正式运行及独立评估于 2026-10-14 20:24:47 UTC 完成；无失败、无替换seed、无追加追分实验。正式流水线墙钟约 3.235 小时，不含此前环境安装和工程预检。

| 数据集 | seed | 独立评估 W-F1 | best epoch（从1计数） | CEL |
| --- | ---: | ---: | ---: | ---: |
| IEMOCAP | 1475 | 70.3589 | 79 | 1334.9748 |
| MELD | 67137 | 66.2072 | 5 | 589.0241 |
| IEMOCAP | 42 | 69.7569 | 64 | 1088.2656 |
| MELD | 42 | 65.7231 | 3 | 568.7218 |
| IEMOCAP | 3407 | 69.7551 | 59 | 1156.5727 |
| MELD | 3407 | 65.2367 | 8 | 705.9671 |
| IEMOCAP | 2025 | 69.9981 | 76 | 1378.4893 |
| MELD | 2025 | 65.3531 | 9 | 725.5200 |
| IEMOCAP | 2026 | 68.2061 | 74 | 1377.9109 |
| MELD | 2026 | 65.1065 | 9 | 642.3671 |

本批固定五seed的均值均未达到论文±0.5点，且两个数据集的五次单模型最高分也低于各自达标区间下界。因此，本批观察到的随机运行波动没有弥合论文差距；这不是对所有可能种子或作者未知五seed集合的证明。按预定预算结束本批，不据测试分数继续调参或增加种子。

有证据的定位与边界：

- 输入/推理/指标链：XP官方checkpoint W-F1为71.9837935825 / 66.8483922007，CEL为1143.4927208051 / 605.0502918512，均通过预定容差；它只验证推理链，不能证明从头训练一致。
- XP与历史V100的公开seed CMERC差值分别为−0.0164 / +0.0305点，表明这两个锚点没有出现足以解释论文差距的XP退化；单次比较不能排除GPU数值影响。baseline也只有单seed，不能支持统计显著性结论。
- 公开脚本、当前XP配置及可取得的旧V100配置已逐项核对；核心超参数一致。历史V100没有与run绑定的代码hash/未提交diff，不能把当前commit倒填为旧运行版本。详见 preflight/XP_V100_CONFIG_AUDIT.md。
- 默认非因果初始化、RNG消费、辅助模型同步和课程顺序审计见 code/reports/XP_PROTOCOL_AUDIT.md。真实损失块、CKSCL和reverse_features的CPU张量实证支持CC/MSCL/CSCL梯度可回传，与仅设备适配后的上游逐位一致；不支持“全局断梯度”归因。CKSCL存在继承的非有限损失归零保护，但未测正式batch触发频率，不能据此解释分差。详见 preflight/XP_LOSS_GRADIENT_AUDIT.md。
- 已知协议差异包括公开代码的测试集逐轮选模、固定作者curriculum，以及未知作者完整五seed与hold-out调参细节。上述差异需要披露，但本批没有分离它们的因果效应。CEL沿用公开实现的真实类log-probability正增量求和尺度，不将其改称论文公式的概率差。

独立CPU复核记录为 independent_verification.json：全部12项顺序epoch、完整课程规模、BN/非因果/official_test配置、best/latest及optimizer/RNG状态、模型张量有限性均通过；1623条IEMOCAP和2610条MELD评测样本ID/标签与官方checkpoint评估一致，逐样本重算W-F1、Accuracy、Macro-F1、混淆矩阵和五路CEL均匹配。冻结模型/输入文件、observer及环境补丁hash均匹配。

预检测试、GPU前反向、两数据集smoke和官方checkpoint证据保留在preflight/；原始训练/评估日志保留在logs/。本次新增seed-aware调度/汇总工具scripts/xp_multiseed.py、限定版本的环境修复脚本scripts/fix_torch_uuid_count.py及对应测试，没有改变训练算法或引入新依赖；原始日志与checkpoint由.gitignore排除。

只读复核命令（在CMERC目录执行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data2/yb/conda_envs/cmerc-xp-repro/bin/python \
  experiments/xp_multiseed/20261014T1627Z/independent_verify.py
```
