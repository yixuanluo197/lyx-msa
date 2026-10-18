# XP mainline validation：V100 V2 训练来源审计

2026-10-16 UTC；只读审计 `experiment_records/03_v100_multiseed/code/`。未运行 GPU、修改档案或执行旧调度器；`AGENTS.source.md` 仅作历史文件哈希对象，不作指令。

## 来源完整性

`provenance/source_snapshot.json` 的 30 个仍位于 `code/` 的文件 SHA-256 均匹配；第 31 个原 `AGENTS.md` 已移至 `provenance/AGENTS.source.md`，字节哈希匹配。V2 冻结摘要 `4f8cd5e6c9fb4f54d7280ed3989942f4a01018f3d496452226c4757dba3b0e33`。这是 V100 V2 训练逻辑来源，不能直接作为本机可运行批次：`code/` 无 data/caldiff，历史 snapshots 位于 archive/provenance，而旧调度器期望 code/snapshots。

## 正确入口与默认陷阱

训练入口 `scripts/run_training.sh Dataset variant validation`。它固化 GRU、lr1e-4、batch16、hyper/direct、concat_DHT、avl、BN、num_L3；IEMOCAP dropout.5/80epoch/K4，MELD dropout.4/40epoch/K3。baseline 不开 CL/HSCL/CC；CMERC 三者全开，IEMOCAP rank/MSCL/CSCL=.005/.05/.05、课程.15；MELD=.002/.15/.15、课程.4；课程步长1。

V2 **必须显式提供**：

```text
--seed <预注册seed> --split-seed 1475 --rng-policy continuous
--batch-policy capacity_safe --edge-capacity 1000 --node-capacity 5200
--batch-max-retries 10000 --num-workers 0 --require-noncausal
--output-dir <新批次/runs/variant>
```

- wrapper 默认 RNG 为 legacy，batch_policy 最终沿用训练器 legacy；未指定 split-seed 时跟随训练 seed。仅把第三参数写成 validation **不等于 V2**。
- continuous 自动派生 `rnn_backend=aten`，不是一个单独 CLI 开关；legacy 为 cudnn。
- 直接运行 train_cm.py 的默认值还包括 LSTM、batch32、60epoch、relation/full、fusion concat、norm LN2、selection official_test、worker2；不能依赖默认参数复用原批次。`--seed` 解析后强制非空。
- wrapper 拒绝 `--causal`；训练器 `--require-noncausal` 拒绝混用因果。capacity_safe 还强制非因果/validation/continuous/hyper/avl，容量不得超过1000/5200。
- 附加参数置于末尾，可以覆盖 batch/epoch/dropout/output-dir 等；需要冻结最终 argv 和 checkpoint args，不能只保存 shell 模板。
- 环境 wrapper 默认 prefix 为旧 `/data2/yb/conda_envs/cmerc-repro`；本批需显式环境路径和 `CMERC_PYTHON`。

独立评估入口：`scripts/evaluate_checkpoint.py --dataset ... --checkpoint .../best.pt --output-dir <独立seed路径> --device cuda:0 --batch-size 16 --num-workers 0 --require-noncausal`。前三项必填，默认 device=cuda、batch16、worker0、ECE bins15。评估自动读取 checkpoint 的 rnn_backend；无该字段时回退 cudnn。应额外要求本批 metadata 为 continuous/aten/validation/capacity_safe；单独 `--require-noncausal` 不验证所有 V2 字段。非因果官方模型不支持 CPU 全前向。

旧 `multiseed_scheduler.py` 和 config 固定 biggpu 卡1/2/3/5并硬禁GPU4；`resume_preflight.py` 固定宿主卡1/2。不得直接复制为本机 GPU 归属/调度策略。新批次可复用训练/评估逻辑，重新绑定资源、快照及输出路径。

## validation 与测试隔离

`get_IEMOCAP_loaders` 使用独立 split_seed 的对话划分：108train/12valid/31test；MELD 明确1038train/114valid/280test。CL 只对排除 valid 后的 train keys 取难度前缀，未将 valid 对话用于本轮参数更新。

epoch 主循环 validation 分支只调用 train/valid；test 指标赋 NaN 并写 JSON null。选模使用 round到2位的 validation W-F1 严格提高，平局保留较早 epoch。训练结束重新载入 best.pt，再调用一次 final test，并独占创建 final_test.json；独立 checkpoint 评估是再次重载同一已选模型作审计，不反馈调参/选模。仅构造 test loader/记录 test IDs、哈希不等于读取测试指标选模。

保留限制：checkpoint 输入架构、损失和全部有效配置必须固定；不能把官方 checkpoint 的 cudnn 评估与本批 ATen 从头训练称作逐位相同协议。

## continuous RNG、ATen 与容量协议

- 初始化播种；continuous 训练函数不再每次重播种。`isolated_evaluation` 装饰器在非训练调用保存/恢复 Python/NumPy/CPU Torch/CUDA RNG。
- loader 采用私有 generator，seed=training_seed+零基epoch；worker 从 torch.initial_seed 派生 Python/NumPy seed。恢复时由 epoch 重建下一轮顺序，不依赖保存一个正在迭代的 sampler。
- `configure_recurrent_backend` 只在 GRU/LSTM forward 内关 cuDNN，其他层保持原设置；目的是避开 cuDNN GRU 进程内不可序列化 dropout cache。ATen 改变数值/随机执行轨迹，是显式协议差异，不能隐藏成“与旧单 seed 相同随机过程”。
- CapacitySafeBatchSampler 保留每轮全部 eligible 对话一次、batch16、末批可不足；若任一批的 `N+3B>1000` 或 `3N>5200`，拒绝整个候选排列，私有 RNG 重抽。无标签/指标筛选，不缩 batch、不删样本、不扩模型；耗尽10000次后报错且不先吐部分 epoch。条件重排改变 batch 联合分布，必须与 legacy/V1分组。
- capacity_safe 只包训练 loader；valid/test 保持固定顺序，新批次启动前仍需 `audit_batch_capacity.py` 或等价 CPU 探测验证它们安全。

## difficulty：允许的 fixed_author 协议与不能推出的结论

直接把两个 V2 CMERC run_manifest 的 split IDs 与当前同哈希官方 difficulty 比较：

| Dataset | difficulty 数量 | 与 train / 当前 valid / test 的 key 交集 |
|---|---:|---|
| IEMOCAP | 120 | 108 / 12 / 0 |
| MELD | 1152 | 1038 / 114 / 0 |

SHA-256 分别为 `db4c7b96f169948a3615a62b29d2a793bd64cccd8a0387b1de9d7ceab348a150` 和 `db005ceb0e4847caffc8571eaa94ab7c5b4a310abd615b12c995c9a22ff060e0`，匹配历史 run provenance。

**包含当前 valid 键不违反已声明的 fixed_author 协议。** 本轮 CL 实际先排除 valid，再排序/取前缀；这些 valid 分数不作为训练样本参与本批反传。作者 difficulty 生成用的 baseline 训练集合及完整历史未公开，因此不能证明该外部难度估计完全未见本轮 holdout；也不能仅凭 key 存在就断定发生了哪一种生成泄漏。应固定作者文件、披露生成未知，不能声称整个预处理流程 validation-free；本批不为此改协议或重新追分。

## Resume 完整性

- latest/best 原子替换，保存主模型、Adam、epoch、best metric/epoch、Python/NumPy/CPU/CUDA RNG、args/provenance、sampling metadata。辅助模型在下次 train/eval 调用从主模型含 BN buffers 同步；没有独立待恢复优化器。
- 新 run 独占 mkdir；resume 只允许同一 run 目录，非阻塞 flock，要求 metrics 的 epoch 连续且与 checkpoint 严格一致，已有 final_test.json 时拒绝恢复完成 run。
- 加载前核验 causal/架构/selection，以及 seed/split_seed/RNG/backend、batch/lr/l2/dropout/损失组件/课程/worker/其他模型开关/容量/provenance；恢复模型、Adam及RNG，从epoch+1继续。
- provenance 绑定完整 split IDs/hash、顶层 Python 源码 hash、数据及所用 difficulty；脚本/环境的完整性依赖批次额外 source snapshot/环境 manifest，不能仅靠 run_provenance。迁到新目录后不能把新 provenance 冒充旧 checkpoint 的无缝恢复。
- 支持 epoch 边界恢复；不保存中途 batch 进度。异常发生在 metrics append 和 checkpoint 原子替换之间会被连续性校验拒绝，不能盲目重提。另须核验保留 best.pt；只剩 latest 而旧best丢失不能保证最终选中模型可测试。
- 历史 V2 真实两epoch恢复预检 verification 为 passed、CPU/CUDA RNG相等、最大参数差8.778180927038193e-06，容差rtol1e-3/atol1e-4。属于历史 V100 证据，不替代本机 XP ATen 的真实恢复验证。

## 可复用测试及覆盖边界

测试位于 archive 的 `tools/current_tests/`，不属于冻结 code，导入需要 `PYTHONPATH=<新code>:<tests路径>`，cwd 应是新code以避免误导入仓库根旧train_cm。归档 `verification.json` 记录：2026-10-15匹配环境CPU suite **51 passed / 3 skipped / 1 deselected**；未在本审计重新运行 pytest。两个skip缺外部数据，一个skip需显式GPU；shell路径用例因测试目录搬迁被排除。

- `test_multiseed_training.py`：四个 capacity sampler 单测；真实IEMOCAP五seed×80epoch/MELD五seed×40epoch全课程与全量容量扫描；fixed split、legacy split、eval RNG隔离、真实训练函数toy dropout推进、worker回放、独占run锁、metrics/checkpoint不匹配拒绝、下一次dropout更新精确恢复、协议变更加载前拒绝、仅RNN切ATen、GPU跨进程RNN恢复（显式GPU开关）。
- `test_training_protocol.py`：对话拆分、validation训练排除、课程在排除valid后选样、选模来源、完整checkpoint/RNG、JSON null、总损失系数与关闭组件。
- `test_multiseed_summary.py`：V1/V2拒绝混算、部分run/causal/非有限/每epoch test指标拒绝、split冲突、样本Std、seed不筛选、独立评估继承ATen、metadata/epoch对齐。
- `test_multiseed_scheduler.py`：冻结校验、输出防覆盖、锁/进程归属、固定矩阵及资源检查；路径与GPU限制针对历史布局，需要在新批次测试其本机适配，不能直接原样调用。
- `scripts/audit_batch_capacity.py`：真实数据1200排列及固定valid/test容量，只读CPU计算，输出必须指向新预检路径。
- `scripts/resume_preflight.py`：真实GPU双进程两epochCMERC恢复对照，必须先适配本机卡与新输出；本审计未执行。
- 分项梯度已有证据：之前 `reports/XP_LOSS_GRADIENT_AUDIT.md` 对真实损失AST块和实际CKSCL/reverse_features的CPU审计确认CC/MSCL/CSCL有限非零hidden梯度；本次核验V2完整移除/伪标签/CC/HSCL块与该XP来源AST完全一致，clloss/model_hyper hash亦一致。**现有 `LossAccountingTests` 只测权重算术，没有分别回传三个真实非因果分量；`test_causal_losses.py` 测因果模型，不能代替本批默认非因果真实GPU梯度预检。**

以上区分本次源码/哈希/真实difficulty key实证、历史测试记录和本机XP仍需执行的预检，不以过去通过代替当前设备验证。

## 本批预检梯度 observer

已另行交付 `experiments/xp_mainline_validation/20261016T050823Z/tools/gradient_observer.py`，由预检 wrapper 的 `CMERC_GRADIENT_PROBE` 分支调用 `install(output_path)`；普通 import 无副作用。默认仅观察前两个真实 CMERC 训练 backward，完成后自动还原 Tensor.backward。分别对实际局部 classification_loss/rank_loss/Mscl_tav/Cscl_c 与 hidden 调用 retain_graph 的 autograd.grad，并保存连接状态、有限性、范数和 shape；条件性零单独标记。Python/NumPy/CPU/CUDA RNG 保存恢复，未增加 forward 或 optimizer step。

本次匹配环境 CPU 内存测试通过：四个分量有限非零；采样两个 batch 后自动卸载；与不安装 observer 的三次 toy 更新比较，loss、模型参数、Adam 状态和 Python/NumPy/Torch RNG 完全相等；即使加入会消耗 RNG 的 backward hook，结果仍一致。另验证常数零/unused hidden 被明确记录且不阻止原 backward。输出由内存 stream 捕获，没有向训练目录写入测试产物。真实 XP 全模型 GPU 观测由主执行通道完成，本报告不提前判定通过。
