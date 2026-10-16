# Causal CMERC 审查后回归与最小修复

2026-10-16。外部审查基准 `91a1bffa763d57015896dee1014f0d11c5d09029`；本轮起始 HEAD `ea84294d61b2168cf5782c8a7cc7ea1598a02011`。HEAD 新增公开实验档案，但八个审查核心源文件与基准无差异，见 `logs/base_to_head_core.patch`（空）。没有回退、清除用户工作区、提交或 push。

本轮先保存 `BASELINE_SNAPSHOT.json`、`BASELINE_ENVIRONMENT.json`、原工作区 patch、完整测试日志和 JUnit；初始 **153 passed、0 failed、0 skipped**。测试全绿未覆盖下面已实际复现的工程缺陷，不构成它们不存在的证据。

## 本轮最终状态

- 核心时间因果性：CPU/GPU各144配置通过，采样未来梯度最大值0，六类负对照全检出。
- 工程修复：四项审查目标及独立审查发现的恢复/划分边界均已修复并回归。
- 全tests：CPU374通过、4项CUDA跳过；GPU378通过、0跳过。原始基线153通过。
- 实际训练函数：CPU/GPU各16配置、正常dropout；2epoch多batch连续/恢复、权重/Adam/日志/next loss在本次审计设置下逐位相同，四种RNG显式核查。
- 真实特征：IEMOCAP/MELD的训练/验证少量对话、生产维度、CPU/GPU均通过；未选择测试对话。
- 正式实验：本轮未启动。因果baseline可按计划进入训练；完整CL仍缺合规真实teacher和difficulty。

## 逐项确认与行为变化

| 发现 | 触发条件与最小反例 | 修复位置与行为 | 修复前证据 |
| --- | --- | --- | --- |
| 1. 默认 CUDA 设备：确认 | 实际 CUDA 可用时，旧 main 默认 `cuda`，调用 `set_device(torch.device('cuda'))` 抛 ValueError；以不存在的 checkpoint 路径复现，确保未运行数据集评测 | `scripts/evaluate_checkpoint.py:226`：公共 API 将无索引设备绑定当前逻辑 index，校验可见范围；causal CPU 可用。解析单测覆盖 cuda/cuda:0/其他索引；真实 CUDA main 已执行 | `logs/baseline_device_reproducer.log`、`evaluator_original_device_probe.log`；`evaluator_red.log`（25 个新增测试失败，包含尚未存在 API 的失败，与真实反例区分） |
| 2. difficulty 来源：确认 | 旧 reader 接受仅含 causal/dataset/difficulties 的 train-only 字典，不能证明教师未训练/选模于验证或测试 | `causal_provenance.py:61,132,225,250`、`dataloader.py:88`、`caldiff/cal_diff.py:455`：显式路径、内容 SHA、embedded/sidecar manifest、源教师实际文件 SHA、特征/有序 train-valid ID、种子、训练/选模协议、架构和 `prefix_mean_v1` 全部核对；独占写入，不覆盖；恢复固定 artifact SHA。缺 provenance、official_test、异划分/种子、被替换文件直接拒绝 | `logs/provenance_baseline_defect.log` 为旧源码接受反例；`provenance_before.log` 为新增 API 前的 collection 失败；`provenance_after.log` 为修复验证 |
| 3. run 混写：确认 | 同 dataset/protocol/seed，改变 CC/HSCL 和图深度，旧目录表达式仍相同；旧 main 可追加日志并覆盖 checkpoint | `causal_run.py:77`、`train_cm.py:45,79`：因果配置哈希目录，非 resume 独占创建；恢复校验 manifest、difficulty、完整配置、历史日志和 best。非因果保留旧路径，但同样拒绝覆盖 | `logs/baseline_run_collision.json`（仅执行原路径表达式，无结果修改）；`training_red.log` |
| 4. 评测架构：确认，原文已披露的限制 | 训练 LSTM/Transformer 或改变 residue/av_encoder/图层数后，旧 evaluator 固定 GRU，拒绝合法 checkpoint | `scripts/evaluate_checkpoint.py:251,302`：从可信 checkpoint 白名单重建，严格 state_dict；12 组 GRU/LSTM/Transformer × residue × av_encoder、非默认图深度 round-trip。非法类型、字段或架构明确拒绝，不使用 strict=False | `logs/baseline_architecture_reproducer.log` 为旧工厂的真实字段拒绝反例；`evaluator_red.log`、`evaluator_cpu_final.log`、最终全套日志 |
| 5. 验证集诊断种子：新增入口的确认缺陷，已修 | checkpoint split_seed=310/1475，但新 `--split validation` 原先默认0，会使用错误划分 | `scripts/evaluate_checkpoint.py:314,375,659`：缺省继承 checkpoint；冲突拒绝；先重算完整特征/ID provenance，再取少量子集。缺来源的诊断须明确原因并标记 unverified | `logs/evaluator_split_red.log`（5 failed）、`evaluator_split_cpu.log`/对应最终日志 |
| 6. 历史日志与空续训：确认，已修 | 删除/清空/截断 metrics 后恢复；或请求总 epochs 不大于已完成轮数，会丢历史或引用未创建 loader | `causal_run.py:65,107`、`train_cm.py:79`：日志必须非空、epoch 连续且 latest 对齐；只允许保持/延长预算且必须仍有下一轮；校验发生在加载权重前 | `logs/training_resume_review_red.log`（6 failed/1 passed，前版本反例） |
| 7. 继承 best lineage：新增分支的确认缺陷，已修 | 原样复制祖先 best 到新分支后，再从它恢复，manifest 仍绑定原目录 | `causal_run.py:140`：导入 best 保留模型/Adam/RNG，只重封装当前 manifest，记录源路径、SHA 与源 manifest ancestry；两次连续 best-resume 已回归 | `logs/training_best_lineage_red.log`（2 failed） |
| 8. 旧非因果自动化路径：新增回归，已修 | 所有 run 都改哈希目录，旧非因果队列/等待器找不到 checkpoint | `causal_run.py:87`：只有因果 run 加哈希，非因果保持旧命名并安全拒绝混写。旧自动化不是新因果 run 的发现器 | 同上 red；`tests/test_followup_training.py` 的 noncausal 命名回归 |
| 9. GPU eval RNN 反向：后端限制，已修适用路径 | cuDNN 的 eval GRU/LSTM 不允许 backward；GPU 因果梯度检查及新的 same_step reference 实际失败 | 因果诊断仅梯度前向局部 ATen，并比较默认后端前向；`train_cm.py:418` 的 same_step confidence 全部同一 ATen/eval，分类仍正常 train/cuDNN；legacy 使用 nullcontext 保留 flags | `logs/causality_gpu.log`、初次 `gpu/CAUSALITY_REGRESSION.json`、`training_gpu_same_step_red.log`；修复后 `gpu_fixed/` 和 CUDA 专项测试 |
| 10. GPU 精确恢复与确定性：不是时间泄漏 | 初始默认 GPU 恢复的逐位日志断言失败；严格 deterministic 又遇 cumsum 无实现 | 不改生产默认；审计显式 CUBLAS workspace 与 deterministic warn_only，保留所有警告。先落六类 abs/rel/exact 数值再判断；CPU exact，GPU 预设同一2e-5/1e-5且离散语义相同，不放宽阈值。本次最终 GPU 恢复也实际 exact | `training_integration_gpu.log`、`training_integration_gpu_deterministic.log`、最终 `training_integration_gpu_final.log` |

实际行号以本次最终源码为准；完整新增回归在 `tests/test_followup_{evaluator,provenance,training,causality}.py`。独立只读审查额外发现的五项 MEDIUM（种子、历史日志、空续训、best lineage、旧路径）均经补丁和测试后复核关闭。不能把新增 API 尚不存在造成的 red collection 与对旧源码执行得到的缺陷反例混为一谈。

## 核心时间因果性

没有确认新的核心前向未来泄漏。`model_cm.py`、`model_hyper.py`、`causal_graph.py`、`clloss.py` 的算法实现本轮未改；不是靠删除旧但未执行的 BN/双向声明来判定。实际执行的 LN 特征维、Transformer 双 mask、定向超边、入度归一化及五视图已用真实模块验证。

CPU 与 GPU 各 **144/144 配置通过**。CPU 最大绝对差 `9.536743e-7`；GPU 数值和所有采样 scalar 的未来梯度见 `CAUSALITY_REGRESSION.json`、`gpu_fixed/CAUSALITY_REGRESSION.json`。六类故障夹具全部被检出。相对差以接近零的 hidden 为分母时可较大，接受标准始终为预设 combined `atol=2e-5, rtol=1e-5`。

实际训练函数 CPU/GPU 各16组、真实训练/验证特征的生产维度有限 batch、恢复的权重/Adam/日志/下步损失分别报告在 `TRAINING_INTEGRATION.md`。全套最终通过/跳过数、命令、环境及文件哈希统一登记于 `RUN_MANIFEST.json` 和 `COMMANDS.md`，不以子任务的局部通过代替全套结果。

## 简化、兼容和剩余风险

重复的来源校验集中在 `causal_provenance.py`，run 身份与恢复校验集中在 `causal_run.py`，复用现有划分和课程 key 选择函数。没有新依赖、全局环境升级、无关网络重构或历史实验档案修改。default auxiliary policy、非因果模型算法与原损失公式保留。

`prefix_mean_v1` 可逆且只是平滑；Intra/Inter 只是 GNN 边消融；legacy confidence 混有 mode/权重滞后，same_step 是独立研究配置。源 checkpoint 的 epoch 与 best_epoch 明确记录；正式计划只使用 validation-selected best，不将任意 latest snapshot 称为最佳教师。

GPU warn_only 不保证其他硬件/长训练逐位复现；本次小检查相等不替代跨硬件保证。原 PyTorch 1.13.1、原始特征提取因果性、完整性能和收敛实验均未验证。可信 pickle/checkpoint/manifest 是本地研究一致性合同，不能抵御能改写全部证据的对手。

没有合规真实因果 baseline 权重，故真实 causal difficulty 和完整 CL 正式实验仍阻塞。随机诊断模型、合成 teacher 元数据夹具、原始非因果 difficulty 均未作为正式实验输入。只写正式实验计划，未启动80/40 epoch、完整预训练或多 seed 矩阵。
