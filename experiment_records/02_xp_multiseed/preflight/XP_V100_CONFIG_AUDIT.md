# XP 与历史 V100：official_test 配置及版本证据审计

审计时间：2026-10-14 17:40–17:44 UTC。XP 批次为 `20261014T1627Z`。本审计仅读取历史产物、正在运行任务的已原子保存 checkpoint、冻结代码和文档；没有运行 GPU、更新模型、改动代码/批次或中断训练。CMERC 尚在训练，本报告不报告其中途性能，也不据此给最终复现结论。

## 结论

两个数据集、两个变体的**公开 seed 对应配置中，全部已有训练参数一致**。历史 V100 checkpoint 的 `args` 有 49 项，当前 XP 有 50 项，原始差异只有 `output_dir` 路径和新增 `causal=False` 字段；未发现 batch、学习率、权重衰减、dropout、模型/图配置、课程、损失权重或选模参数变化。旧 V100 实际同样使用 `num_workers=0`，因此 XP 与历史 V100 的 worker 数没有差异。

这一证据排除了“这些已记录参数被改掉”作为解释，但不足以排除硬件数值差异、未记录运行时配置、未绑定到 run 的历史代码/数据差异或其他未保存行为。缺少历史代码指纹是可追踪性的限制，**不是发现了一个确定的差距原因**。

## 读取范围与执行方式

历史根目录：`runs/formal/`。逐一读取以下四个 run 的 `best.pt` 和 `latest.pt`：

- `baseline/IEMOCAP_official_test_seed1475`
- `baseline/MELD_official_test_seed67137`
- `cmerc/IEMOCAP_official_test_seed1475`
- `cmerc/MELD_official_test_seed67137`

XP 根目录：`experiments/xp_multiseed/20261014T1627Z/runs/`。读取两个完成 baseline 的 `best.pt/latest.pt`，以及对应两个 CMERC 锚点已保存 checkpoint；进一步核验两数据集 seed 42 的已保存 `latest.pt` 配置。所有 checkpoint 直接使用匹配环境 Python 和 `torch.load(..., map_location='cpu')`，未使用会初始化 CUDA 的批次 profiling Python。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro \
  bash scripts/with_cmerc_env.sh \
  /data2/yb/conda_envs/cmerc-xp-repro/bin/python - <<'PY'
import torch
checkpoint = torch.load('runs/formal/baseline/IEMOCAP_official_test_seed1475/latest.pt',
                        map_location='cpu')
print(checkpoint['args'])
PY
```

CPU 读取环境实测 torch `1.13.1+cu117`。正在运行的 checkpoint 可在后续 epoch 被替换；这里核验的是成功读到的完整保存快照的**配置**，不是要求其 epoch 此后保持不变。读取没有向运行目录写入文件。

## 全部训练参数对齐

下表值对同数据集、同变体的 V100/XP 公共 seed 配对均相同。非差异字段已逐 key 比较，不仅检查表中概括项。

| 范围 | 字段及相同取值 |
|---|---|
| 共同模型及输入路径 | `base_model=GRU, graph_model=True, graph_type=hyper, graph_construct=direct, multi_modal=True, mm_fusion_mthd=concat_DHT, modals=avl, norm=BN` |
| 共同结构 | `num_L=3, Deep_GCN_nlayers=4, use_gcn=False, use_residue=False, use_speaker=True, use_modal=False, use_topic=False, av_using_lstm=False` |
| 共同注意力/窗口配置 | `nodal_attention=True, active_listener=False, attention=general, alpha=0.2, multiheads=6, windowp=10, windowf=10` |
| 共同优化/数据装载 | `batch_size=16, lr=0.0001, l2=0.00003, rec_dropout=0.1, class_weight=True, num_workers=0` |
| 共同执行/选模 | `selection_protocol=official_test, testing=False, tensorboard=False, resume=None, no_cuda=False, cuda=True` |
| IEMOCAP 共同数据集配置 | `Dataset=IEMOCAP, seed=1475, dropout=0.5, epochs=80, num_K=4` |
| MELD 共同数据集配置 | `Dataset=MELD, seed=67137, dropout=0.4, epochs=40, num_K=3` |
| 两数据集 baseline | `calibrate=False, contrastlearning=False, courselearning=False, rank_coff=0, mscl_coff=0, cscl_coff=0, epoch_ratio=0, scheduler_steps=0` |
| IEMOCAP CMERC | `calibrate=True, contrastlearning=True, courselearning=True, rank_coff=0.005, mscl_coff=0.05, cscl_coff=0.05, epoch_ratio=0.15, scheduler_steps=1` |
| MELD CMERC | `calibrate=True, contrastlearning=True, courselearning=True, rank_coff=0.002, mscl_coff=0.15, cscl_coff=0.15, epoch_ratio=0.4, scheduler_steps=1` |

原始差异：

| 项目 | V100 | XP | 意义 |
|---|---|---|---|
| `args.output_dir` | `/data2/yb/paper/cmerc_reproduction_20261013/runs/formal/<variant>` | 当前 XP 独立批次 `runs/<variant>` | 输出隔离路径，无训练数学变化。 |
| `args.causal` | 未保存此字段 | `False` | XP 显式记录默认非因果。历史 checkpoint 产生于加入该 schema 前；字段缺失本身不说明历史使用了因果模型。 |
| checkpoint 顶层 `causal` | 无此项 | `False` | 与上述一致的新增记录字段。 |
| 每 epoch `causal/effective_norm` | 历史 JSONL 未记录 | 当前 JSONL 显式记录非因果/BN | 记录增强；实际 `norm=BN` 在两组 checkpoint args 中一致。 |

两数据集的 XP seed 42 checkpoint 与各自官方 seed 的 XP 配置相比较，**仅 `seed` 改成 42**；其余 49 个字段均一致。它们不是对应 V100 官方 seed 的配对训练，应作为预先指定的独立种子。

四个历史 run 的 best/latest 参数字典在各 run 内完全相同。两个 XP 完成 baseline 的 best/latest 参数字典也相同，且 latest 保存完成 80/40 epoch。当前 CMERC 的配置不依赖其尚未完成的选中 epoch 或指标。

## 优化器、结构、损失与 RNG

**Adam 已保存超参数一致。** 四组对应 run 的 `optimizer_state_dict.param_groups`（移除参数编号列表后的超参数）完全相同：`lr=0.0001, weight_decay=0.00003, betas=(0.9,0.999), eps=1e-8, amsgrad=False, maximize=False, capturable=False, differentiable=False, foreach=None, fused=False`。更新次数和动量 tensor 随训练阶段/轨迹改变，不属于超参数变化；本审计未声称优化器状态值相同。

**已保存模型结构一致。** 对应最新 checkpoint 的 state dict 键顺序、每个 tensor shape 和 dtype 完全相同：IEMOCAP 每主模型 85 项、MELD 83 项，包含参数和 BN buffers。该检查没有比较训练后的权重值相等，也不能排除参数结构相同但前向/损失代码不同的情形。

**损失事实与元数据边界。** 当前冻结 `code/train_cm.py` 中 IEMOCAP 由 `class_weight=True` 使用固定逆类别比例权重的 `nn.NLLLoss`；MELD 无条件使用上游非标准 `FocalLoss`，不是普通 NLL。CC/HSCL 组件及权重如上表，baseline 关闭三个组件。历史 args 保存 `Dataset`、`class_weight` 和这些权重，历史训练报告称保留官方损失/算法；但是 checkpoint 没有 `loss_class`、损失源码哈希或单独记录 focal 的 gamma/alpha。因此元数据可以排除组件开关/系数变化，**不能独立证明历史每一行损失公式与当前冻结源码完全相同**。当前非因果路径对公开上游的详细代码检查见 `XP_PROTOCOL_AUDIT.md`，不能把该检查反推成历史 run 的完整代码指纹。

**保存的 RNG 快照相等，但不能代表训练轨迹相等。** 四组对应 latest checkpoint 的 Python、NumPy、CPU Torch 和 CUDA RNG 保存快照经 CPU 内容比较均相等。当前代码每次 train/eval 调用开始执行 `seed_everything(args.seed)`；epoch 末 test 又重新播种，eval 路径关闭 dropout。因此终点 RNG 相等可能只是这些重播种的结果，不能证明训练期间每个 kernel 的随机数消耗、浮点输出、梯度或更新轨迹相同，也不能排除显卡相关因素。

## 历史版本证据与 XP 冻结证据

**历史 V100 可用证据：** checkpoint args、Adam/model/RNG/进度；`logs/training/*_official_test.log` 首行 Namespace；`metrics.jsonl` 中课程比例、训练对话数及分项损失；`reports/02_ENVIRONMENT.md`、`03_DATA_AUDIT.md`、`05_TRAINING_REPORT.md` 的环境/输入/协议说明。这些可以相互验证相同配置和已记录协议行为。

**历史 V100 当前本地产物缺口：** 四个旧 run 的目录只有 best/latest/JSONL；checkpoint 顶层没有 code commit、源码 hash、完整 diff、环境 freeze 或 run-linked 数据 hash。相关本地报告说明上游为 `815c7c0` 并列出所做工程修复，但没有把一份完整“上游 + 当时未提交修改”指纹绑定到每个 official_test run。旧输出路径 `/data2/yb/paper/cmerc_reproduction_20261013` 在本机不存在；这是 checkpoint 留存的历史服务器路径，不能据此认为其历史训练失败。未连接远端补查，本结论限于当前本地提供的旧产物。

当前发布 commit `91a1bffa763d57015896dee1014f0d11c5d09029` 的提交日期为 2026-10-14，旧训练报告日期为 2026-10-13。该提交包含已有复现工程和随后加入的因果变体；**不能仅因旧结果被复制到这个仓库，就宣称旧训练直接引用此 commit**。仓库保留上游历史，但没有独立中间提交明确冻结旧正式 run 的完整工程版本。其他项目或后续 validation 多 seed 的 snapshot/manifest 不等于这四个旧 run 的版本证据。

**XP 可用冻结证据：** `batch_manifest.json` 和 `snapshot.json` 明确记录 code root、commit、文件 SHA-256、四份输入和两份固定作者 difficulty；`source.diff` 为空，`code_diff.patch` 仅增加忽略 XP 产物的规则。本次实际核验冻结代码中 `train_cm.py/model_cm.py/model_hyper.py/dataloader.py/clloss.py/HypergraphConv.py/scripts/run_training.sh` 及两份 difficulty 的 SHA-256，均匹配 manifest。

XP 的运行 wrapper 与环境补丁也有独立 hash/说明：torch UUID/NVML 枚举回退修复；`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`；`set_per_process_memory_fraction(1.0)` 启用 GC 且保留全容量；`profile_run.py` 包装峰值 allocated 查询以保存 reserved/时间统计。这些属于明确的**运行时差异**，不是 checkpoint args 中的模型参数变化。源码可见其目标是设备识别/分配/统计，没有改变 batch、损失或优化步骤；历史 run 不保存同级 runtime manifest，不能证明其运行时变量全部相同，亦无证据将它们认定为分数差距原因。

历史环境文档与 XP 预检均报告 Python 3.10.13、torch 1.13.1+cu117、PyG 2.4.0、scatter 2.1.0+pt113cu117、sparse 0.6.15+pt113cu117、NumPy 1.26.4 等。版本号对齐比直接使用 torch 2.2 更可比，但旧软件二进制 hash、底层 kernel 实际选择、未记录环境变量没有 run 级指纹，不能称为逐字节相同环境。硬件确实不同：历史 V100、本批 TITAN Xp、论文 RTX 4090。

## 可以和不可以排除的因素

- **可以排除：** 对应公开 seed 的上述已保存 CLI 参数变化；两个 XP baseline 的中途改参数；Adam 已保存超参数变化；已保存主模型参数/buffer 结构变化；XP 官方 seed 与 seed42 之间除预定 seed 外的配置变化。
- **有可用支持但不能完全排除：** 相同数据/difficulty 的历史实际使用（既有数据审计有官方哈希，但旧 checkpoint 不绑定输入 hash）；相同损失及前向实现（报告称忠实，上游与当前代码已审计，但旧 run 无完整源码指纹）。
- **不能据此排除：** GPU 浮点/扩展运算的数值差异、训练期 RNG 消耗差异、未记录环境和运行时配置、旧 run 的未冻结源码细节。缺口需证据补齐；当前配置审计不指定它们中的任何一个为确定原因。
- **本报告不回答：** 当前 CMERC 最终性能、五 seed 均值与论文阈值是否达标。必须等计划内训练及独立评测完成；不能用正在运行的 checkpoint 指标取消种子、调参或扩大预算。
