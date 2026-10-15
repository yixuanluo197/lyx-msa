# XP 批次：默认非因果路径与公开上游协议审计

审计日期：2026-10-14（UTC）。本审计只读取代码、Git 对象、文档与官方 difficulty，并运行隔离 CPU 检查；没有运行 GPU、修改训练算法或启动正式 run。

## 审计结论

在指定的 `causal=False / official_test / GRU / hyper / concat_DHT / avl / BN / float32` 路径中，未发现因果代码合并意外改变初始化、课程样本顺序、辅助模型同步、Adam 范围或 CC/HSCL 数学定义的证据。四模型构造与上游逐 tensor 相等，CPU RNG 状态也完全相等。唯一影响默认图前向的实质补丁是 `HypergraphConv.py` 的 PyG 2.4 `size` 兼容修复：已在 CPU 重现原版报错、补丁版有限前反向成功。这不是严格的未修改源码运行，应保留补丁记录。

本结论只支持“未发现代码审计范围内的意外偏离”，不能据此宣称完整训练与论文等价。正式 XP 训练仍须由独立匹配环境中的官方 checkpoint 评测、完整比例和长 batch 显存探测以及真实训练日志验证。

## 来源与版本

- 个人仓库 `origin`：`https://github.com/yixuanluo197/lyx-msa.git`。
- 上游 `upstream`：`https://github.com/HITSZ-HLT/CMERC.git`。
- 当前 `HEAD` / `origin/main`：`91a1bffa763d57015896dee1014f0d11c5d09029`。
- 本地已保存的上游 `upstream/master` / `upstream/HEAD`：`815c7c060914f5b55db816c0ba3662b11d08552f`。
- 比较直接使用 `git show 815c7c0:<file>` 与 `git diff 815c7c0 HEAD -- <file>`；没有把本地上游 ref 称作刚查询的远端最新版本。
- 审计时九个相关核心文件与 `HEAD` 的工作区 diff 为空；新增批次辅助文件由其他执行通道负责，未纳入算法等价结论。
- 已阅读 `docs/UPSTREAM_README.md`、`REPRODUCTION_GUIDE.md`、`reports/01_CODE_AUDIT.md`、`reports/03_DATA_AUDIT.md`、`reports/05_TRAINING_REPORT.md`、`reports/07_REPRODUCTION_COMPARISON.md` 及相应实际训练/评测源码。

## 逐项对齐

| 项目 | 实际代码证据 | 判定及训练意义 |
|---|---|---|
| 非因果开关与 BN | `train_cm.py:606,699-703`；`model_cm.py` 三种构造器；`model_hyper.py:116-137` | `--causal` 默认为假，LN 强制切换只位于 causal 分支；GRU 非因果宽度/方向保持原值。未额外构造 causal 层，未消耗其参数初始化 RNG。 |
| 初始化及参数结构 | `train_cm.py:767-897`；四构造器 CPU 对比 | 两数据集主模型、模态辅助模型、intra/inter 辅助模型 state dict 的键序列和每个 tensor 完全一致；四次构造完成后的 CPU RNG 状态完全一致。 |
| 随机数行为 | `train_cm.py:167-183,387` | `seed_everything`、`_init_fn`、`get_train_valid_sampler`、`create_class_weight_SCL` 与上游 AST 完全一致。每次 train、空 valid、test 调用均重新播种，而不是只在初始化播种；这是上游行为，未静默修正。 |
| DataLoader 与 worker | `train_cm.py:216-240,279-302` | IEMOCAP 顺序遍历；MELD 使用 `SubsetRandomSampler`。正式计划 worker 从上游固定 2 改成 0；真实数据 `__getitem__` 为确定性取值，无增强。CPU 确定性数据实验中，worker 0/2 的顺序与完整 train/空 valid/test 后主 CPU RNG 一致。GPU及目标 torch 版本仍需正式验证。 |
| 课程前缀与排序 | `dataloader.py:118-130,177-189`；`train_cm.py:1001-1006` | `official_test` 不提供显式 split/keys，继续按 difficulty dict 排序截取，再恢复源数据 keys 顺序；未走新 validation-only 选择函数。比例公式保留上游的两位 round。 |
| 辅助模型同步 | `train_cm.py:375-387` | 每次调用开始均从主模型完整复制 state dict，包含 BN buffers，三个辅助模型设 eval；训练中的辅助模型保持 epoch 开始的快照，主模型为 train。验证/测试调用重新复制当时主模型。不是逐 batch 同步。 |
| 五路信息移除 | `train_cm.py:428-446`；`model_cm.py:1240,1595`；`model_hyper.py:365-394` | 三模态使用 padded batch 时间维均值重复，保留 padding 参与均值；intra 移除同说话者边，inter 移除不同说话者边。没有改成置零、仅有效 token 均值或 causal 前缀均值。 |
| CC | `train_cm.py:457-478` | 累加五路移除后真实类 log probability 相对完整模型的正增量；无 batch size / utterance 数归一化。移除模型输出在 no-grad 下；完整模型可获得 CC 梯度。保留原公式和差值尺度。 |
| HSCL | `train_cm.py:479-542`；`clloss.py:265-351` | 伪标签纠正/破坏/置信度条件、类别反频率权重、温度保持原实现；device 补丁仅把 `.cuda()` 改成 `.to(features.device)`。`ft_scl=hidden` 是别名，未 detach；进入 no-grad 代码块不移除已有 hidden 的图。 |
| 总损失 | `train_cm.py:114-142,544-563` | `combine_training_losses` 保持 `L_cls + rank_coff*L_CC + mscl_coff*L_MSCL + cscl_coff*L_CSCL` 的原加法次序。新分项 `.detach().item()` 用于记录，未替换反传 tensor；新增同步可能影响耗时，不能推断提升/降低 W-F1。 |
| 分类损失 | `train_cm.py:929-950`；`model_cm.py:15-50` | IEMOCAP 默认 NLL；**MELD 强制使用上游非标准 FocalLoss**。其 log-softmax、全类别求和/平均定义保留；新 one-hot 使用输入 device/dtype，正式 float32 与上游默认 float32 相同。不可将 MELD 改成标准 focal/NLL 作为追分修复。 |
| 优化器 | `train_cm.py:952` | Adam 仅主模型参数、同 lr/l2；辅助网络和无参数 CKSCL 不进入优化器。课程 `scheduler_steps` 不等于学习率调度器。 |
| 测试选模 | `train_cm.py:589-591,1031-1065` | `official_test` 每 epoch 用四舍五入到 2 位的测试 W-F1，以严格 `>` 更新 best；平局保留最早 epoch，与原分数追踪逻辑一致。公开代码的测试选模偏差仍存在。新 best.pt 以该指标保存，未按独立评测的更多小数追选。 |
| 独立分类指标 | `scripts/evaluate_checkpoint.py:101-158` | W-F1、Macro-F1、Accuracy 根据完整逐 utterance 预测重算，保留更多小数；CPU 例子与 sklearn 完全一致。训练选模值两位、独立报告更多位是精度差异，不是新选模协议。 |
| 独立 CEL | `scripts/evaluate_checkpoint.py:276-322`；`evaluation_metrics.py:86-108` | 完整和五路移除均 eval/no-grad、上下文辅助模型严格同步；模态移除使用同一主模型，eval 下等价于完全同步的 model_mm。CEL 为五路真实类 log-probability 正差总和；累加使用 float64，微小尾数差可来自累加精度。 |
| PyG 兼容修复 | `HypergraphConv.py:151` | 第二次传播 target-to-source 仍按 incidence 行的节点/超边维度传 `size=(num_nodes,num_edges)`。上游交换后的 `size` 在 3 节点/2 超边例子触发实际维度错误；补丁返回 3 节点并成功反传。 |
| 恢复与持久化 | `train_cm.py:24-103,1099-1107` | 保存 Python/NumPy/Torch/CUDA RNG、主模型、Adam、epoch；辅助模型在下一调用重新同步，无需保存独立优化器。非因果协议未严格核验 seed/损失/batch 等全部参数，因此调度器必须另核验，不能把 loader API 校验当作完整协议校验。 |

`model.py`、`train.py`、`high_fre_conv.py`、`model_GCN.py` 对上游的 Git diff 为空。独立非因果评测主模型沿用 `model.py`，训练主模型沿用 `model_cm.py`；后者多返回 hidden，默认模型参数结构未变。

## 可复核 CPU 实证

检查环境：`/home/yangbin/miniconda3/envs/multimodalerc310/bin/python`，torch `2.2.2+cu121`、PyG `2.4.0`；命令显式设置 `CUDA_VISIBLE_DEVICES=''` 和 `PYTHONDONTWRITEBYTECODE=1`。该环境仅用于 CPU 审计，不替代正式 XP 的 torch 1.13.1 环境。

1. 将 `git show 815c7c0:model_hyper.py` 和 `model_cm.py` 加载到独立内存模块，仅删除已不被 PyG 导出的无用 `topk` 导入并改写模块 import 名称；未更改构造器逻辑。两套构造器以同一 CPU seed、同一官方架构配置，按 `Model → Model → Model_Intra → Model_Inter` 次序构造。逐 key/tensor 比较及最终 CPU RNG 比较均通过：

   ```text
   IEMOCAP four constructor state_dicts exact; tensors=340; CPU RNG exact
   MELD four constructor state_dicts exact; tensors=332; CPU RNG exact
   ```

2. 四个随机数/采样/类别权重函数的 AST 无位置比较完全相同。固定的确定性 120-item Dataset，分别用 sequential / subset-random sampler、batch 16、worker 0 / 2，在 train、空 valid、test 全部遍历后比较顺序和主 CPU RNG，均一致。此实验不声称两种 worker 设置在任意随机增强 Dataset 上等价。
3. 真实 pickle 的训练对话 key 集与 difficulty key 集逐一匹配。复核 SHA-256 和全部课程比例：

   | Dataset | Difficulty SHA-256 | 对话数 / 前缀计数 |
   |---|---|---|
   | IEMOCAP | `db4c7b96f169948a3615a62b29d2a793bd64cccd8a0387b1de9d7ceab348a150` | 120 / `18,36,54,72,90,108,120` |
   | MELD | `db005ceb0e4847caffc8571eaa94ab7c5b4a310abd615b12c995c9a22ff060e0` | 1152 / `460,921,1152` |

4. 原版 HypergraphConv（3 节点、2 超边、宽度 4）在当前 PyG 2.4.0 实际失败：`ValueError: Encountered tensor with size 3 in dimension 0, but expected size 2.`。同初始化权重的补丁版输出 `(3,4)`，输出及输入梯度全部有限。
5. 独立分类指标在固定 6 样本、3 类预测例子上与 sklearn 的 weighted/macro F1 和 Accuracy 相等；示例均为 `66.66666666666666%`。这验证指标算法，不等同于官方 checkpoint 的实测。

## 已有报告的精确纠正与实质限制

- `reports/01_CODE_AUDIT.md` 把 FocalLoss 的问题描述为“官方配置未使用时不改”。实际 `train_cm.py` 的 MELD 分支及上游均无条件构造 FocalLoss，**本批 MELD 确实使用它**。这是文档审计不足；并未发现本地训练换损失的证据。
- 辅助同步可更精确表述为“每次 train/eval 调用开头同步；训练内固定 epoch 开头模型”。测试阶段辅助模型不是仍使用训练开头旧权重。
- `official_test` 的 MELD 训练集合包含 1038 train + 114 valid = 1152 对话；IEMOCAP 120 对话。使用新 validation split、固定 split seed 或重新生成每 seed difficulty 会改变研究问题，不属于当前批次。
- 非因果 mask 均值包含 padding，BN 混合 batch/时间统计；改变 batch size 或 padding 组成可能改变输出及 CEL。应严格保留 batch 16 和对话顺序，不能以 batch 1 的 CEL 直接作正式对照。
- 单 seed V100 落差、未知原作者五 seed、论文 hold-out 描述与公开代码测试选模差异均有既有来源支持，但目前没有足够证据把任何一个确定为训练差距的唯一原因。
- CPU 初始化相等不证明 GPU scatter 数值确定性，也不证明 XP/V100/4090 可逐位复现；软件/硬件归因须依靠正式 checkpoint 和训练锚点对照。
- 未测试正式匹配环境、CUDA/PyG 全模型前反向、两个官方 checkpoint 实测、长对话/满课程/测试峰值显存及 XP 正式训练。均交由主执行通道完成，不能在本报告提前判定通过。

## 核心文件 SHA-256

```text
4436841d8725f67992aceb2626279ff5e2875fdc64c3d2595e4408926b105064  train_cm.py
94acf11e2e9edd1bcd98049eb12d85b3604f8d71d46f64f0c3b8b9837fa12625  model_cm.py
fb9ef2ee268afa697474c9199d2136a7519b5226adb4abeec4a9cf1aea678aba  model_hyper.py
9297a572e1763a74c7984eb75dfb15fa8143bf131a44cf50433c307ef7f3b5d2  HypergraphConv.py
012c5d08f51eaaa99bdbd10d61e3d108275bf3dd44e42767b2e186e5be72ec3f  clloss.py
337a7dc92d9e69e4f8854a8031197085ffca84cdc08b070b6ef5b05ebb6f3390  dataloader.py
3b18b42762a5012610e4c90d95c991223342db12a7c45bd2a60cd025e9a24459  scripts/run_training.sh
980c2fbc40c77f6d246e5f471820712a77c0ef62bb6b0c29538e2aaa15589e00  scripts/evaluate_checkpoint.py
ed831a1f38d5706d6d47d7310d03e8b35f6eb4573623c9650934c638d65835c8  evaluation_metrics.py
```
