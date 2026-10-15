# CMERC 官方代码审计

审计对象：`HITSZ-HLT/CMERC@815c7c060914f5b55db816c0ba3662b11d08552f`。论文：ACM MM 2024，PDF 第 1–10 页（论文页码 9621–9630）。

## 结论

官方仓库包含论文核心实现，但原样代码不能形成可持续复现闭环。最严重的问题是训练期间使用测试集选模、训练脚本不保存 checkpoint、MELD 难度命令写错、难度脚本与 README 的 `batch_size=16` 不兼容，以及若干无条件 CUDA 调用。后续工程会保留可追踪的“忠实官方协议”，另提供严格使用验证集选模的“规范协议”；两者结果不得混报。

## 入口与模型关系

| 功能 | 入口 | 证据 |
|---|---|---|
| M³Net baseline | `train.py` + `model.py` | `train.py:9-10` |
| CMERC / 可关闭校准的 baseline | `train_cm.py` + `model_cm.py` | `train_cm.py:8-18` |
| 对话难度计算 | `caldiff/cal_diff.py` | `caldiff/cal_diff.py:177-360,669,704-717` |
| 官方 checkpoint 评测 | `train.py --testing` | `train.py:482-491` |

论文第 3–4 页明确 CMERC 是不改变 backbone 结构的校准框架，第 4 页采用 M³Net 作为 baseline。`model.py::Model` 与 `model_cm.py::Model` 主体一致；后者额外返回分类前隐藏表示（`model.py:865-872`，`model_cm.py:865-873`）。完整 CMERC 另建 `model_mm`、`Model_Intra`、`Model_Inter` 生成反事实输出（`train_cm.py:535-658`），这些辅助模型不参与优化。

## CL、HSCL、CC 的实现映射

### Curriculum Learning（CL）

- 论文公式 (1)–(4) 用移除模态/上下文后的置信度增益定义对话难度，并采用 baby-step 课程学习。
- 代码每轮使用 `ratio = min(1, epoch_ratio * int(e / scheduler_steps + 1))`（`train_cm.py:737-742`）。Dataset 按 difficulty 排序、截取前缀，再恢复原顺序（`dataloader.py:23-36,73-86`）。
- `scheduler_steps` 不是学习率调度器。IEMOCAP 的比例为 15%、30%……直至 100%；MELD 为 40%、80%、100%。

### Hybrid Supervised Contrastive Learning（HSCL）

- 模态分支调用 `CKSCL_tav`，上下文分支调用 `CKSCL_context`（`train_cm.py:344-346`；`clloss.py:271-355`）。温度在 `clloss.py` 中固定为 `0.07`。
- 总损失实现为 `L_cls + rank_coff*L_CC + mscl_coff*L_MSCL + cscl_coff*L_CSCL`（`train_cm.py:348-358`）。README 的 IEMOCAP/MELD 权重与参数名一致。
- 论文 Algorithm 1 以“移除后真实类置信度上升”定义不确定标签；代码 `train_cm.py:283-341` 还引入预测是否被纠正/破坏的分支，并将未覆盖情形保留为 0。这不是公式的逐字实现。
- `clloss.py:298-313` 额外使用伪标签类别权重，论文公式 (5)–(7) 未描述这一项。

### Confidence Constraint（CC）与 CEL

- A/V/T 移除并非置零，而是用时间维均值替换（`train_cm.py:235-246`）。X/O 上下文移除通过删除同说话者边或不同说话者边实现（`model_cm.py:1204-1209,1552-1557`；`model_hyper.py:310-338`）。
- `train_cm.py:261-282` 在移除信息后真实类 log-probability 不低于完整模型时累加单边差值；`rank_coff` 对应论文 `γ_s`。
- 论文称 `o` 为预测概率，代码直接比较 `log_softmax`。上升关系单调等价，但差值尺度及 CEL 数值不等价。
- 论文公式 (8) 与 Algorithm 1 对 CC 是否除以 batch size 存在内部矛盾；代码不除以 batch size。

## 输入、输出和主要维度

| 张量 | IEMOCAP | MELD |
|---|---:|---:|
| 4 个 RoBERTa 层 | `[T,B,1024]` | `[T,B,1024]` |
| 视觉 | `[T,B,342]` | `[T,B,342]` |
| 音频 | `[T,B,1582]` | `[T,B,300]` |
| speaker mask | `[T,B,2]` | `[T,B,9]` |
| utterance mask / label | `[B,T]` | `[B,T]` |

固定输入维度见 `train_cm.py:495-530`。GRU/线性编码后每模态维度为 `D_g`，有效 utterance 数为 `N=sum(lengths)`。三模态图有 `3N` 节点；官方 `concat_DHT + avl` 且 `use_residue=False` 时，分类前 hidden 为 `[N,3072]`，输出为 `[N,6]` 或 `[N,7]`（`model_cm.py:694-698,865-873`）。

## 问题清单

| 严重度 | 文件与位置 | 问题 | 影响 / 处理方案 |
|---|---|---|---|
| 严重 | `train_cm.py:745-773`；`train.py:468-512` | 正式训练传 `valid=0.0`，每轮在 test 上评估并按 test loss/F1 选最佳结果。 | 测试集泄漏。保留官方协议并明确标注；新增 validation-selected 协议，test 只在最终 checkpoint 上运行一次。 |
| 严重 | `train.py`、`train_cm.py` 全文件 | 无 `torch.save`；只保存部分摘要或不保存。 | README 的“训练 M³Net → 计算 difficulty”无法闭环，也不能断点续训。新增不改数学定义的 checkpoint 管理层，保存模型、优化器、轮次、配置和 RNG 状态。 |
| 高 | `README.md:69-72` | MELD difficulty 命令误调用 `train_cm.py`，后者无 `--ckpt_path`。 | 命令必然失败。正确入口为 `caldiff/cal_diff.py`。 |
| 高 | `caldiff/cal_diff.py:358-360` | README 使用 batch 16，但脚本每批仅以 `data[-1][0]` 写一条 difficulty，数值却汇总整批 utterance。 | 难度缺项且混合对话。忠实生成时强制 `batch_size=1`；若修代码，需逐对话聚合回归测试。 |
| 高 | `train_cm.py:731-734` | `--testing` 调用 `train_or_eval_graph_model` 的参数签名错误。 | CM 入口测试直接 `TypeError`。官方 checkpoint 评测使用 README 指定的 `train.py --testing`；规范评测工具另建。 |
| 高 | `train_cm.py:286-288,320-321`；`model_hyper.py:142,191-251,340`；`clloss.py:299-311,344-348` | 多处无条件 `.cuda()`。 | `--no-cuda` 不能使 CMERC/hyper 路径在 CPU 工作。以 device-aware 最小补丁修复，并做 CPU/GPU 回归测试。 |
| 高 | `dataloader.py:66-70` | MELD 读入 `validIds` 但 `self.keys` 使用含 train+valid 的 `trainVid`；官方 validation 实际混入训练。 | 规范协议必须按 ID 显式拆分；官方协议照实报告。 |
| 高 | `train_cm.py:84-99`；`train.py:65-80` | IEMOCAP 即使设置非零 valid，训练 loader 也不使用已创建的 `train_sampler`。 | validation 对话仍进入训练。规范 loader 必须排除验证 sampler。 |
| 中 | `train_cm.py:348-356` | `losses.append(loss.item())` 位于加入 CC/HSCL 之前。 | 日志中的 train loss 只有分类损失，不是实际反传总损失。新增分量化日志，不改变反传公式。 |
| 中 | `train_cm.py:181-190` | 辅助模型仅在每个 epoch 开头复制主模型，并强制 eval；主模型处于 train。 | 反事实标签基于滞后模型，且混入 dropout/BN 状态差异。属于忠实实现语义，不静默修正；在报告中披露。 |
| 中 | `train.py:216-229`；`train_cm.py:361-374` | `vids += data[-1]` 位于 batch 循环外。 | 只保留最后一批 ID。规范预测导出工具逐批保存 ID。 |
| 中 | `train.py:482-488`；`caldiff/cal_diff.py:669` | `torch.load` 无 `map_location`。 | 跨设备加载不稳定。评测工具显式指定 device。 |
| 中 | `train.py:161-172` | 评估未使用 `torch.no_grad()`。 | 不必要的计算图和显存占用。规范评测工具使用 inference/no-grad。 |
| 中 | `train_cm.py:392,412,446` 等 | `store_true, default=True` 导致 CLI 无法关闭。 | 部分消融开关无效；消融前必须验证真实代码路径。 |
| 中 | `train_cm.py:465-466,739` | 开启 CL 且未传 `scheduler_steps` 会除零。 | 非 README 调用可能崩溃。入口校验必须拒绝无效配置。 |
| 中 | `model_cm.py:23-48` | `FocalLoss` 强制 CUDA，并对 log-softmax 再做 log-softmax，且累计所有类别。 | 非标准实现；官方配置未使用时不改，若启用必须另行审计。 |
| 低 | `README.md:34-37` | IEMOCAP baseline 命令最后残留续行反斜杠。 | 直接复制后 shell 等待下一行。复现指南给出可直接运行命令。 |

## Checkpoint 与优化器

`train_cm.py:713` 只将主模型参数加入 Adam。三个辅助模型在 `no_grad()` 中生成反事实结果，CKSCL 无可训练参数，因此该优化器范围合理。官方 checkpoint 是单一 `OrderedDict` state dict，包含推理所需主模型权重，但不含配置、优化器、轮次、课程进度或随机状态，不能用于完整断点续训。

## 论文与代码的协议边界

论文第 4 页称超参数用 hold-out validation 调整，并报告 5 次随机运行平均；官方源码没有实现这一选模协议，也未公开 5 个 seed。论文 Table 5 未明确 ECE bins、RMSCE 细节或 AUPRC 正类方向。规范评测将把 15 个等宽 bins、最大 softmax 置信度、AUPR-Success 等设为显式配置和“依据引用的实现假设”，不能称为论文完全公开的协议。
