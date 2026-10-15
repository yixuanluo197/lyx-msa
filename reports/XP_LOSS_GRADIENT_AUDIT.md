# 默认非因果 CC / MSCL / CSCL 梯度审计

日期：2026-10-14 UTC。仅检查源码和 CPU toy 张量；匹配环境 torch `1.13.1+cu117`，使用 `CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1` 和 `scripts/with_cmerc_env.sh`。没有 GPU workload、optimizer step、训练代码/批次改动或追分修复。

## 结论

**未发现 `ft_scl=hidden` 附近使默认非因果 CC/MSCL/CSCL 全局断梯度的问题。** 实际训练损失代码块和实际 CKSCL 模块在 CPU toy 数据上，三个分量都回传到主 hidden，梯度有限且非零；通过实际 `reverse_features` 切片/拼接后的 MSCL/CSCL 也回传到其上游图张量。与 `815c7c0` 上游相同路径的损失和梯度完全一致。

单独存在继承自上游的**条件性**无梯度情形：CKSCL 的非有限损失保护返回不带梯度的零；单 utterance toy 例子实际触发。它不能被描述成所有 HSCL 都失效，当前没有正式 batch 触发频率或它造成分数差距的证据。

## 实际路径及 grad mode

- 主 `Model.forward` 的 hyper 分支在 `model_cm.py:893-901` 取得图输出，`emotions_feat_clone=emotions_feat` 为返回 hidden；没有 detach。其后 dropout、ReLU、分类器形成分类 log-probability。
- `HyperGCN.reverse_features`（`model_hyper.py:309-325`）按对话对图 tensor 切片，再 concat 模态块；在主模型 forward 期间执行，**发生在训练器的 no-grad 信息移除块之前**。上游与当前该方法 AST 完全相同。
- `train_cm.py:428-455` 的 no-grad 块仅保留 `ft_scl=hidden` 的别名并计算辅助移除模型输出/预测。仅凭别名不能证明之后的运算可微；本审计因此继续检查实际消费者。
- CC（`train_cm.py:457-478`）在退出 no-grad 后累加五路移除输出与主 `log_prob` 的真实类正差。主输出仍接入图，辅助输出不求梯度；条件选择本身不可微，这是原算法的分段路径。
- MSCL/CSCL 调用位于 `train_cm.py:541-542`，同样**在 no-grad 块外**。`CKSCL_base`（`clloss.py:211-261`）的 view/unsqueeze/normalize/unbind/concat、矩阵相似度和 log 均在恢复的 grad mode 中。默认 `contrast_mode='all'`，`features[:,0]` 属于未启用的 `'one'` 分支；没有隐藏的 no-grad 包装或 feature detach。
- `create_class_weight_SCL` 对伪标签 `.cpu().detach().numpy()`，用于离散类别计数；被 detach 的不是 hidden。`CKSCL_tav/context` 新建的权重 tensor 没有梯度，但它们相乘的 logits 保持图。

## CPU 实证

从实际 `train_or_eval_graph_model` AST 提取完整 `if args.contrastlearning or args.calibrate` 块，包括原 no-grad 范围、五路信息移除调用、伪标签、CC 和 HSCL；并执行实际 `create_class_weight_SCL` 与实际 `CKSCL_tav/context`。测试隐藏张量为 `[6,12]`，固定 CPU seed 9，标签 `[0,1,2,0,1,2]`；主 log probability 由 hidden 的前三列形成，辅助模型为输出序列 stub，产生交替的置信度上升/分类破坏，以覆盖有效正差及两类伪标签。

通过 `torch.autograd.grad(loss, hidden, retain_graph=True)` 分开检查三个分量：

| 分量 | 当前 loss | hidden 梯度 L2 norm | requires_grad / 有限 / 非零 | 与上游 loss、梯度 |
|---|---:|---:|---|---|
| CC | 0.699892520904541 | 0.6926277279853821 | 均通过 | 逐位相等 |
| MSCL | 19.470327377319336 | 17.524587631225586 | 均通过 | 逐位相等 |
| CSCL | 12.980217933654785 | 11.68305778503418 | 均通过 | 逐位相等 |

上游源码由 `git show 815c7c0:train_cm.py/clloss.py` 内存加载。上游无条件 `.cuda()` 阻止原样 CPU 执行，因此**仅在内存测试模块中**将设备转移改成 CPU/feature device，并把旧 device 判断改为 feature device；没有更改损失运算、grad mode 或持久源码。以上上游比较是“仅设备适配后的 CPU 运算对比”，不是未经修改的上游 CUDA 实测。

对实际 CKSCL 的 `torch.cat(torch.unbind(...))` 增加仅进程内 observer，三条 MSCL 子路径均观测到 `(torch.is_grad_enabled=True, requires_grad=True, grad_fn=CatBackward0)`。负对照把完整 CKSCL 调用人为放进 no-grad，loss 的 `requires_grad=False`；实际代码采用的是前一种范围。

另用实际 `reverse_features` 将 `[18,4]`、`requires_grad=True` 的图 tensor 按 `[2,4]` 对话长度重排成 `[6,12]`；保留真实切片/concat 运算，再在 no-grad 中赋别名、退出后使用实际 CKSCL。固定 seed 17、两类伪标签：MSCL 和 CSCL 均回传有限非零梯度到原图 tensor，72 个元素全部非零。这证明相关切片/concat 没有在此 CPU 复现中切断主图输出之前的连接。

## 条件性边界与未验证范围

- `clloss.py:307-310,346-349` 保留上游的 Inf/NaN 检查，并用 `torch.zeros_like(loss)` 返回零。该零默认 `requires_grad=False`。N=1 的实际 CKSCL_context，以及设备适配的上游 CKSCL_tav/context 均返回 `0.0 / requires_grad=False`。若发生于训练，HSCL 分量不提供梯度；分类/有效 CC 路径仍可反传。本审计没有发现正式 batch 的触发计数，不能将 toy 例子当作本批正式训练失效证据。
- CC 没有任何正增量时原代码可能保持整数零，这也是条件性没有 CC 梯度，不是结构性断图。
- 未做默认非因果全模型 CPU forward：核心图构造仍有 GPU 绑定。本实证覆盖真实损失块、真实 CKSCL 和真实图输出重排；辅助模型与主图前向使用 toy/stub，不覆盖 CUDA scatter/GRU 全模型数值及全部主参数的实际梯度。
- 未测真实运行每个 batch 的梯度、非有限保护触发率或性能影响。因而不能把上述条件性保护或 loss 数值关联写成 XP/V100/论文差距的确定原因；没有为此修改或重启正式训练。
