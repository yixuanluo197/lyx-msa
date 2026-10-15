# GPT / 人工代码审查指南

## 建议阅读顺序

1. 根目录 `README.md`：确认仓库来源、目录与公开范围。
2. `CAUSAL_README.md`：确认时间因果约束和已验证边界。
3. `REPRODUCTION_GUIDE.md`：确认数据划分、选模和恢复协议。
4. `model_cm.py` → `model_hyper.py` → `causal_graph.py`：核对模型、mask、图边方向和归一化。
5. `dataloader.py` → `train_cm.py` → `clloss.py`：核对数据、信息移除、损失和 checkpoint。
6. `caldiff/cal_diff.py` → `scripts/evaluate_checkpoint.py` → `evaluation_metrics.py`：核对难度生成与评测一致性。
7. `tests/test_causal*.py` 及其余 `tests/`：评估测试是否真正覆盖所声称的行为。
8. `reports/01_CODE_AUDIT.md`、`reports/07_REPRODUCTION_COMPARISON.md`：参考历史审计和已有实验边界。

## 可直接复制的审查请求

```text
请审查这个仓库：https://github.com/yixuanluo197/lyx-msa

先读取 README.md、CAUSAL_README.md、REPRODUCTION_GUIDE.md，
再按 docs/GPT_REVIEW_GUIDE.md 的顺序读取实际源码和测试。
请列出你实际读取的文件；若无法获取源码，先明确指出缺失文件。

重点检查：
1. causal=True 时是否存在未来信息泄漏：双向编码、Transformer mask、
   padding、归一化统计、超图回传、图度归一化和模态移除的均值。
2. 主模型与模态／上下文移除分支是否遵循一致的信息可见性约束。
3. IEMOCAP/MELD 的训练、验证、测试划分和 checkpoint 选取是否正确；
   区分 official_test 与 validation 协议。
4. 损失、梯度、设备、张量维度和可训练参数是否符合实现意图。
5. 完整断点恢复是否校验架构和实验协议，并恢复优化器与随机状态。
6. 课程难度文件、因果 checkpoint 和非因果 checkpoint 是否被明确隔离。
7. 测试是否有遗漏，是否只验证了实现自身而没有验证要求。

只报告有源码依据的问题。每项提供严重程度、文件与行号、触发条件、
影响、修复建议和最小验证方法。区分确认的缺陷、需要验证的疑点
及已在文档中明确披露的限制。不要把未训练的因果变体写成已有性能结果。
```

## 审查材料边界

公开仓库没有数据特征、模型权重和完整运行产物。仅凭仓库可以审查实现和测试，不能独立确认历史性能数字，也不能验证未经公开的原始特征提取链。

报告带有各自的记录日期；模型后续修改可能晚于部分报告。比较上游代码可使用保留的版本：

```bash
git diff 815c7c060914f5b55db816c0ba3662b11d08552f HEAD -- \
  model_cm.py model_hyper.py dataloader.py train_cm.py clloss.py caldiff/cal_diff.py
```

如果使用的审查工具无法读取仓库，可在 GitHub 的 `Code` 菜单下载 ZIP，或提供相关源码文件与本指南。
