# 学习率分析候选资格实现澄清

记录时间：2026-10-23T09:17:07.230249+00:00。本记录补齐分析程序对原训练前登记的执行，不修改原登记、registry、训练源码、超参数、任务矩阵或停止规则。

## 原规则与实现偏差

原[PREREGISTRATION.md](PREREGISTRATION.md)“选模、候选与停止规则”第一条规定：最佳验证W-F1高于XP04且末10轮均值不低于XP04，才登记为值得第二Seed验证的候选。随后“若多个候选”的排序，仅适用于该合格集合。registry的strong/weak字段也将“最佳升、末段降”定义为混合线索。

旧choose_candidate虽然正确计算strong/weak标签，却从所有最佳值改善的run中排序。因此某个末段下降的weak run可能以更高峰值盖过真正合格的strong run；仅存在weak时仍生成recommended_lr，也错误暗示第二Seed资格。

## 最小修复

1. 未完整完成两项lr时保持pending，不做最终资格判断。
2. 先筛选Δbest>0且Δlast10>=0的合格强候选，仅在此集合内按best、last10排序，完全相同优先5e-5。
3. 无强候选但存在weak时，decision仍报告混合线索，recommended_lr和run_id均为null，weak_run_ids只作描述列表。
4. 两项均无最佳改善则stop_lr_search，不扩大范围；本脚本从不自动启动第二Seed。

## 与实际结果的时序边界

修复记录时只读取metrics.jsonl的epoch字段确认进度，不提取当前分数用于修规则。UTC快照为2026-10-23T09:17:07.230249+00:00：

| Run | 已完成/计划Epoch |
|---|---:|
| MELD_cmerc_lr5e-5_seed67137 | 19/40 |
| MELD_cmerc_lr2e-4_seed67137 | 16/40 |
| IEMOCAP_cmerc_lr5e-5_seed1475 | 0/80 |
| IEMOCAP_cmerc_lr2e-4_seed1475 | 0/80 |

两项MELD均未进入第31–40轮尾窗，两项IEMOCAP亦未完成计划训练；没有任何最终last10结果可供本修复择利。原登记的阈值、排序优先级和预算均未改变。修复使用合成反例，不解析测试指标/预测，不执行GPU或模型前向。

## 红—绿CPU证据

先在原实现运行7项合成回归，其中恰好2项因预期行为失败：weak高峰覆盖strong；仅weak错误产生推荐。其余排序、平局、停止、未完成门禁和输入不变检查通过。随后最小修复，同一7项全部通过。原始输出及UTC、修复前后脚本SHA见[analysis_rule_regression.log](../logs/analysis_rule_regression.log)，机器记录见[analysis_rule_fix.json](../provenance/analysis_rule_fix.json)。

分析main与重绘没有执行；由主代理统一更新预览和最终交付。训练与科学结论不受本次代码修复自动改写。
