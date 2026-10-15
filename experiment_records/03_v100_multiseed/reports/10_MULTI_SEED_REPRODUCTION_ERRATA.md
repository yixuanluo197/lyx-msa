# V2冻结报告文字勘误

自动报告生成器 `scripts/summarize_results.py:339` 对V1容量失败写作“第5epoch”。该数值是内部零基epoch=5，对应人类计数**第6个epoch**；V1此前完整完成5个epoch。正确证据及处理见 `PROTOCOL_AND_RESOURCE_AUDIT.md` 和原V1失败日志。

为避免修改正在正式运行的冻结执行源码，本批次保留此独立文字勘误。该问题不影响实际训练、checkpoint epoch、选模、曲线、数值统计或任务门控；这些使用逐轮日志中的一基epoch。下一批源码更新时再修正生成器文字，不改本批科学结果。
