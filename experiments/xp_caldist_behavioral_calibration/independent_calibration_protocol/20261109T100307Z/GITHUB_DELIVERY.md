# CMERC Independent Calibration/Evaluation Split Design 交付

本目录公开 CMERC 独立校准/评价数据协议的只读审计与预注册设计。目标是将模型训练、checkpoint 选择、校准参数拟合和最终评价划分为互不重叠的数据角色。

## 审计范围

- IEMOCAP：7,433 条话语、151 个对话、5 个 Session。
- MELD：13,708 条话语、1,432 个对话、190 个 Season–Episode 组。
- 恢复并核验 XP04 原始 train/validation/test 血缘、Session/Episode/Dialogue 分组以及历史研究暴露。
- 比较固定四路分组、五折旋转 cross-fitting 和外部独立数据三类协议。

本阶段没有重新训练 CMERC，没有运行 GPU 前向，没有读取保存的 Test 性能指标，也没有拟合 Global TS 或 Hierarchical TS。

## 推荐协议

推荐五折 Group-level Rotating Cross-fitting：

- IEMOCAP 每折使用一个完整 Session 作为 Evaluation、下一个 Session 作为 Calibration，其余三个 Session 在 Dialogue 层划分 Training/Development。
- MELD 将 190 个 Season–Episode 组平衡划入10个固定块；每折使用2块 Evaluation、1块 Calibration、1块 Development、6块 Training。
- 每个数据集、Seed 和 Fold 都必须重新训练 CMERC；旧 XP04 checkpoint 不能复用。

正式五 Seed 设计共需50个模型，预计约28.8 GPU小时，加入25%余量约36.0小时，完整运行产物约15.1 GB。

## 公开内容与验证

目录包含逐话语数据血缘、Session/Episode/Dialog inventory、候选划分清单、类别覆盖、集合交叠审计、资源估算、预注册配置、来源哈希和完整报告。候选清单标记为 `candidate_not_approved`，不能在研究者审核前作为正式划分使用。

独立验证已通过：21,141条样本唯一，4个候选协议共12个候选折完整覆盖，所有规定的Dialogue/Session/Episode交叠均为零，全部候选角色覆盖完整类别，11个来源文件SHA-256复算一致。

主要结论见 [`FINAL_PROTOCOL_RECOMMENDATION.md`](FINAL_PROTOCOL_RECOMMENDATION.md)，执行边界见 [`CALIBRATION_EVALUATION_PREREGISTRATION.md`](CALIBRATION_EVALUATION_PREREGISTRATION.md)。
