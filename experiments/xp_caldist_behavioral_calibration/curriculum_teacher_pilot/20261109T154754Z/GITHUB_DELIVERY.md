# CMERC Fold-local Teacher Pilot GitHub 交付

本目录公开交付 Protocol B 的两个 Fold 1 Training-only Teacher Pilot、真实六视图 removal parity、mask-aware Difficulty，以及 CMERC supervision-mask Runner 的训练前审计。

## 已验证结论

- IEMOCAP Teacher 使用且仅使用 84 个 Fold 1 Training Dialogue，从随机初始化完成固定 80 epochs。
- MELD Teacher 使用且仅使用 852 个 Fold 1 Training Dialogue，从随机初始化完成固定 40 epochs。
- 两个访问 Tripwire 均报告 `rejected_ids=[]`，未创建 Development、Calibration、Evaluation 或历史 Test Loader。
- 真实 `model_cm.Model`、`Model_Intra`、`Model_Inter` 的 Batch-1 六视图输出通过 `atol=1e-6, rtol=1e-5` parity。
- IEMOCAP Fold 1 Training 中的 23 条 `xxx` 话语保留完整上下文，并在所有标签依赖计算前屏蔽。
- Difficulty 精确覆盖 84/84 和 852/852 个 Training Dialogue，且严格排除其他数据角色。
- 56 项测试通过；两个 checkpoint 均通过 strict loading；本阶段未启动 CMERC，也未执行 Evaluation 前向。

## 公开范围

GitHub 版本包含报告、配置、实现、测试、冻结源码快照、Difficulty JSON、审计 JSON 和可复算哈希。以下大文件保留在实验服务器，不进入 Git：

- Teacher checkpoint (`*.pt`)；
- Training role feature artifact (`*.pt`)；
- 六视图完整概率 (`*.npz`)；
- 逐 batch/epoch 大日志与隔离的失败 checkpoint。

这些本地产物的 SHA-256 记录在 `provenance/source_hashes.json` 和 `verification.json`。失败的首轮普通 shuffle 运行已隔离，未恢复其 checkpoint；正式结果从随机初始化重新训练并使用 XP04 容量安全批采样。

最终判断见 `FINAL_TEACHER_PILOT_DECISION.md`，完整文件边界见 `publication_manifest.json`。
