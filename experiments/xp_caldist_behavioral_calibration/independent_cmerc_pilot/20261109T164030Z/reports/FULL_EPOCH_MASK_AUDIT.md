# Full Epoch and Unknown-Label Mask Audit

真实首轮训练门槛通过。首轮直接使用真实 CMERC 图网络、真实 Fold-local Difficulty、CapacitySafeBatchSampler 和全部六视图前向；完成 backward 与 optimizer.step 后继续同一模型、优化器和 RNG 状态，未重新初始化。

IEMOCAP Fold 1 的 23 条 Training unknown 全部保留特征、qmask、umask、图节点和上下文。首轮 curriculum 子集包含 1 条 unknown，用于真实梯度门槛。以下 11 条标签依赖路径的 unknown 行直接监督梯度绝对值最大值均为 0：Classification、5 个 Rank、3 个 MSCL、2 个 CSCL；有效行 Classification 梯度绝对值和为 1.000000。未知节点仍在完整模型前向中，因此可能通过上下文间接影响其他有效节点；本审计只证明其未被当作真实标签 Anchor、Positive、Negative 或分类目标。

批准 Manifest 的 53 条异常分布为 Training 23、Development 14、Calibration 8、Evaluation 8。前三个已访问角色均应用 supervision_mask；Evaluation 的 8 条继续封存。本轮未修改原始标签、pickle、qmask 或 umask。

独立验证共执行 4,440 条断言，检查 mask、角色覆盖、概率、checkpoint、哈希和数值稳定性，零失败。
