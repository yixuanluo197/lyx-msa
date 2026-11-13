# E3 Pilot 最终判断

**结论：部分有效，但核心行为增量假设尚未得到支持。** MELD 的 E3-A/B 在这个 Seed 的 OOF 点估计上同时优于原始 E1；IEMOCAP 没有三指标一致收益。MELD 的优势不能直接归功于随机扰动行为，因为只用 `logit(C)` 的 L0 也优于 E1，而且 E3-A/B 的 Brier/NLL 反而不如 L0。

1. IEMOCAP 原始 C 整体偏高；E3-A/B 将均值降至约 0.66，但 ECE 的下降伴随 Brier/NLL 恶化。E3-C/D 更差。
2. MELD 原始 C 整体偏低；E3-A/B 将均值升至约 0.67，但大部分错误预测也被提高，且行为特征未稳定超越 L0。
3. μ 与 δ 联合没有比 `logit(C)` 单特征带来一致增量；JS 在完整 E3-B 中无一致收益。单独 JS、视觉消融的微小 MELD 点估计信号均带宽区间且属于多配置探索。
4. 完整模态输入 E3-C 没有展示互补收益；预测类别 one-hot 的 E3-D 明显恶化。
5. MELD E3-A/B 相对 E1 的三项差值的 Dialogue bootstrap 95% 区间均跨 0，IEMOCAP 只有 12 个 Dialogue，不能把单 Seed OOF 点估计提升当成稳定结论。

## 后续建议

若研究者希望按“E3-A/B 在 MELD OOF 点估计优于 E1”的预定规则继续，建议**只针对 MELD**进行五 Seed 确认，并将 E1、L0、E3-A、E3-B 同场固定比较；五 Seed 不能把共享 Dialogue 当作独立数据集。IEMOCAP 的 E3-C/D 与类别 one-hot 暂不推进。只有行为组跨 Seed 在 ECE、Brier、NLL 上稳定超过 **L0 与 E1**，才可称为随机扰动行为的增量价值。当前不建议进入原始文本、音频、视频干扰阶段。

本轮到此停止，等待研究者审核。未重新训练 CMERC，未运行 GPU 推理，未生成新扰动，未使用 Test，未训练五 Seed，未提交或推送 Git。
