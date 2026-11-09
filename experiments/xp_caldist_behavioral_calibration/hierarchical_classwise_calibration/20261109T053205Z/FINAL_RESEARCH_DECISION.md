# Final Research Decision

## 决策

**暂停 Hierarchical Predicted-Class TS，不进入历史 Test 或正式五 Seed独立评价。保留 Global TS 作为标准基线。**

| Dataset | Method | NLL | Brier | ECE15 |
|---|---|---|---|---|
| IEMOCAP | E0 | 1.033637 | 0.513962 | 0.136499 |
| IEMOCAP | E1 | 0.948872 | 0.509061 | 0.134696 |
| IEMOCAP | E2 | 0.987616 | 0.518450 | 0.142081 |
| IEMOCAP | E3 | 0.966100 | 0.513831 | 0.136009 |
| MELD | E0 | 1.132528 | 0.527992 | 0.209270 |
| MELD | E1 | 0.995087 | 0.472355 | 0.029213 |
| MELD | E2 | 0.997036 | 0.472765 | 0.028896 |
| MELD | E3 | 0.995081 | 0.472330 | 0.029734 |

1. **类别条件化是否优于 Global TS？** 否。IEMOCAP E3 的 NLL/Brier 在 5/5 Seed 均比 E1 差；MELD 的平均变化近零且方向不一致。
2. **层次正则化是否抑制过拟合？** 是。它将类别 log-temperature 平均偏移缩小约 77%（IEMOCAP）和 80%（MELD），并普遍优于 E2；但仍未超越 E1。
3. **IEMOCAP 是否有值得继续验证的类别收益？** angry 有 5/5 Seed 的小幅类内 NLL改善，neutral/frustrated 有弱信号；sad 明显退化。鉴于整体退化和仅 12 个对话，这只是后续独立数据的假设，不足以立项正式 Test。
4. **MELD 的 Global TS 是否足够？** 对当前 Validation OOF 是。E1 已大幅改善 E0，E3 的增量约为零。
5. **哪些类别应独立、哪些应共享？** 当前没有类别具备足够证据采用独立温度。fear/disgust 等低支持类必须共享或强收缩；其余类别也应暂用 Global T。
6. **温度是否稳定？** MELD 类别温度较稳定但高度接近 Global T；IEMOCAP 跨 Seed 和对话折波动明显。
7. **是否存在超越 ECE 分箱波动的真实收益？** 未建立。IEMOCAP 的 NLL/Brier 明确反对 E3；MELD 的 NLL/Brier变化极小，ECE也不一致。
8. **当前 Validation 是否足够支持方法选择？** 足够否定当前 E3 作为优于 E1 的候选；不足以精确估计 IEMOCAP 类别温度或作确认性推广。
9. **能否构建真正独立划分？** 现有 checkpoint 下不能。需要分组重划并重训练，或引入外部未见数据。
10. **下一步是否进行正式五 Seed独立评价？** 当前不值得。下一项最有价值的单个实验是预先固定 Global TS，并在重新训练后形成的独立 grouped calibration/evaluation split 上确认；若资源有限，先停止新增校准复杂度。

Bootstrap 区间来自固定 OOF 概率的对话级重采样，不包含校准器重新拟合的不确定性。IEMOCAP 只有 12 个对话，区间尤其应保守解释。
