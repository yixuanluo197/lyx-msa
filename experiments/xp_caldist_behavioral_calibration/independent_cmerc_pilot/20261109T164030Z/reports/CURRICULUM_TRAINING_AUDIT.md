# Curriculum Training Audit

两个 Pilot 均严格使用 Fold-local Difficulty：

- IEMOCAP SHA-256：`3b53dc2a8baf32ce23df8f4c2793fefef5ac05201f7d99b38c1868def7115149`；84 个 Training Dialogue 完整覆盖。
- MELD SHA-256：`ec0dd56f4fec836cbb864c88b274b11c15137700c5d3f87170359db5949fe7a0`；852 个 Training Dialogue 完整覆盖。

IEMOCAP 比例为 0.15、0.30、0.45、0.60、0.75、0.90、1.00，此后保持 1.00；对应选择 12、25、37、50、63、75、84 个 Dialogue。MELD 比例为 0.40、0.80、1.00，此后保持 1.00；对应 340、681、852 个 Dialogue。

每轮对当前 curriculum 选中集合恰好覆盖一次，无重复、无遗漏；比例达到 1.00 后每轮完整覆盖全部 Training Dialogue。Development 每轮完整覆盖且不更新参数。独立复算的每轮 Difficulty 排序、选择成员和日志完全一致。

容量上限为 1000 edges / 5200 nodes。实测最大值：IEMOCAP 1000/2856，MELD 343/885，均未越界。Calibration 和 Evaluation ID 从未进入 Curriculum sampler。
