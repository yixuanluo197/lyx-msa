# CMERC Hierarchical Predicted-Class Temperature Scaling Pilot

本目录记录严格限制在真正 Validation Split 上的校准方法探索。十个 XP04 checkpoint 均复用已保存完整概率；未进行 GPU 推理、CMERC 重训练、checkpoint 修改或 Test 访问。

四种方法均以外层 Leave-One-Dialogue-Out OOF 概率评价：E0 原始 CMERC，E1 Global TS，E2 不正则化预测类别温度，E3 层次化预测类别温度。E3 的 λ 仅由外层训练对话内的 5 折 Grouped CV 选择；推理时仅由**原始预测类别**选择一个正标量温度。

## 五 Seed OOF 均值

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

主要结论：IEMOCAP 上 E3 相对 E1 的 NLL 与 Brier 在 5/5 Seed 均退化；MELD 的差异接近零且方向不一致。E3 明显压缩了 E2 的类别温度偏移并改善其过拟合，但没有建立超越 Global TS 的净收益。因此当前结果不支持进入历史 Test 或把类别条件化校准作为正式方法。

## 复现

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_hierarchical_pilot.py --output .
python scripts/plot_results.py
python scripts/verify_results.py --output .
```

完整结果见 [FINAL_RESEARCH_DECISION.md](FINAL_RESEARCH_DECISION.md)，预注册见 [PREREGISTRATION.md](PREREGISTRATION.md)。
