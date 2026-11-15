# XP04统一模块Pilot交付

入口报告：[DATA_RESULTS_REPORT.md](DATA_RESULTS_REPORT.md)。最终判断：[FINAL_DECISION.md](FINAL_DECISION.md)。

正式运行环境 `/data2/yb/conda_envs/cmerc-xp-repro/bin/python`，依赖numpy/pandas/scipy/scikit-learn；图表使用已有base环境matplotlib。运行目录为本目录，源数据路径由同级Pilot目录确定。

复核（无需GPU）：

```bash
/data2/yb/conda_envs/cmerc-xp-repro/bin/python -m unittest test_calibration_core -v
/data2/yb/conda_envs/cmerc-xp-repro/bin/python verify_module.py
```

`run_experiment.py`是完整复算入口，会重新拟合CPU校准器并写入本目录结果；审阅时优先运行只读数值复核入口，保留原运行日志与哈希。`verify_module.py`仅更新verification.json中的核验字段。

数据来源：真正Validation 527/1109条，Seed1475；旧XP04 validation/实际Test预测不使用。详见协议及provenance。
