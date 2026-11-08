# CMERC Validation Modality Evidence Sensitivity

本目录承接 `20261107T172302Z` 单模态行为扰动Pilot，独立研究目标话语的预提取表征依赖程度。仅运行 IEMOCAP seed1475、MELD seed67137 的真正Validation Split。未训练CMERC，未使用Test，未拟合可靠性评分或温度，未执行联合模态/额外seed。

- 科学预注册：[reports/PREREGISTRATION.md](reports/PREREGISTRATION.md)。机器可读配置：[configs/evidence_sensitivity_config.json](configs/evidence_sensitivity_config.json)。
- 输入及train-only均值/标准差独立复算：[provenance/audit.json](provenance/audit.json)。
- 全原始batch与强干预的原模型/索引缓存完整输出对照：[provenance/graph_cache_parity.json](provenance/graph_cache_parity.json)。仅采用eager；不采用未通过等价性的CUDA Graph技术探索。
- 完整过程、GPU UUID、PID、运行命令、退出码及资源：[provenance/execution_jobs.json](provenance/execution_jobs.json)、`logs/`。
- 推理脚本：[scripts/run_evidence.py](scripts/run_evidence.py)；干预/单样本指标：[scripts/evidence_core.py](scripts/evidence_core.py)。
- 预测文件：`runs/{dataset}/seed{seed}/baseline_validation.npz` 与 `conditions/{condition_id}.npz`。每个条件保存全部目标概率、log概率、标签及原始上下文身份元数据；`active=False`明确标记缺失视觉跳过，未保存重复的完整特征。
- 主要结果：`results/per_utterance_sensitivity.csv`、`modality_sensitivity_summary.csv`、`classwise_sensitivity.csv`、`visual_missingness_analysis.csv`。summary兼容副本为`sensitivity_summary.csv`。随机五次结果按重复独立计算指标后求平均，不做概率集成。
- 报告：`reports/{TEXT,AUDIO,VISUAL}_EVIDENCE_SENSITIVITY.md`、`FINAL_EVIDENCE_SENSITIVITY.md`。
- 六组真实数据图：`figures/`，SVG/PDF/PNG；图形预注册为`configs/figure_contract.md`。
- 复核：`verification.json`、`provenance/source_hashes.json`与`artifact_manifest.json`。

## 执行顺序与复算

沿用既有环境 `/data2/yb/conda_envs/cmerc-xp-repro/bin/python`（无依赖升级），在健康的本地GPU3运行。严格先两个数据集的整体T-All/A/V收缩，再四层分别收缩，再随机维度均值掩盖。持久监督脚本拒绝覆盖已完成运行；阶段间要求两anchor均通过。命令与环境在execution_jobs中保存。

生成预测后，可直接用原环境运行以下离线分析，不需再次模型前向：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /data2/yb/conda_envs/cmerc-xp-repro/bin/python scripts/analyze_evidence.py --root "$PWD"
/home/yangbin/miniconda3/bin/python scripts/plot_evidence.py --root "$PWD"
```

绘图沿用已安装的Python/matplotlib，不安装依赖。独立复核脚本为`scripts/verify_evidence.py`。分析重建会重写本新目录的派生CSV/报告/图片；不会修改历史目录、模型或特征。完整源码及输入哈希由provenance记录。

## 解释边界

T1–T4是四个存储RoBERTa数组，不是已确认的编码器层编号。训练均值是固定计算参考，不能保证情绪中性或仍位于数据流形；rho=1不等价于物理移除模态。音频与视觉无可靠特征索引映射，因此不宣称干预了音调、音色、韵律或面部动作。全零视觉输入保留且不做V干预，缺失样本与非缺失样本分开分析。

统计采用2000次对话级配对bootstrap，不将话语当独立样本；IEMOCAP仅12个对话，区间不确定性较大。所有类别和强度比较为探索性。敏感性表示对表征的依赖，不能直接视为错误、低可靠性或校准获益。完成本阶段后停止。

## 实际运行记录

六个GPU任务全部exit0，共112条件、274848条保存概率向量、272544次有效目标前向；2304条缺失视觉观测精确保留原预测。单张本地GPU3共享运行，总墙时94.40分钟，完成后GPU计算进程已释放。全部25项CPU测试通过。最终独立复算状态以`verification.json`为准；报告和六组SVG/PDF/PNG图均在本目录。
