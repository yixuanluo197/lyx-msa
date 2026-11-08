# CMERC GPU 对齐与单模态行为鲁棒性 Pilot

本阶段已经完成并通过独立验证。完整实验说明和文件导航见 [原始 README](README.md)。本目录是本地原实验批次的逐文件同哈希副本；原文件不因上传而改写。

## 已完成的工作

- 全部10个 XP04 validation-selected checkpoint 的 Test GPU 原始概率对齐通过，FP32/eval/ATen、batch16、原始顺序，atol=1e-6、rtol=1e-5。
- 全部10份真正 Validation 预测：IEMOCAP 12对话/527话语；MELD 114对话/1109话语。
- IEMOCAP seed1475、MELD seed67137 的 Audio → Visual → Text Representation Pilot：Gaussian 0.05、Dropout 0.05各5次，Attenuation 0.95一次。
- 53988个扰动概率向量、14724行话语行为指标、18组汇总；27项测试、76次图索引缓存等价检查及独立指标重算通过。
- 每次仅改变目标话语指定模态，保留原批次组成、padding、mask与图结构；没有重新训练、联合模态、额外seed扰动或温度拟合。

## 主要结论

| 数据集 | Audio Dropout翻转率 | Visual Dropout翻转率 | Text Representation Dropout翻转率 |
|---|---:|---:|---:|
| IEMOCAP | 1.442% | 0.076% | 0.759% |
| MELD | 0.721% | 0.307% | 0.902% |

18组控制原始置信度后的增量AUROC 95%区间全部跨0；辅助正确性预测器的二元Brier/NLL各有15/18组点估计退化，尚未确认稳定的独立正确性信息。Stable but Wrong普遍，暂不建议直接推进A+V。完整指标、分层、局限和案例见 [诊断报告](reports/SINGLE_MODAL_PILOT_REPORT.md)。

IEMOCAP 1条、MELD 95条Validation视觉特征全零，Dropout/衰减对其无输入作用。全部实验均为特征空间扰动，不能解释为语义文本、音调音色或真实视频物理扰动。

## 公开内容与来源

上传代码、配置、报告、原始及扰动预测NPZ、CSV、验证记录和诊断运行日志。两份train_statistics.npz仅含train-only逐维均值/标准差/数量，未包含逐话语特征。未上传模型权重、原始媒体、原文语料、逐话语输入特征或训练日志。

原实验位于本地 `/data2/yb/paper/lyx-msa/experiments/xp_caldist_behavioral_calibration/20261107T172302Z`；GitHub仓库根目录对应本地 `CMERC/`。原脚本/JSON中的绝对路径用于来源追踪，不保证在克隆机器上存在。完整GPU重跑及verify_results.py的原数据/权重来源核查需要本地只读XP04批次；公开副本足以离线重算所有本阶段诊断。

## 克隆后离线复核

在已有兼容环境中运行，不自动安装依赖、不使用GPU。原环境版本见 [runtime_environment.json](provenance/runtime_environment.json)。

从仓库根目录执行单元测试，显式指定冻结指标模块位置，以适配公开仓库布局：

```bash
RUN=experiments/xp_caldist_behavioral_calibration/20261107T172302Z
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD/experiment_records/04_xp_mainline_validation/code" \
python -m unittest discover -s "$RUN/scripts" -p 'test_*.py'
```

若需从预测重新生成CSV和诊断，先复制本目录到一个新的输出目录，避免覆盖冻结报告和来源清单，再执行：

```bash
REPO="$PWD"
RUN="$REPO/experiments/xp_caldist_behavioral_calibration/20261107T172302Z"
OUTPUT="/path/to/a/new/pilot-recomputed"
cp -a "$RUN" "$OUTPUT"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$REPO/experiment_records/04_xp_mainline_validation/code" \
python "$RUN/scripts/analyze_pilot.py" --root "$OUTPUT"
```

该离线命令只读取Validation预测，不读取Test预测、数据特征或checkpoint。重新生成的报告不含人工追加的阶段判断；原判断保留在冻结报告中。

[verification.json](verification.json) 是原实验验收；[artifact_manifest.json](provenance/artifact_manifest.json) 是原151文件哈希；publication_manifest.json另外记录全部152个原文件（包含原manifest）在本次公开副本中的一致性。
