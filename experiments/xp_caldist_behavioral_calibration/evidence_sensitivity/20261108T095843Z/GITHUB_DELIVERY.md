# CMERC 模态证据敏感性：Validation Pilot

本阶段已完成。四份科研报告、六组 SVG/PDF/PNG 图片和全部冻结预测保存在本目录；[完整报告](reports/FINAL_EVIDENCE_SENSITIVITY.md)逐项回答十个研究问题，[README](README.md)说明原始运行过程，[verification.json](verification.json)记录原环境中的独立验收。

## 范围与核验

- 仅 IEMOCAP seed1475：12对话／527话语；MELD seed67137：114对话／1109话语，均为真正 Validation Split。
- Text 四个存储数组、T-All、Audio、Visual：训练均值收缩 rho=0.25/0.50/0.75/1.00；随机维度均值掩盖 ratio=0.10/0.30/0.50/0.80，各5次。两种操作独立报告。
- 112个条件、274848条保存概率向量、272544次有效目标前向；2304条视觉缺失观测保留原预测。所有干预均为完整原batch16中的单目标话语干预。
- 六个GPU任务均exit0；25项原环境测试、全部逐话语指标／汇总／类别结果及2000次对话级配对bootstrap区间独立复核通过。GPU推理墙时94.40分钟。
- 未训练、未执行Test／联合模态／五Seed正式实验，未拟合Reliability Score或Temperature Scaling。

## 主要结果

以下为目标表征完全替换训练均值时的准确率：

| 数据集 | 原始 | T-All | Audio | Visual |
|---|---:|---:|---:|---:|
| IEMOCAP | 66.03% | 62.81% | 62.24% | 66.22% |
| MELD | 66.28% | 39.68% | 65.46% | 65.73% |

MELD文本准确率变化为−26.60个百分点，95%区间[−30.53, −22.45]；IEMOCAP音频变化为−3.80个百分点，区间[−7.45, +0.70]，准确率变化仍有较大不确定性。文本T1的翻转／真实类概率变化较大，但不同指标给出的数组排序不同，不能认定唯一重要编码层。

IEMOCAP 1条、MELD 95条全零视觉输入保留且不做V干预。视觉非缺失组单独分析。另有1／9条缺失视觉错误样本，在分别进行T-All和Audio完全收缩后仍保持原错误预测，不能把稳定视为正确。

这些是目标预提取表征的依赖结果，保留了其他话语的同模态上下文。训练均值不是已证明的情绪中性参考；不声称移除了语义、音调、音色或面部表情，也没有证明可改善校准。

## 文件与来源

- [逐话语结果](results/per_utterance_sensitivity.csv)、[模态汇总](results/modality_sensitivity_summary.csv)、[类别结果](results/classwise_sensitivity.csv)、[视觉缺失分析](results/visual_missingness_analysis.csv)。
- [正确／错误组直接配对差值](results/correctness_group_contrasts.csv)、[前批轻扰与强削弱的交叉统计](results/previous_pilot_sensitivity_comparison.csv)。
- `runs/`包含原始Validation概率、112组干预完整概率／log概率，以及恢复用的预测快照；没有完整输入特征副本。
- [配置](configs/evidence_sensitivity_config.json)、[预注册](reports/PREREGISTRATION.md)、[来源哈希](provenance/source_hashes.json)、[六任务日志索引](provenance/execution_jobs.json)、[图片复核](figures/FIGURE_QA.md)。

原实验目录为本地 `/data2/yb/paper/lyx-msa/experiments/xp_caldist_behavioral_calibration/evidence_sensitivity/20261108T095843Z`，仓库根目录对应本地`CMERC/`。所有430个原文件（含原artifact manifest）逐文件同哈希复制，原目录未改写。仅新增本交付页、公开离线复算适配器和公开验证／哈希清单。

未上传checkpoint、原始媒体、原始语料或逐话语输入特征。来源JSON／原始脚本中的绝对路径保留为溯源记录，不保证在克隆机器上存在。完整`verify_evidence.py`的原数据和权重审计需要原本地只读环境；其已通过结果保存在`verification.json`，不是克隆后重新执行原数据审计的承诺。

## 克隆后离线复算

公开文件足以重算全部九份指标CSV、共享bootstrap权重，以及与[前一阶段](../../20261107T172302Z/GITHUB_DELIVERY.md)的交叉统计。使用已有兼容环境，原版本见[runtime_environment.json](provenance/runtime_environment.json)；不自动安装依赖，不使用GPU。

从仓库根目录执行，`--output`必须是尚不存在的新目录：

```bash
RUN=experiments/xp_caldist_behavioral_calibration/evidence_sensitivity/20261108T095843Z
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python "$RUN/scripts/recompute_published.py" --output /path/to/new/evidence-recomputed
```

适配器复制到新目录，仅调整前批公开文件位置；复制的原验收与来源清单会移入`provenance/published_reference/`，避免误当作新输出的来源审计。它调用未改写的冻结分析代码，将九份重算CSV与公开原件按字节比对，并逐数组比对两份bootstrap权重；状态写入新目录`offline_recomputation.json`。它不会加载模型、拟合校准参数或读取Test预测。人工追加的Stable-but-Wrong报告段落及完整复算逻辑保留在原报告和[stable_missing_check.json](logs/stable_missing_check.json)。

该命令已在独立临时目录实测，九份CSV、两份权重数组全部一致，并验证拒绝覆盖已有目录，见[publication_verification.json](publication_verification.json)。另已验证20项可移植指标／分析测试；原25项中的runner控制测试依赖本地冻结XP04与前批脚本路径，不能直接当作任意克隆环境的测试承诺。

[publication_manifest.json](publication_manifest.json)记录430个原文件的复制一致性及新增交付文件哈希；原[artifact_manifest.json](provenance/artifact_manifest.json)保持原字节不变。
