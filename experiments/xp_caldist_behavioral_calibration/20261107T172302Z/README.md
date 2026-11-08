# CMERC 单模态特征空间 Pilot

本批仅执行十个 XP04 checkpoint 的 GPU Test 原预测对齐、真正 Validation 推理，以及两个指定 checkpoint 的 Audio → Visual → Text Representation Pilot。禁止训练、联合模态、测试集调参和覆盖历史文件。

实施与验证顺序：
1. 记录资源、原环境、源码/数据/checkpoint SHA，验证扰动函数的定域、随机数及指标定义。
2. 原 FP32、ATen recurrent backend、eval、batch16、原顺序对齐全部十个 Test 预测；任何失败禁止 Pilot。
3. 十个 checkpoint 推理原 Validation，检查固定划分、规模、顺序与无交集。
4. 按模态顺序完成两个 anchor checkpoint 的逐目标完整批次前向，保存每个独立类型的五次随机或一次确定性概率。
5. 仅基于保存的 Validation 概率进行诊断、对话级 bootstrap 与置信度条件分析，生成报告及 verification。

所有扰动为 feature-space 操作。Text Representation 独立预注册见 configs/perturbation_config.json，四层各用 train-only 标准差，不声称语义扰动。无新依赖。历史代码只读导入。完整特征只在内存使用，不落重复副本。

## 交付导航

- [GPU原预测对齐](reports/GPU_ALIGNMENT_REPORT.md)
- [真正Validation推理](reports/VALIDATION_INFERENCE_REPORT.md)
- [单模态诊断](reports/SINGLE_MODAL_PILOT_REPORT.md)
- [运行与受控恢复记录](reports/RUN_HISTORY.md)
- [冻结扰动配置](configs/perturbation_config.json)
- [逐样本行为指标](results/utterance_behavior_metrics.csv)
- [模态/类型汇总](results/modality_pilot_summary.csv)
- [同原置信度分层](results/confidence_stratified_behavior.csv)
- [LODO额外信息诊断](results/conditional_information_diagnostics.csv)
- [全零特征标记](results/validation_feature_availability.csv)
- [验证结果](verification.json)
- [来源SHA](provenance/source_hashes.json)
- [运行PID、命令、退出状态](provenance/execution_jobs.json)

十份原始验证集预测保存在 `runs/<dataset>/seed<seed>/validation_predictions.npz`。两个anchor的十八份扰动结果保存在其 `perturbations/`，显式区分 `original_predictions` 与 `perturbed_predictions`；同时保存完整probability和log probability。

只基于保存预测重算本阶段全部诊断，无需GPU或模型前向：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data2/yb/conda_envs/cmerc-xp-repro/bin/python scripts/analyze_pilot.py \
  --root /data2/yb/paper/lyx-msa/experiments/xp_caldist_behavioral_calibration/20261107T172302Z
```

命令假设当前目录为本README所在目录。审计原数据来源和训练统计量的独立验证入口为 `scripts/verify_results.py --root <本目录绝对路径>`，需要原始只读XP04文件可用，不需要GPU。不要在已有批次重启模型推理或覆盖预测；复跑需创建新时间戳目录。

结果尺度：汇总ECE15、AUROC Success为百分数；Accuracy、Flip Rate、μ、confidence变化为原始比例；Brier为多分类平方误差总和均值，NLL、JS为自然对数单位。随机扰动指标按每次重复分别计算后平均，不是概率ensemble。

本阶段只提供两个单seed验证集的探索性诊断。LODO监督分类器只用于检查原始置信度之外的信息，不作为已部署Reliability Score或CMERC概率校准器。其每个待评估对话的标签不参与该折拟合。没有拟合温度，没有新测试集方法评估，没有训练CMERC。
