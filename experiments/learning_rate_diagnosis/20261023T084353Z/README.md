# CMERC 学习率诊断：20261023T084353Z

这是已完成实验的冻结记录：新增4次训练，仅比较学习率0.00005、0.0002，使用XP04已有0.0001轨迹作为控制。全部采用验证集W-F1选模，不进行最终测试。

- [公开结果与发布范围](../../../experiment_records/07_learning_rate_diagnosis/README.md)
- [完整诊断报告](reports/CMERC_LEARNING_RATE_DIAGNOSIS.md)
- [训练前登记](reports/PREREGISTRATION.md)、[完整参数及argv](configs/registry.json)
- [源码与输入冻结哈希](provenance/frozen_execution.json)、[跳过最终测试补丁](provenance/skip_final_test.patch)
- [预检首步配对](preflight/LR_first_step_parity.json)、[容量与完整训练池覆盖](preflight/capacity_cpu.json)
- [最终验收](provenance/final_verification.json)、[报告和图表交付检查](provenance/analysis_delivery.json)
- [结果CSV](results/comparison.csv)、[分项Loss](results/loss_curves.csv)、[曲线图清单](results/figures/manifest.json)

## 文件作用

`code_xp04/`是本次实际执行的XP04源码副本，独立增加跳过最终测试的工程开关；训练循环、模型和Loss数学不变。`scripts/watch_training.py`只观察原训练调用和Adam更新，阻止测试前向；相应CPU接口测试与首步GPU预检证据已保留。`scripts/analyze_lr.py`和`plot_lr.py`只读取已保存的训练/验证数据。

`configs/`保存预登记配置，`runs/`公开各项实际manifest和原始逐轮聚合指标，`results/`保存观测器逐轮汇总、四项结果与12张诊断图（各有PNG/SVG）。`provenance/`和`preflight/`保留环境、数据/源码哈希及独立验收证据。

## 运行与审查说明

本批已结束，不要直接执行历史监督器覆盖旧目录。`configs/registry.json`包含实际启动时的Python路径、GPU UUID、环境变量、输出路径和完整argv；这些绝对路径是溯源记录，不是可跨机器直接使用的默认值。

原训练环境为Python3.10.13、PyTorch1.13.1+cu117、PyG2.4.0，完整记录见[environment.json](provenance/environment.json)。设备为本地user23的TITAN Xp GPU3。依照仓库[复现指南](../../../REPRODUCTION_GUIDE.md)准备外部特征和相应环境；克隆后的重跑必须另建新批次、明确替换环境/数据/输出路径、验证冻结输入并重新登记。不能将改路径后的运行冒充本批原始实验。

以下接口和候选排序测试只使用CPU合成数据，不启动正式训练或测试集评估；在已准备上述依赖的环境中，可从本目录运行：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s scripts -p 'test_*.py' -v
```

服务器原环境已执行12项CPU测试。此命令检查观测器不增加调用或改变RNG、测试loader在前向之前被拒绝、独立补丁及候选资格排序。完整`verify_lr.py`还需要服务器Checkpoint、原始逐Batch跟踪及历史数据；`analyze_lr.py`使用实际run绝对路径，不能宣称纯GitHub副本已重新执行完整验收。图表脚本可在具备Matplotlib的环境中，从已发布CSV绘制曲线。

## 产物保全

原始[artifact_manifest.json](provenance/artifact_manifest.json)覆盖服务器产物，发布时不重写。新增`.gitignore`和本README属于发布导航，不属于训练时冻结清单。GitHub实际文件以[第07组发布清单](../../../experiment_records/07_learning_rate_diagnosis/publication_manifest.json)为准；原始数据、8个模型权重、训练/资源原始日志、逐Batch跟踪、锁和缓存不上传。两个小型CPU测试日志作为审查证据单独公开。作者已公开的两个小型难度文件单独允许发布，并保留来源未知的生成历史限制。
