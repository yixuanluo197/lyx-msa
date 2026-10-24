# 第08组：MELD 学习率第二Seed确认报告发布

本页记录第08组实验完成后的GitHub发布范围。[原始交付README](README.md)、[完整中文报告](../../reports/CMERC_LR_SECOND_SEED_CONFIRMATION.md)及实验原始清单保持原字节；README中的“服务器本地交付”和“未自动推送”描述实验结束时的状态。本次按用户明确要求发布审查材料，没有追加正式训练、测试评估或参数搜索。

## 验证集结果

| Seed | 最佳验证W-F1：0.0001 → 0.0002 | 差值 | 末10轮均值：0.0001 → 0.0002 | 差值 |
|---:|---|---:|---|---:|
| 67137 | 65.24% → 66.07% | +0.830 pp | 62.909% → 62.983% | +0.074 pp |
| 2024 | 66.01% → 65.46% | −0.550 pp | 61.223% → 62.034% | +0.811 pp |

本次仅新增MELD seed2024、lr0.0002的一次完整CMERC训练，40轮、验证集选模、零测试前向；0.0001控制来自XP04，seed67137结果来自07。2024的最佳Epoch为5，按原规则属于主指标未改善，末段均值改善作为辅助事实保留。未确认首Seed的最佳F1收益；暂缓最终五Seed与Batch8，不将两Seed称为统计显著。

## 公开文件

- [完整报告](../../reports/CMERC_LR_SECOND_SEED_CONFIRMATION.md)与[原始实验目录](../../experiments/lr_second_seed_confirmation/20261023T175409Z/README.md)
- [训练前登记](../../experiments/lr_second_seed_confirmation/20261023T175409Z/reports/PREREGISTRATION.md)、[配置和完整argv](../../experiments/lr_second_seed_confirmation/20261023T175409Z/configs/registry.json)
- [逐项结果](../../experiments/lr_second_seed_confirmation/20261023T175409Z/results/per_run.csv)、[两Seed对照](../../experiments/lr_second_seed_confirmation/20261023T175409Z/results/comparison.csv)、[汇总JSON](../../experiments/lr_second_seed_confirmation/20261023T175409Z/results/metrics_summary.json)
- [训练/验证曲线CSV](../../experiments/lr_second_seed_confirmation/20261023T175409Z/results/curves.csv)、[全部分项Loss](../../experiments/lr_second_seed_confirmation/20261023T175409Z/results/loss_curves.csv)、[18张PNG/SVG图的哈希](../../experiments/lr_second_seed_confirmation/20261023T175409Z/results/figures/manifest.json)
- 实际执行的源码副本、跳过最终测试的独立补丁、观测器/CPU接口测试、环境、数据和源码哈希、首步预检及逐轮聚合指标
- [最终验收](../../experiments/lr_second_seed_confirmation/20261023T175409Z/provenance/final_verification.json)：40轮训练/验证、2548个Batch、两个Checkpoint及1555个历史文件保全；[报告与图表交付检查](../../experiments/lr_second_seed_confirmation/20261023T175409Z/provenance/analysis_delivery.json)

实际上传文件以[publication_manifest.json](publication_manifest.json)为准。作者已公开的小型`MELD_diff.pkl`与本仓库原文件哈希一致，单独允许公开；其生成历史未知的限制保留。

## 服务器产物与复核边界

两个模型权重、原始数据及其链接、原始训练/资源日志、逐Batch跟踪、锁及缓存保留在服务器。原始[artifact_manifest.json](../../experiments/lr_second_seed_confirmation/20261023T175409Z/provenance/artifact_manifest.json)和冻结清单没有因发布重写，其中的服务器绝对路径或未上传文件哈希仅用于溯源。仅克隆GitHub副本可审查公开源码和聚合证据，完整Checkpoint/逐Batch重算仍需要服务器产物。

沿用实际环境Python3.10.13、PyTorch1.13.1+cu117、PyG2.4.0；绘图另使用已有Matplotlib环境。依赖和数据准备参见[复现指南](../../REPRODUCTION_GUIDE.md)。历史启动器包含防覆盖检查及实际绝对路径，不应直接用来覆盖已完成批次；新运行需单独登记和准备环境。

9项CPU接口及资格规则测试在实验交付前已通过。此轮发布仅核对既有产物、文件哈希、图表输入输出和文档链接，没有重新运行模型或生成测试分数。GPU共驻、CUDA非确定性、作者课程来源和两Seed样本量的解释限制详见完整报告。
