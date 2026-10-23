# CMERC 最小消融：验证集选模与根因定位

本批完成3项预注册、从头训练的正式实验：MELD seed67137关闭CC且保留HSCL/CL，以及IEMOCAP seed1475固定源顺序Batch的Baseline和完整CMERC。每项以验证集两位小数Weighted F1严格提高时选模，同分保留首次最高轮；训练结束只对固定最佳Checkpoint遍历测试集一次。没有增加Seed、搜索系数或因分数重训。

| 实验 | 轮数 | 最佳验证W-F1 % | 选中Epoch | 最终测试W-F1 % |
|---|---:|---:|---:|---:|
| MELD w/o CC | 40 | 65.7700 | 8 | 65.3096 |
| IEMOCAP source_order Baseline | 80 | 65.5800 | 62 | 66.8159 |
| IEMOCAP source_order CMERC | 80 | 64.9600 | 75 | 68.2501 |

MELD最佳验证变化为+0.5300个百分点、末10轮均值变化为−0.1430，尚不能稳定支持CC限制分类性能。IEMOCAP的验证CMERC−Baseline从−1.4300变为−0.6200，末段增量从−1.2690变为−1.0480；这是负增量缩小，两臂绝对验证分数均下降，不能称为CMERC性能提升。单Seed、历史完整CMERC跨物理卡及CUDA非确定性限制因果解释；作者课程生成历史未知。本轮建议暂缓最终五Seed，后续消融仅作建议。

## 阅读顺序与代码

1. [训练前登记](reports/PREREGISTRATION.md)：Seed、划分、超参数、输出和唯一科学因素。
2. [完整中文诊断报告](reports/CMERC_MINIMAL_ABLATION_RESULTS.md)：全部指标、Loss、曲线、事实与假设。
3. [正式运行登记](configs/registry.json)与各run的`run_manifest.json`：完整argv、实际解析参数及原始服务器路径。
4. [比较CSV](results/comparison.csv)、[运行CSV](results/per_run.csv)、[汇总JSON](results/metrics_summary.json)、[训练/验证轨迹](results/epoch_curves.csv)、[分项Loss](results/epoch_loss_curves.csv)。
5. [最终独立CPU验收](provenance/final_verification.json)：3项完成、逐轮覆盖/损失、唯一最终测试及6个Checkpoint的状态与哈希；1154个历史常规文件未变。

`code_xp04/`是与XP04逐字节匹配的12个训练源文件副本；MELD只通过参数关闭加权CC。`code_source_order/`仅在`train_cm.py`新增固定源顺序采样，独立差异见[source_order.patch](provenance/source_order.patch)。训练池、课程成员、continuous RNG、ATen GRU、机制系数及验证选模均按登记保持，原始Loss数学未改。

`scripts/observe_training.py`外部记录既有前向输出，不增加训练前向或优化步骤。Baseline的五路移除只在唯一最终测试中进行，训练/验证CEL为null。[硬件配对边界](provenance/hardware_pairing.json)、[环境](provenance/environment.json)、[冻结文件](provenance/frozen_execution.json)及[数据SHA](provenance/data_inputs.json)单独保留。

## GitHub发布范围与验证边界

公开源码、作者已经公开的小型课程文件、配置、逐轮聚合指标、报告、PNG/SVG、预检和溯源JSON。数据特征与其绝对符号链接、6个best/latest权重、逐样本预测/首Batch数组、原始stdout/资源日志、逐Batch跟踪和锁文件保留在服务器。原记录中的路径及SHA不意味着对应二进制已上传。

[第06组发布入口](../../../experiment_records/06_minimal_ablation_validation/README.md)与[发布文件清单](../../../experiment_records/06_minimal_ablation_validation/publication_manifest.json)记录实际GitHub范围。原[artifact_manifest.json](provenance/artifact_manifest.json)覆盖训练完成时的服务器产物，原样保留；它不是公开副本的文件齐备清单。

纯GitHub副本可审查源码、参数、聚合指标和图表；完整`verify_results.py`、`summarize_minimal.py`及训练预检仍需要服务器保留的数据、权重、数组、原始跟踪和XP04原档案。已有完整验收在服务器完成，本次发布没有再次训练或测试。

安装Matplotlib后，在本目录可仅用已公开CSV重绘诊断图，此命令已在现有CPU绘图环境执行验证：

```bash
python scripts/plot_minimal.py
```

历史启动命令和登记含服务器绝对路径，执行器还会核对冻结SHA并拒绝覆盖已有输出。迁移或新实验应另行登记路径和环境，不能直接重跑历史队列。训练环境为Python 3.10.13、PyTorch 1.13.1+cu117、PyG 2.4.0，详细依赖和数据准备参见[仓库复现指南](../../../REPRODUCTION_GUIDE.md)。
