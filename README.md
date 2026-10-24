# lyx-msa

CMERC 多模态对话情绪识别代码、复现工程与时间因果变体，供科研复现和代码审查使用。仓库根目录对应本地的 `CMERC/`，克隆后无需再进入同名子目录。

基于 [HITSZ-HLT/CMERC](https://github.com/HITSZ-HLT/CMERC)，上游版本为 `815c7c060914f5b55db816c0ba3662b11d08552f`，对应 ACM MM 2024 论文 *Multimodal Emotion Recognition Calibration in Conversations*。本仓库是包含本地修改的研究版本；原作者说明、下载链接和论文引用完整保存在 [原始 README](docs/UPSTREAM_README.md)。

## 阅读入口

- [实验档案总览](experiment_records/README.md)：历史复现、XP04 五 seed 主线，以及本次官方 README 单 Seed 对照的代码版本、配置、逐轮指标和最终结果。
- [官方 README 四组对照实验](experiment_records/05_official_readme_reproduction/README.md)：IEMOCAP seed1475、MELD seed67137 的 Baseline/CMERC 实测、源码协议审计和与 XP04 的对照；保留官方测试集选模。
- [CMERC 最小消融根因验证](experiment_records/06_minimal_ablation_validation/README.md)：MELD去CC及IEMOCAP固定源顺序Batch的3项正式实验；验证集选模、最终测试一次，区分相对负增量缩小与绝对性能提升。
- [CMERC 学习率诊断](experiment_records/07_learning_rate_diagnosis/README.md)：固定XP04完整模型，仅比较0.00005、0.0002与历史0.0001；4项训练均完成，只报告训练/验证，零测试前向。
- [MELD 学习率第二Seed确认](experiment_records/08_lr_second_seed_confirmation/GITHUB_DELIVERY.md)：seed2024的唯一40轮训练已完成；最佳验证F1下降0.55个百分点、末段均值提高0.811个百分点，按预登记规则未确认主指标收益。
- [MELD Batch Size诊断：第09组](experiment_records/09_batch_size_diagnosis/GITHUB_DELIVERY.md)：Batch8、seed2024、LR0.0001的唯一训练已登记；CPU预检通过，GPU资源排队，尚无新增验证F1。
- [高置信度错误诊断：实验一至五](checkpoints/analysis/high_confidence_errors/experiments_2_to_5/README.md)：固定 validation checkpoint 的全量预测、模态均值替换、逐样本对比及统计；提供 [Word 报告](checkpoints/analysis/high_confidence_errors/experiments_2_to_5/CMERC_高置信度错误诊断报告_完整版.docx)、[Excel 全量结果](checkpoints/analysis/high_confidence_errors/experiments_2_to_5/CMERC_高置信度错误诊断结果_完整版.xlsx) 和 [统计摘要](checkpoints/analysis/high_confidence_errors/experiments_2_to_5/SUMMARY.md)。原始实验一 Word 母版未找到，新报告明确采用原 Markdown 转录。
- [实验六：概率校准分析](checkpoints/analysis/high_confidence_errors/experiment_6_calibration/README.md)：沿用前五项固定预测，提供 ECE、MCE、Brier、NLL、十区间明细与两个数据集的可靠性图；[实验一至六 Word 报告](checkpoints/analysis/high_confidence_errors/experiment_6_calibration/CMERC_高置信度错误诊断报告_实验一至六.docx)完整保留前五项正文，另附[实验六 Excel](checkpoints/analysis/high_confidence_errors/experiment_6_calibration/CMERC_实验六_概率校准结果.xlsx)及[结果分析](checkpoints/analysis/high_confidence_errors/experiment_6_calibration/SUMMARY.md)。
- [GPT / 人工代码审查指南](docs/GPT_REVIEW_GUIDE.md)：建议阅读顺序、重点文件和可复制的审查提示词。
- [复现运行指南](REPRODUCTION_GUIDE.md)：环境、数据准备、训练、评测和断点恢复。
- [时间因果变体说明](CAUSAL_README.md)：历史与当前话语的可见性约束、实现范围和验证边界。
- [原代码审计](reports/01_CODE_AUDIT.md)与[复现结果对比](reports/07_REPRODUCTION_COMPARISON.md)：原实现的协议差异与已有实验记录。

## 目录与主要入口

| 路径 | 用途 |
| --- | --- |
| `train_cm.py` | 主训练入口、损失组合、选模协议与完整 checkpoint 恢复 |
| `model_cm.py` | 主模型以及模态、上下文移除分支 |
| `model_hyper.py`、`HypergraphConv.py`、`model_GCN.py` | 图与超图模块 |
| `causal_graph.py` | 时间因果图构造与消息传递 |
| `dataloader.py` | 数据划分、课程难度、历史均值与 checkpoint 协议检查 |
| `clloss.py`、`evaluation_metrics.py` | 对比损失与评估指标 |
| `caldiff/` | 对话难度计算入口及上游提供的两份小型难度文件 |
| `scripts/` | 环境、训练队列、Smoke Test、独立评测和结果汇总 |
| `tests/` | 已有回归测试与合成输入因果性检查 |
| `configs/` | 历史复现环境的依赖快照 |
| `reports/` | 带日期的环境、数据、训练与评测报告 |
| `model.py`、`train.py`、`high_fre_conv.py` | 保留的上游模型与运行入口 |

## 获取与运行

```bash
git clone https://github.com/yixuanluo197/lyx-msa.git
cd lyx-msa
```

上游使用 Python 3.10、PyTorch 1.13.1 与 CUDA 11.7，具体版本及安装步骤见 [原始 README](docs/UPSTREAM_README.md) 和 [环境报告](reports/02_ENVIRONMENT.md)。`configs/requirements-lock.txt` 是既有环境快照，PyTorch / PyG 扩展需要匹配对应的 wheel 源。

环境脚本包含原实验服务器路径。使用前按自己的环境设置 `CMERC_ENV_PREFIX`、`CONDA_SH` 和 `CMERC_PYTHON`；GPU 编号同样需要按所在机器指定。例如：

```bash
CMERC_ENV_PREFIX="$HOME/conda-envs/cmerc-repro" \
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh" \
  bash scripts/bootstrap_environment.sh

CMERC_ENV_PREFIX="$HOME/conda-envs/cmerc-repro" \
  bash scripts/with_cmerc_env.sh python -m pytest -q tests
```

数据准备后，使用现有训练包装器：

```bash
CUDA_VISIBLE_DEVICES=0 \
CMERC_ENV_PREFIX="$HOME/conda-envs/cmerc-repro" \
CMERC_PYTHON="$HOME/conda-envs/cmerc-repro/bin/python" \
  bash scripts/with_cmerc_env.sh bash scripts/run_training.sh \
  IEMOCAP cmerc validation --num-workers 0
```

以上是可调整路径的使用示例；此次公开发布没有重新安装环境或启动训练。完整命令与协议说明见 [复现运行指南](REPRODUCTION_GUIDE.md)。

## 数据、权重与结果

公开仓库包含源码、测试、脚本、依赖快照和复现报告。[实验档案](experiment_records/README.md)另保留经过筛选的运行配置、逐轮指标、聚合评测与结果图。本次额外公开 `checkpoints/analysis/high_confidence_errors/` 下实验一至六的逐话语预测、CSV、Word、Excel 和来源清单，便于核验全部诊断结果。数据特征、模型 checkpoint 和原始训练日志仍保留在实验服务器；`.gitignore` 仅为上述诊断目录设置例外，继续排除其他本地运行产物。

从 [原始 README](docs/UPSTREAM_README.md) 的作者链接获取特征和官方 checkpoint，放入根目录的 `data/`、`checkpoints/`。文件清单与校验值见 [数据审计](reports/03_DATA_AUDIT.md)。`caldiff/IEMOCAP_diff.pkl` 和 `caldiff/MELD_diff.pkl` 是上游已跟踪的小型课程难度文件，继续保留。

`reports/` 记录的是既有实验。部分报告链接到未上传的本地运行产物；这些报告不等于已将原始实验数据公开，也不意味着克隆后已有模型权重。

## 研究与验证边界

- 默认路径是非因果模型；`--causal` 显式启用时间因果变体。
- `official_test` 保留上游测试集选模行为；`validation` 使用验证集选模。比较实验时必须区分这两套协议。
- 因果变体的现有检查使用 CPU 合成输入；尚未训练该变体，不能据此声称性能提升。
- 因果约束从已提取的逐话语特征开始，原始特征提取链的端到端因果性尚未核实。
- 历史环境与 GPU 复现记录见 `reports/`；当前因果测试使用的环境见 `CAUSAL_README.md`。

上游该版本未附独立 LICENSE 文件；本仓库保留作者归属与引用，不额外声明新的开源许可证。

## 后续更新

本地工作目录仍为 `/data2/yb/paper/lyx-msa/CMERC`。上传后可在该目录查看和推送后续修改：

```bash
git status
git add <本次要上传的代码或文档>
git commit
git push
```

提交前检查暂存文件，确保没有加入数据、权重或凭据。提交说明遵循项目的 Lore 协议。
