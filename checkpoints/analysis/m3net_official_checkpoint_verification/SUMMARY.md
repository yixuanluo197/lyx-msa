# M³Net 官方 IEMOCAP Checkpoint 性能验证

本轮只评估预先固定的作者发布权重 `Roberta_IEMOCAP3.pth.tar`，不训练、不模态删除、不计算六项诊断指标、不运行实验二完整错误分析、不评估其他两个候选。官方 CMERC 的结果直接复用原始预测与指标，未重跑或重新选择权重。

## 权重来源、结构及配置

- 权重路径：`/data2/yb/paper/cmerc_reproduction_20261013/checkpoints/baseline_candidates/m3net_official_20261025/Roberta_IEMOCAP3.pth.tar`。
- SHA-256：`fb939de1f88855fda46a9d491e5cce4400c88c7187f3b792bb40f61c4ed03faa`。
- 来源：[M³Net 作者 README](https://github.com/feiyuchen7/M3NET/tree/4d0fa88cd4f92c72b053527e877763f54e29ccb2#testing-and-checkpoints) 指向的 [官方公开 Dropbox](https://www.dropbox.com/sh/gd32s36v7l3c3u9/AACOipUURd7gEbEcdYSrmP-0a?dl=0)。沿用已有下载文件，无重复下载。
- CPU `torch.load(weights_only=True)` 加载为纯 OrderedDict；85 个张量，无缺失、额外或形状不兼容参数，`strict=True` 完整加载。
- 有效模型为两仓库一致的三模态 GRU + concat_DHT 超图结构：D_g=512、graph_hidden=512、num_L=3、num_K=4、dropout=0.5、norm=BN。复用冻结 CMERC 官方代码及既有两处 PyG 兼容补丁，未修改模型或增加新补丁。
- 为明确更换预训练权重，官方 train.py 原有 checkpoint 请求由只读加载器映射到上述唯一固定 M³Net 路径；清单同时记录 requested 与 actual_loaded_checkpoint。没有加载 CMERC 权重执行前向。
- 纯 state_dict 不包含实际训练 seed、epoch、选模方式或与发布 CMERC 权重的训练配对关系；不能从文件名、参数形状或当前测试成绩补造这些历史。

## 环境和测试协议

服务器 `ssh biggpu` / user22，宿主机 GPU 1（UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`），逻辑 cuda:0；GPU 4 禁用。Python 3.10.13、PyTorch 1.13.1+cu117、CUDA 11.7、cuDNN 8500。使用原 `/data2/yb/conda_envs/cmerc-repro` 环境。

完整文本/音频/视觉，batch=16，num_workers=2，不打乱测试对话，原 Padding/Mask 及 dialogue-major flatten；六类顺序 happy/sad/neutral/angry/excited/frustrated。cuDNN enabled=True、deterministic=True，评估 seed=1475，仅为评估种子。实际两批次 forward，model.eval() 和 no_grad，optimizer step=0，无模态删除。

## 实际分类结果

单位：Accuracy / F1 为百分数；CSV 保留精度。

| Dataset | Model | Accuracy (%) | Weighted F1 (%) | Macro F1 (%) | Samples |
|---|---|---:|---:|---:|---:|
| IEMOCAP | Official M³Net | 72.4584103512 | 72.6601311656 | 71.5532217841 | 1623 |
| IEMOCAP | Official CMERC | 71.9038817006 | 71.9837935825 | 71.3746631560 | 1623 |

CMERC − M³Net 的 Weighted F1 差为 **-0.6763375832 个百分点**。本次固定公开 M³Net 权重的实测 F1 更高；不得因此测试另两份候选、更换 checkpoint、调整概率或以测试分数择优。该差值描述两份发布权重，既不能归因于某个训练模块，也不能据此否定其严格受控实验中的效果。

## 样本对齐及质量验证

两模型全部 **1623 条**有效话语的 dataset、dialogue ID、utterance index、source utterance ID、真实标签、原文和测试顺序完全一致；类别概率字段顺序和标签映射一致。概率数值不要求两模型相等，而是各自有效并对应同一类别/话语。

已对可信 SHA 的原始测试特征重新核验标签、原文、ID 和话语顺序；Padding 排除。所有概率有限、在 [0,1]、各行和约等于 1，预测为 argmax；CSV/NPZ 逐值一致。Accuracy、六类 Precision/Recall/F1、Weighted/Macro F1、Confusion Matrix 均由独立 Counter 统计复算通过。参数和 BN buffers 推理前后哈希不变；原 checkpoint/特征哈希不变；此前 291 个实验文件 SHA 不变。详见 `verification.json`。

## 可比性结论与下一步边界

**A. 公开权重行为对照可行。** 结构、特征、测试样本、batch/Padding/Mask、cuDNN 及有效评估路径已核验一致，可基于这两份固定公开权重开展后续实验二的描述性逐样本行为对比。本轮仅保存准备数据，没有执行纠错类型统计。

**B. 严格配对的模块效果对照尚未确认。** 两份公开权重的训练 seed、轮次、checkpoint 选择准则与训练配对关系未知，无法把预测差异归因于 CMERC 的课程学习、对比学习或置信度约束。公开训练脚本存在测试集选模风险，但不能据此断言这两份发布权重实际采用了该方式；本轮未用测试集选模或调参。

细项证据与已核实差异见 `pairing_feasibility.md` / `protocol_audit/`，历史来源审计保持原样。仅限 IEMOCAP，不据此补造 MELD Baseline。

## 文件与复现

`evaluation_results.csv` 为两模型性能表，`classification_report.csv` / `confusion_matrix.csv` 为本次 M³Net 全类别结果，`all_predictions.csv` / `predictions.npz` 为全部话语原始概率。`checkpoint_audit.json` 为实际载入权重参数清单，`run_manifest.json` 与 `evaluation.log` 记录命令和真实执行，原始观察产物完整保存在 `inference/`。

运行入口 `run_evaluation.sh` 固定上述唯一权重和健康卡，已完成产物不需重复推理，脚本拒绝覆盖完成的运行。CPU 独立复算：`python scripts/summarize_m3net_verification.py`。运行脚本及其源码副本不改变已有模型文件、标签或预测。
