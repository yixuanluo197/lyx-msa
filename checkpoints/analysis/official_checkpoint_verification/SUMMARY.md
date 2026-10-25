# CMERC 作者公开 Checkpoint 性能验证

## 实验目的与结论

重新加载作者发布的两份权重，实际执行官方 train.py --testing --epochs 0，验证完整三模态测试集分类性能。本次不重新训练、不执行信息移除、不计算ECE/MCE/Brier/NLL/CEL，不重复此前六项诊断。历史成绩没有代替本次推理。

两个数据集均达到预先规定的 |实测W-F1−论文W-F1| ≤0.5个百分点标准，实测W-F1四舍五入两位后与论文相同。

| Dataset | Model | Accuracy (%) | Weighted F1 (%) | Macro F1 (%) | Valid samples | Paper W-F1 (%) | Gap (百分点) | ≤0.5点 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| IEMOCAP | Official CMERC | 71.9038817006 | 71.9837935825 | 71.3746631560 | 1623 | 71.98 | +0.0037935825 | 是 |
| MELD | Official CMERC | 67.7777777778 | 66.8483922007 | 51.0770228192 | 2610 | 66.85 | -0.0016077993 | 是 |

CSV和JSON保留完整浮点精度；以上是本次实测值，不是复制论文数字。Gap=实测−论文，单位百分点。

## 权重来源和结构

官方README下载入口：https://drive.google.com/drive/folders/1ty0XonRQG-DOyNLASHd9bkothMlBwjI8?usp=sharing。优先使用服务器已有文件，没有重复下载。文件大小和SHA与2026-10-13下载审计一致，且本地与biggpu副本逐字节哈希一致。

| Dataset | 文件 | 字节 | SHA-256 | state_dict张量键数 |
| --- | --- | ---: | --- | ---: |
| IEMOCAP | `/data2/yb/paper/cmerc_reproduction_20261013/checkpoints/IEMOCAP_checkpoint.pkl` | 26278739 | `f3098fed2734c18171c02a59d014ee27320d845bc1ad5dfa17162b8dcfb49925` | 85 |
| MELD | `/data2/yb/paper/cmerc_reproduction_20261013/checkpoints/MELD_checkpoint.pkl` | 56500755 | `1c72e29c173b749675885bdd4f477bd5d3702ab7cca7eb03cb145850bf0beba7` | 83 |

权重均为纯OrderedDict，安全weights_only读取，严格strict=True加载；缺失键、额外键、形状不一致均为0。完整参数名称、shape、dtype保存在checkpoint_inventory.json。没有epoch、训练seed、优化器、训练flags或选模方式元数据，均标记未知，不从缺失字段推断训练历史。作者README把这两个文件用于CMERC评估，本次沿用该来源称谓，不以参数形状独立证明训练机制。

## 模型代码与配置

固定官方commit：`815c7c060914f5b55db816c0ba3662b11d08552f`。code_official保留14个逐字节核对git blob的原文件；code_runtime仅有compatibility.patch记录的两处既有PyG兼容修改：删除未用且当前不可导入的topk；修正HypergraphConv反向propagate的size轴。train.py、model.py、model_cm.py、dataloader.py与官方原文件一致。不能把此运行称为完全无补丁的官方源码执行。

| 设置 | IEMOCAP | MELD |
| --- | --- | --- |
| 结构 | GRU / hyper / direct / concat_DHT / BN | 同左 |
| 输入 | avl完整三模态 | avl完整三模态 |
| 文本 / 音频 / 视觉维数 | 1024 / 1582 / 342 | 1024 / 300 / 342 |
| 类别数 / speaker数 | 6 / 2 | 7 / 9 |
| dropout / num_L / num_K | 0.5 / 3 / 4 | 0.4 / 3 / 3 |
| batch / workers / shuffle | 16 / 2 / False | 同左 |
| eval seed | 官方命令未传，默认1475 | 官方命令未传，默认1475 |
| RNN后端 | 默认cuDNN，deterministic=True | 同左 |
| test对话 / 有效话语 / forward | 31 / 1623 / 2 | 280 / 2610 / 18 |

MELD训练命令的seed67137不等于本次README评估命令的默认seed。作者权重实际训练seed仍未知。完整解析参数、环境版本和每batch形状见run_manifest.json。

## 数据、话语顺序和指标

使用官方4个真实特征文件，SHA见preflight.json。IEMOCAP类别顺序happy/sad/neutral/angry/excited/frustrated；MELD为neutral/surprise/fear/sadness/joy/disgust/anger。测试DataLoader保持原pickle testVid顺序，batch16、不排序、不打乱、不丢最后批。
官方model输出为log_softmax。官方函数按对话顺序拼接label[j][:length]，observer在同位置只读记录log_prob、原始ID和长度，检查每条标签与原pickle对应，排除padding，概率从原log_prob副本exp得到。没有改变输入张量或模型输出。
官方函数返回的vids仅累积最后batch，是原代码的ID记录缺口；本次observer逐batch记录完整ID，没有将最后batch ID重复映射。
Accuracy为正确话语比例；Weighted F1按真实类别support加权；Macro F1对全部固定类别取均值。指标用本次原始argmax结果由sklearn计算，并由独立Counter实现逐类复算；与官方函数两位小数返回结果一致。

## IEMOCAP逐类别结果

| 情绪 | Precision (%) | Recall (%) | F1 (%) | Support |
| --- | ---: | ---: | ---: | ---: |
| happy | 57.861635 | 63.888889 | 60.726073 | 144 |
| sad | 82.572614 | 81.224490 | 81.893004 | 245 |
| neutral | 69.704433 | 73.697917 | 71.645570 | 384 |
| angry | 72.151899 | 67.058824 | 69.512195 | 170 |
| excited | 79.787234 | 75.250836 | 77.452668 | 299 |
| frustrated | 67.374005 | 66.666667 | 67.018470 | 381 |

完整混淆矩阵见 `IEMOCAP_confusion_matrix.csv`，行是真实类别、列是预测类别。全部话语、标签、原文及各类概率见 `IEMOCAP_predictions.csv`，原log_prob另存NPZ。

## MELD逐类别结果

| 情绪 | Precision (%) | Recall (%) | F1 (%) | Support |
| --- | ---: | ---: | ---: | ---: |
| neutral | 77.035950 | 83.598726 | 80.183276 | 1256 |
| surprise | 60.000000 | 60.854093 | 60.424028 | 281 |
| fear | 32.258065 | 20.000000 | 24.691358 | 50 |
| sadness | 53.125000 | 32.692308 | 40.476190 | 208 |
| joy | 63.529412 | 67.164179 | 65.296252 | 402 |
| disgust | 33.870968 | 30.882353 | 32.307692 | 68 |
| anger | 56.645570 | 51.884058 | 54.160363 | 345 |

完整混淆矩阵见 `MELD_confusion_matrix.csv`，行是真实类别、列是预测类别。全部话语、标签、原文及各类概率见 `MELD_predictions.csv`，原log_prob另存NPZ。

## 执行证据和风险边界

- 实际服务器biggpu/user22，宿主健康GPU1，进程逻辑cuda:0；GPU UUID为GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049。禁用宿主GPU4，未使用。两数据集顺序执行，进程PID、启动时间和命令均已记录，完成后本次GPU进程退出。
- 官方testing分支实际运行，外层torch.no_grad，模型eval；仅创建官方原有Adam对象，不调用step，设置了禁止更新保护。前后模型参数/BN缓冲区、checkpoint和数据SHA一致，所有参数梯度为空。
- 最初启动因PyTorch1.13的UUID可见性解析返回0设备而在模型构建前失败，尚无模型forward。已用最小硬件检查确定原因，改用CUDA_DEVICE_ORDER=PCI_BUS_ID和CUDA_VISIBLE_DEVICES=1，并用nvidia-smi进程PID→GPU UUID独立核对。失败日志与原脚本保存在attempts/attempt1_uuid；这不是按测试分数选择配置。
- 原train.py训练路径存在测试集逐轮选模风险；作者纯权重不含真实训练和选模历史，无法由此证明其独立选模过程。本次只有单个已发布文件的固定测试，不做任何测试选模。
- 论文报告五次运行均值，而作者公开的是单份状态；本次证明公开权重在当前兼容环境下达到接近论文的分类数值，不重建作者五次训练历史，也不验证其全部校准指标。
- 本次两个W-F1差距均处于论文两位小数的报告精度范围，未触发>0.5点的额外差距排查。没有搜索batch、调权重、改标签或更换checkpoint以追分。
- 前六项诊断使用本地validation权重，本次只使用作者pkl。原94份诊断文件哈希保持不变，结果不混用。

## 可复现命令与文件

实际持久启动脚本为run_evaluation.sh；其中两个dataset子进程调用source_code/verify_author_checkpoints.py，再执行code_runtime/train.py原始testing入口。evaluation.log保留官方argv、完整Namespace、终端指标和精确实测输出。
本目录保存eval结果CSV、各类报告、混淆矩阵、预测CSV/NPZ、checkpoint_inventory.json、run_manifest.json、preflight.json、protocol_audit.md、verification.json和artifact_manifest.json。
复跑请新建独立输出目录、复制固定代码/预检清单并调整命令路径；脚本拒绝覆盖已有同数据集结果。只读重算本次文件可运行：

```bash
/home/yangbin/miniconda3/bin/python checkpoints/analysis/official_checkpoint_verification/source_code/summarize_results.py
```

该重算命令不运行模型、不加载checkpoint，不生成六项诊断；仅核验已保存分类预测并重写本目录汇总。
