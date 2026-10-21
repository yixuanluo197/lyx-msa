# 官方源码与 README 预审计

本报告审计官方仓库 `https://github.com/HITSZ-HLT/CMERC` 的固定 commit `815c7c060914f5b55db816c0ba3662b11d08552f`。文中源码行号均指本实验目录下未经修改的 `official_code/` 冻结副本；工程兼容补丁和运行副本应另外记录，不得改变本报告所指源码。

审计方式为读取该 commit 的实际源码，并只读核查本地输入 pickle 的划分与作者课程文件覆盖。未以已有 XP04 报告代替源码核验。正式运行状态以 `03_ENVIRONMENT_PREFLIGHT.md` 和 `04_FOUR_RUN_TRAINING_REPORT.md` 为准。

## 1. README 实际命令与配置

下表优先依据 README 实际命令。IEMOCAP Baseline 位于 `official_code/README.md:34–36`，完整 CMERC 位于 `49–54`；MELD 分别位于 `62–64` 和 `78–83`。Baseline 没有传入三个机制开关，源码默认均为 False（`official_code/train_cm.py:459–466`）。

| 配置 | IEMOCAP Baseline | IEMOCAP CMERC | MELD Baseline | MELD CMERC |
|---|---|---|---|---|
| Dataset | IEMOCAP | IEMOCAP | MELD | MELD |
| seed | 1475 | 1475 | 67137 | 67137 |
| epochs | 80 | 80 | 40 | 40 |
| batch_size | 16 | 16 | 16 | 16 |
| lr | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| dropout | 0.5 | 0.5 | 0.4 | 0.4 |
| base_model | GRU | GRU | GRU | GRU |
| graph_type / graph_construct | hyper / direct | hyper / direct | hyper / direct | hyper / direct |
| multi_modal / modals | True / avl | True / avl | True / avl | True / avl |
| mm_fusion_mthd / norm | concat_DHT / BN | concat_DHT / BN | concat_DHT / BN | concat_DHT / BN |
| num_L / num_K | 3 / 4 | 3 / 4 | 3 / 3 | 3 / 3 |
| 分类 Loss | 类别加权 NLL | 类别加权 NLL | 作者 FocalLoss | 作者 FocalLoss |
| calibrate / rank_coff | False / 0 | True / 0.005 | False / 0 | True / 0.002 |
| contrastlearning | False | True | False | True |
| mscl_coff / cscl_coff | 0 / 0 | 0.05 / 0.05 | 0 / 0 | 0.15 / 0.15 |
| courselearning | False | True | False | True |
| epoch_ratio / scheduler_steps | 不使用 | 0.15 / 1 | 不使用 | 0.4 / 1 |

README 未显式给出的设置通过源码补足，不能误写成 README 指定：Adam 的 weight_decay=`3e-5`（`train_cm.py:402,713`）；use_speaker=True、use_modal=False、av_using_lstm=False、use_residue=False（`440,446,448,432`）。超图内部使用 lamda=0.5、alpha=0.1、variant=True（`model_cm.py:660–662`），这与 parser 的通用 alpha 默认值不是同一位置。图隐藏维度为512（`train_cm.py:528`），IEMOCAP/MELD 的 D_g 分别为512/1024（`523`）。

README 环境要求：Python 3.10.13、PyTorch 1.13.1、torch_geometric 2.4.0、torch-scatter 2.1.0、torch-sparse 0.5.15、CUDA 11.7（`README.md:8–14`）。实际环境差异单列于环境预检报告。

## 2. RNG、编码器与 Batch 行为

### 2.1 RNG

`train_cm.py:37–44` 重置 Python、NumPy、Torch CPU/CUDA RNG，并设置 cuDNN benchmark=False、deterministic=True。模型初始化前调用一次（`532–533`），每个 train/valid/test 函数入口又调用一次（`192`；非图路径为 `125`）。因此官方流程不是 continuous RNG，每轮训练的随机流也不是自然延续前一轮。该实现没有禁用 cuDNN，不能将其记为 ATen-only RNN。

### 2.2 GRU 和归一化

文本编码器是两层双向 `nn.GRU`（`model_cm.py:626–632`），README 的 dropout 参数也传给 GRU。音视频默认仅作线性映射；只有启用 av_using_lstm 时才建立并使用对应 GRU（`616–625,774–785`）。IEMOCAP 文本先线性映射，MELD 跳过该映射后进入 GRU（`786–796`）。

BN 对 padded batch 展平后计算，包括 padding 行（`model_cm.py:730–734`）。遮蔽特征通过沿整个时间维求均值再 repeat 构造（`train_cm.py:235–241`），也未先根据 umask 排除 padding。Batch 组成变化因此可能影响均值遮蔽、BN 和训练随机过程；这只是机制层面的影响路径，尚不能据此认定 XP04 的 F1 差距来源。

### 2.3 实际划分与采样

主循环在两数据集上均使用 valid=0.0（`train_cm.py:745–754`），每个 epoch 重建数据集及 loader。验证 loader 为空，调用返回 NaN 指标（`364–365`）。

- IEMOCAP：训练 loader 未使用刚刚构建的 train_sampler，也未设置 shuffle，故按 dataset keys 顺序组成 batch（`84–99`）。
- MELD：训练 loader 使用 `SubsetRandomSampler`（`56–72`），但函数入口重复设种子，因此不能将其描述成每轮独立连续 RNG 随机采样。
- 两数据集测试 loader 都不 shuffle（`74–79,101–106`）。没有 capacity_safe batch 重组逻辑。

数据类直接使用 raw feature pickle 的 trainVid/testVid（`dataloader.py:10–19,61–70`）。本地输入只读核查如下：

| 数据集 | raw 训练对话 / 句子 | raw 测试对话 / 句子 | 对划分的核查 |
|---|---:|---:|---|
| IEMOCAP | 120 / 5810 | 31 / 1623 | 按 raw trainVid/testVid |
| MELD | 1152 / 11098 | 280 / 2610 | raw train 包含 RoBERTa trainIds 1038 个对话及 validIds 114 个对话 |

MELD 的 RoBERTa pickle 加载了 trainIds/testIds/validIds（`dataloader.py:66–68`），但 keys 实际仍用 raw.trainVid（`70`），因此官方训练合并了这些输入中的 train+valid，并未保留这114个对话作验证。核查中 raw train 与 RoBERTa testIds 的交集为0。这与使用验证集选模的实验协议有实质差异。

## 3. 分类和校准 Loss

### 3.1 分类 Loss

IEMOCAP 使用固定六类逆频率权重的 `nn.NLLLoss`（`train_cm.py:690–704`）。MELD 使用作者 `FocalLoss`（`698–699`），默认 gamma=2.5、alpha=1（`model_cm.py:15–21`）。作者实现对输入再做 log_softmax，以 `label_onehot * log_p` 构造 pt，最后对全部元素求均值（`40–48`）；不能替换为常见标准 focal 公式后仍称官方数学实现。

### 3.2 CC

三种模态遮蔽和两种上下文遮蔽的辅助预测在 no_grad 中执行（`train_cm.py:232–250`）。针对真实类别累加遮蔽 log-prob 高于主模型 log-prob 的正差，模态项和上下文项相加（`261–282`），未按句子数或 batch 数归一化。总 Loss 根据 calibrate 加入 rank_coff 倍的该项（`353–354`）。

上下文遮蔽通过删除同说话者或异说话者的 GNN 边实现（`model_hyper.py:310–340`）。不能把它概括为删除所有上下文结构：同一个 HyperGCN 还会构造并使用超图分支（`153–175`）。

### 3.3 MSCL、CSCL

三种模态及两种上下文二值伪标签由主预测/遮蔽预测的正确性与真实类置信度比较生成（`train_cm.py:286–341`）。类别权重为 batch 内伪类别总数/该类数（`21–34`）。MSCL、CSCL 分别使用三组、两组对比项（`345–346`；`clloss.py:271–355`），温度默认0.07（`clloss.py:195–200`），根据归一化特征的相似度计算（`234–267`）。非有限总对比 Loss 会静默替换成0（`314–315,353–354`）。

`ft_scl=hidden` 虽放在 no_grad 区域内（`train_cm.py:233–234`），但这是对象引用赋值，并没有 detach；不能仅依据该位置声称 HSCL 没有梯度。

三个辅助模型在每次整段训练或评估函数入口复制主模型参数并设为 eval（`185–190`），没有在训练段内每个 optimizer.step 后重新同步。主模型训练期间更新，辅助模型在这一训练段内保持入口时状态。

最终训练损失为分类项，加开关控制的 CC、MSCL、CSCL（`train_cm.py:348–358`）。官方 `losses.append(loss.item())` 位于附加项之前（`351`），所以打印的 train_loss/test_loss 是分类 Loss 的 batch 均值（`376`），并非总训练目标；官方不保存四项 Loss 的逐 epoch 分解。

## 4. 课程来源、README 错误与覆盖

本次选择 README Quick Start 允许使用的作者提供课程文件（`README.md:86–87`）。固定 commit 文件内容与本地训练对话集合核查如下：

| 文件 | 对话数 | 训练缺失 / 额外 / 测试交集 | SHA256 |
|---|---:|---:|---|
| `official_code/caldiff/IEMOCAP_diff.pkl` | 120 | 0 / 0 / 0 | `db4c7b96f169948a3615a62b29d2a793bd64cccd8a0387b1de9d7ceab348a150` |
| `official_code/caldiff/MELD_diff.pkl` | 1152 | 0 / 0 / 0 | `db005ceb0e4847caffc8571eaa94ab7c5b4a310abd615b12c995c9a22ff060e0` |

生成 checkpoint、训练种子、生成批大小及历史未知。文件 keys 不包含测试对话，只能证明其覆盖集合；不能证明生成模型未使用测试集选模，亦不能据此排除间接泄漏。

README 课程步骤存在两处已由代码证实的问题：

1. MELD 第2步调用 `train_cm.py --ckpt_path=...`（`README.md:69–72`），但 `train_cm.py` parser 没有 ckpt_path（`457–468`），会拒绝该参数。真正的难度工具 `caldiff/cal_diff.py` 接受该参数（`445`），并加载指定 baseline state_dict（`669`）。
2. IEMOCAP 难度命令仍指定 batch_size=16（`README.md:41–44`）。工具将整 batch 的 rank_loss 写给 `data[-1][0]` 一个对话键（`caldiff/cal_diff.py:358–360`），每 batch只产生一个键。按 README 对120个训练对话最多写8键；若相同方式用于1152个 MELD 对话，最多写72键。不能用该命令解释上游完整120/1152键文件的生成历史，也不能静默改为 batch1后宣称完全相同协议。

课程实现按难度升序截取 `int(ratio*N)` 个对话，再恢复原始 keys 顺序（`dataloader.py:23–36,73–86`）。比例为 `min(1, round(epoch_ratio*int(e/scheduler_steps+1),2))`（`train_cm.py:737–742`）：IEMOCAP依次0.15、0.30、0.45、0.60、0.75、0.90、1；MELD依次0.40、0.80、1。测试集构建未传入 epoch_ratio，不执行课程筛选。

README 完整训练命令没有加载预训练模型参数，`train_cm.py` 正常训练路径也没有加载 checkpoint；作者文件在此提供的是难度排序。本次四个训练均应从头初始化，不能把 Quick Start 中提到 checkpoint 的自然语言理解成可任意加载历史训练权重。

## 5. 测试选模、Checkpoint 与日志缺口

每个 epoch 训练后都评估测试集（`train_cm.py:758–760`），W-F1先四舍五入到百分比两位（`378`），再以严格大于比较更新 best_fscore（`771–774`）。本实验对应结果必须标记 `official_test_selection`，不能称为无测试泄漏的独立评估，亦不能直接与 XP04 验证集选模结果作同协议因果比较。

同时，best_label/best_pred 还会被更小 test_loss 更新（`768–769`）。因此每10轮打印的 classification_report/confusion_matrix（`783–786`）未必对应最高 W-F1 的同一个 epoch。应由明确保存的 epoch checkpoint与独立记录核验对应关系。

`train_cm.py` 全文没有 `torch.save`，也没有保存最终模型、optimizer、完整 RNG 状态或逐 epoch结构化结果。`path='./saved/IEMOCAP/'`（`384`）没有构成实际保存逻辑。为了满足本任务的 best/final checkpoint及 Loss分项交付，需要独立、可审计的观测/保存补丁，并记录其不改变训练前向、损失或随机调用的验证证据。

README 的评估入口是另一文件 `train.py`（`README.md:112,117`），其导入 `model`（`train.py:10`），并加载作者 `checkpoints/*_checkpoint.pkl`（`484–488`）。它不是四个新 run 自然产生的 checkpoint。`train_cm.py --testing` 另有参数签名不匹配的调用（`734`），不能直接假定其可用于独立评估。

## 6. 运行阻碍与证据等级

**已经证实的事实：** 原始副本预检在 `official_code/model_hyper.py:16` 的 PyG 内部 `topk` 导入失败；运行证据见 `logs/preflight_original_IEMOCAP_baseline.log`。该原始尝试尚未进入正式训练。后续正式运行采用独立兼容补丁及观测，不能称为未经修改运行；实时状态见报告04及 state/status.json。

**已由后续预检证实的接口错误：** `HypergraphConv.py:147–151` 第一轮传播使用 size=(num_nodes,num_edges)，切换 flow=target_to_source 后第二轮传入 size=(num_edges,num_nodes)。仅移除 topk 导入后的真实首 Batch 在第二传播处报 `Encountered tensor with size 2607 ... expected size 917`，见 `logs/preflight_importfix_IEMOCAP_baseline.log`。独立运行副本按PyG索引轴契约修复 size，并完成 CPU shape/backward 和四组首Batch检查；完整diff与证据见报告03。

**固定容量事实及限制：** `model_hyper.py:118–119` 的 hyperedge_weight 长1000、EW_weight长5200；每批超边为总句数+3×对话数。当前 IEMOCAP 原始顺序 batch16训练最大869句/917超边，测试最大861句/909超边，未超过1000。EW_weight实际按节点索引取值（`HypergraphConv.py:142`）；不能仅凭关联数超过5200认定越界。课程子集改变批组成，仍需完整预检，而非假定所有课程阶段安全。

**有证据但未隔离变量的推断：** 重复重置 RNG、合并验证集参与训练、官方测试选模、顺序/随机 batch差异，以及 padding参与BN/遮蔽均值，都可能改变优化轨迹或报告分数。它们目前不能被指定为 XP04 F1差距的单一原因。

**需要后续实验的假设：** 任一具体协议因素是否足以恢复论文提升、CC/HSCL是否改善泛化、作者课程生成方式是否影响性能，均须在本次真实运行及必要的最小变量对照后判断。本文件只记录源码与预检证据；四组实训终态、实测结果及结论分别见报告04–06。
