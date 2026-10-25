# 作者公开 checkpoint 的官方完整输入评估协议审计

本说明审计本地固定官方源码、兼容执行副本和评估包装方式，不提供本次实测分数。历史日志、已有报告和论文分数均不能代替本次加载作者权重后的实际推理结果。本轮仅执行完整文本、音频、视觉输入的测试推理，不训练、不做模态或上下文移除，不重新计算 ECE、CEL 或其他历史可靠性诊断。

## 1. 固定源码及逐字节核对

官方固定 commit：`815c7c060914f5b55db816c0ba3662b11d08552f`。

审计使用本地 `git show <commit>:<文件>` 读取 blob，并与 `experiments/official_readme_reproduction/official_code/` 的实际文件 bytes 比较。以下六个文件全部逐字节一致：

| 文件 | 官方 blob 的 SHA256 |
|---|---|
| `README.md` | `5b02de406be3b0ec68f2ea3de65f3fc2a1b059cb091c98b1b02f15d9a8f4000b` |
| `train.py` | `35a18a2bf33d8e3e0d8459955b1f0ce525d4f28970f9af801fb27087bcf02ad7` |
| `model.py` | `ee84329b800e37a8ac652b58aa6f0fc3dee47cb2cf38feda14f5c0de26ec9a82` |
| `model_cm.py` | `f08a6673649029ca0889b89dcc1ce47cb8a88126e023d46ee1cb874f6945a216` |
| `model_hyper.py` | `c786622f150df0479c6774f16766e648034fd6914fdcd991d33c724969b93abe` |
| `dataloader.py` | `7ff80e8e2de73aa228effb4df56a5168c05068e8fa585a06d2e8b813e2aa4f95` |

`runtime_code/` 中的 `train.py`、`model.py`、`model_cm.py` 和 `dataloader.py` 同样与上述官方 blob 一致。其运行兼容改动集中在下面两处，已有 `provenance/compatibility.patch`、`patch_classification.json` 记录：

1. `model_hyper.py` 删除未使用的 `from torch_geometric.nn.pool.topk_pool import topk`。该符号已不在当前 PyG 2.4.0 的原位置，未经处理会阻断导入。
2. `HypergraphConv.py` 在 `flow='target_to_source'` 的第二次 `propagate` 中，将 `size=(num_edges, num_nodes)` 改为 `size=(num_nodes, num_edges)`，使 size 对应索引张量的节点轴和超边轴，避免现有 PyG 的形状检查异常。

因此本次应描述为“固定官方测试入口及模型，附已记录的运行兼容补丁”。原样官方实现在该环境中不能完整运行，不能宣称已经证明补丁执行结果与未打补丁的实现数值完全等价。两处补丁不增加或删除 checkpoint 参数。

## 2. 官方 testing 参数

以下命令来自固定官方 `README.md:112`、`:117`。权重路径由官方 `train.py:482–488` 按数据集确定。

```bash
python -u train.py --base-model GRU --dropout 0.5 --lr 0.0001 \
  --batch-size 16 --graph_type hyper --epochs 0 --graph_construct direct \
  --multi_modal --mm_fusion_mthd concat_DHT --modals avl \
  --Dataset IEMOCAP --norm BN --testing

python -u train.py --base-model GRU --dropout 0.4 --lr 0.0001 \
  --batch-size 16 --graph_type hyper --epochs 0 --graph_construct direct \
  --multi_modal --mm_fusion_mthd concat_DHT --modals avl \
  --Dataset MELD --norm BN --num_L 3 --num_K 3 --testing
```

两个 evaluation 命令均没有显式传 `--seed`，因此均使用 `train.py:312` 的默认值 **1475**。README 的 MELD 训练 seed 67137 不应被自动带入本次 evaluation。IEMOCAP 的 `num_L=3`、`num_K=4` 使用解析器默认值，MELD 显式使用 3、3。

`train.py:468–475` 创建测试 DataLoader 时使用 **batch size 16、num_workers 2**。本次保留此顺序、batch 划分和 padding，不按长度重排，不为降低显存擅自改 batch size。

官方 `train.py:18–25` 设置随机种子、`cudnn.benchmark=False`、`cudnn.deterministic=True`，没有禁用 cuDNN，也没有 XP validation 批次的 ATen RNN 包装。本次保留 CUDA 上默认 cuDNN RNN 后端；实际环境、设备及后端状态由运行 manifest 记录。

## 3. 测试集、padding 与展平对应关系

- `dataloader.py:19`、`:70` 按数据 pickle 内的 `testVid` 列表构造测试对话顺序。测试 DataLoader 没有指定 shuffle 或随机 sampler。
- `dataloader.py:55–57`、`:111–113` 将前七个特征/说话人张量按时间优先 padding，将 umask 和标签按 batch 优先 padding；最后一项保留 batch 的原始对话 ID 列表。
- `train.py:197` 根据每个 umask 的最后有效位置取得话语长度；`:206` 按 batch 对话顺序拼接每个 `label[j][:lengths[j]]`，排除 padding 标签。
- `model.py:458–464` 的 `simple_batch_graphify` 以相同顺序拼接有效话语；`model_hyper.py:255–270` 的 `reverse_features` 恢复各模态按对话、话语排列的对应关系。
- `model.py:796` 在 padding 后张量上执行双向 GRU，随后裁剪有效节点。batch 的最大长度会影响 padding 上下文，因此保持官方 batch16 和固定对话顺序是复现条件。
- `model.py:869` 返回原始 `F.log_softmax(..., 1)`。观察器应保留原输出，再从副本计算概率，不能先改预测或标签。

**官方返回 IDs 有局限：** `train.py:222` 的 `vids += data[-1]` 位于循环外，最终返回的 vids 只包含最后一个 batch。不得用该返回值推断全量话语 ID。本次观察器直接读取每个 batch 的 `data[-1]`、`lengths` 和已裁剪标签，并为每个对话保存从 0 开始的话语索引；逐条核对原数据的 `videoLabels` 和原始文本。

## 4. 最小完整输入评估包装

新增执行入口为 `scripts/verify_author_checkpoints.py`。其用途是实际运行官方 testing 分支，而不是用旧统计结果填表。包装策略如下：

1. 在独立进程中读取预检查清单，验证源码、数据和作者 checkpoint 的 SHA256，拒绝覆盖已有同数据集运行 manifest。
2. 将运行代码目录放在 Python 导入路径首位，将工作目录设为包含 `data/`、`checkpoints/` 的输入根目录，以绝对路径 `runpy.run_path(.../train.py, run_name='__main__')` 执行上述官方 argv。
3. 外层使用 `torch.no_grad()`。官方评估函数自身使用 `model.eval()`，二者同时保留。
4. 官方 testing 分支只加载指定作者权重；包装器要求严格 state_dict 加载并检查键和形状，拒绝非严格匹配。Adam 的 step 被设置为一旦调用即报错的保护，不执行更新。
5. AST 定位官方图评估函数中的 `loss_function(log_prob, label)` 语句，在该语句执行前使用只读 trace observer 获取当前 batch 的原始输出和已裁剪标签。观察器不改 frame locals、不替换输入、不额外调用模型。
6. 保留每条有效话语的对话 ID、位置、原始文本、标签、完整类别概率，并检查 umask 是连续有效前缀、原始长度及标签严格一致。最终核对拼接 labels/argmax 与官方函数返回值完全一致。
7. 检查 full forward 次数等于测试 batch 数、梯度关闭、模型处于 eval、参数及 buffers 未变化、数据和 checkpoint 文件未变化。实际结果、源码 hash、环境和审计证据由输出 manifest 保存。

`train.py:490` 调用的是同文件的 `train_or_eval_graph_model`，该入口导入 `model.Model`，不构造 `model_cm` 的辅助移除模型。`--epochs 0` 跳过 `:493` 起的训练循环；`--testing` 跳过 `:532` 起的训练记录写入。官方仍会构造 Adam 并读取训练/验证 dataset，这是初始化行为，不代表训练或使用训练样本推理。

不得调用旧 `evaluate_checkpoint.run_inference` 或 `experiments/official_readme_reproduction/scripts/evaluate_saved.py`：后者会启用 calibrate 并执行模态和上下文移除，不符合本次仅 full 的范围。本次不计算 ECE、CEL，不复算六路诊断，不重新训练。

## 5. 作者权重身份与选模协议的证据边界

实际 CPU 加载确认两个作者文件均为仅含张量的纯 `OrderedDict`，没有训练 args、seed、epoch、校准/对比/课程训练开关或 checkpoint 选择协议：

| 文件 | 张量键数 | SHA256 |
|---|---:|---|
| `checkpoints/IEMOCAP_checkpoint.pkl` | 85 | `f3098fed2734c18171c02a59d014ee27320d845bc1ad5dfa17162b8dcfb49925` |
| `checkpoints/MELD_checkpoint.pkl` | 83 | `1c72e29c173b749675885bdd4f477bd5d3702ab7cca7eb03cb145850bf0beba7` |

README 将这些文件作为 CMERC evaluation 的公开权重提供，因此可称“作者公开 checkpoint，按官方 CMERC testing 命令评估”。但 Baseline 与 CMERC 可能共享参数结构，不能仅凭文件名或结构还原其真实训练身份；真实训练 seed、轮次及选模元数据均为 **unknown**。

固定官方 `train.py:507–512` 在训练时按测试 loss、测试 F1 更新 best 记录，`train_cm.py` 也包含测试集选模逻辑。该实现存在测试集选模风险，应与独立验证集选模实验区分。不过作者公开权重没有足够元数据证明它具体采用了何种选模过程，不能把源码风险写成该权重已确认的训练事实。本次只对固定权重执行最终测试，不搜索 epoch、不按测试分数选择其他权重、不调整模型参数。

## 6. 本说明与本次结果的关系

本文件仅记录已核验的协议和可复现边界，不包含实时成绩。两数据集的实际运行状态、服务器、健康卡映射、命令、日志和数值以本批次执行记录及最终 manifest 为准；不能将此前 XP/V100 的成绩或先前作者权重诊断结果标记为本次新实测。
