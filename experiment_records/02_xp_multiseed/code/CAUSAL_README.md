# CMERC 时间因果版

本次实现沿用之前 SDT 的定义：位置 `t` 只能读取同一对话的 `1..t`，包括当前完整话语及其三模态特征、speaker。模型不读取未来话语，也不将标签作为前向输入。这是时间信息可见性约束，不是因果效应识别。

## 实现方案与验收

保留默认非因果路径；使用 `--causal` 显式启用因果版，不覆盖原 SDT 因果代码或已有 CMERC 结果。

1. 主模型与模态／上下文移除分支使用同一因果配置。
2. 双向 GRU/LSTM 改为单向编码，维持输出宽度；Transformer 加上未来和 padding mask。
3. 文本统一使用逐话语特征维 LayerNorm，避免 BN 与跨时间 LN2 的未来统计。
4. 同模态超边改为历史前缀收集、只向当前节点投递；跨模态超边限于同一话语。不能把前缀超边回传所有历史成员。
5. 图边只允许历史源节点向当前目标节点传递；同一时刻跨模态边保留。图归一化只使用当前目标的入边，避免未来出度改变历史输出。
6. 模态移除的整段均值改为有效历史前缀均值；校准与对比损失组成保留。课程学习难度文件、checkpoint 与输出目录区分因果／原版协议。
7. 只用 CPU 合成输入检查未来扰动、前缀截断、padding、批次独立性、梯度与参数可训练性。此次不启动模型训练、真实数据评测或性能实验。

不采用整段模型逐前缀重跑：虽能隔离未来，但重复计算代价更大；不采用只删未来图边：双向编码、超边和归一化仍会泄漏。

## 范围与边界

- 支持论文使用的 `graph_type=hyper`、`mm_fusion_mthd=concat_DHT`、`modals=avl`，基础编码器为 GRU/LSTM/Transformer，非空对话与右侧 padding。
- 因果版新增有向超图层及共享类型权重，不再使用原图的固定容量、批内位置权重。它是原方法的时间因果变体，参数结构与原官方 checkpoint 不兼容，不能把原权重直接当作因果模型或完整续训。
- `Model_Intra` 继续表示移除同 speaker 时序图边；`Model_Inter` 表示移除异 speaker 时序图边，沿用原论文上下文移除口径。超图分支在三种模型中均保留，但均满足因果约束。
- 离线训练仍可使用所有训练标签、跨样本对比损失及整段训练对话难度排序；这些不作为单次预测的未来输入，不能据此宣称在线学习。
- 保证限于已经提取、按话语对齐的输入特征。原始文本／音频／视觉提取链是否使用未来片段或整段统计尚未核实；本次不宣称原始流端到端因果性。
- 当前未训练因果版，不能声称效果提升或获得论文性能。

## 使用入口（仅示例，本次未执行实验）

Python 接口：`Model(..., causal=True)`，`Model_Intra`、`Model_Inter` 同样支持该参数。指定 `BN` 或 `LN2` 时，因果模型实际使用 `LN`；训练入口会打印并记录有效配置。

训练包装脚本已有参数可直接追加 `--causal`。例如后续需要训练因果 backbone 时：

```bash
bash scripts/run_training.sh IEMOCAP baseline validation --causal
```

该例结果放在 `runs/train_cm/baseline/IEMOCAP_validation_seed1475_causal/`，不覆盖原版 run。完整 CMERC 的课程学习必须先使用因果 backbone 的 checkpoint 生成因果难度文件：

```bash
python caldiff/cal_diff.py --causal --base-model GRU \
  --dropout 0.5 --batch-size 16 --graph_type hyper \
  --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT \
  --modals avl --Dataset IEMOCAP --norm LN --num_L 3 --num_K 4 \
  --seed 1475 \
  --ckpt_path runs/train_cm/baseline/IEMOCAP_validation_seed1475_causal/best.pt

bash scripts/run_training.sh IEMOCAP cmerc validation --causal
```

难度生成只写 `caldiff/IEMOCAP_causal_diff.pkl` 的因果元数据封装，不改 `IEMOCAP_diff.pkl`。MELD 使用对应数据集、`dropout=0.4`、`num_K=3`、`seed=67137`。未生成因果难度时，课程学习会明确报错，不退回原文件。

评测入口支持 `scripts/evaluate_checkpoint.py --causal`，并检查 checkpoint 协议。它仍按论文默认 GRU 配置构建模型；自定义 LSTM/Transformer、残差或其他结构配置须使用一致的 Python 构建参数，不能直接交给这个固定配置评测器。

## 验证与环境

新增 `tests/test_causal_model.py`、`test_causal_graph.py`、`test_causal_protocol.py`、`test_causal_losses.py`。这些仅使用随机／手工构造的 CPU 输入，不读取数据集或官方模型权重。图层参数更新检查使用合成损失做一次 SGD 更新，属于可训练性单元检查。

当前机器没有复现指南所列的 `/data2/yb/conda_envs/cmerc-repro` 环境。本次复用现有 `multimodalerc310`（PyTorch 2.2.2+cu121、PyG 2.4.0），并从已有 `msa_extract` 环境借用 pytest 9.0.2，没有安装依赖：

```bash
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c \
  'import sys; sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages"); import pytest; raise SystemExit(pytest.main(["-q", "tests"]))'
```

损失中的辅助张量改为跟随输入设备，保留原数学公式，使 CPU 检查可覆盖分类、校准和三模态／上下文对比损失。尚未验证 GPU 执行、原 PyTorch 1.13.1 环境、真实特征或训练后 checkpoint 的因果性；图构造仍为平方复杂度，没有实现流式缓存。

2026-10-14 最终验证：全部现有及新增测试 **138 passed（5.60 秒）**；上述运行源文件 `py_compile` 与识别原 CRLF 的 `git diff --check` 通过。三个入口的 `--help` 通过。独立代码审查未发现新增前向未来泄漏；审查发现的续训协议缺口已修复并以 13 项参数化回归测试覆盖。因果完整续训会拒绝种子、dropout、batch size、优化器超参数、损失与课程设置变化，允许延长总 epoch 数。

本次文件范围：

| 文件 | 改动 |
| --- | --- |
| `model_cm.py` | 三类模型的因果编码器、归一化、mask；FocalLoss 设备处理 |
| `model_hyper.py`、新增 `causal_graph.py` | 因果图分支；去除因果路径固定容量与位置权重 |
| `dataloader.py` | 复用前缀均值与协议检查；隔离因果课程难度 |
| `train_cm.py`、`caldiff/cal_diff.py` | 因果入口、信息移除、产物隔离与续训约束 |
| `scripts/evaluate_checkpoint.py` | 因果 checkpoint 与 CPU 推理入口 |
| `clloss.py` | 辅助张量跟随输入设备，不改损失公式 |
| 新增四份 `tests/test_causal*.py` | 模型、图、协议与完整损失合成检查 |
| `CAUSAL_README.md`、`REPRODUCTION_GUIDE.md` | 因果定义、命令、验证证据与限制 |

其他原有未提交复现修改保留；本次没有提交或推送代码。
