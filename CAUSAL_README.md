# CMERC 时间因果版

位置 `t` 的预测及辅助视图只能读取同一对话 `1..t` 的完整话语特征和 speaker。约束作用于预提取、按话语对齐的特征；原始特征提取链未审计。它不是因果效应识别，也不是要求离线训练总损失对所有未来训练样本梯度为零。

2026-10-16 审查后验证、修复证据和原始失败日志见 [reports/causal_review_followup](reports/causal_review_followup/IMPLEMENTATION_FIXES.md)。原 2026-10-14 的 138 项 CPU 检查是历史证据，本轮初始全套测试为 153 项；后续数值、真实特征、GPU 和恢复状态分别记录，不能互相替代。

## 模型与视图

使用 `--causal` 或 `Model(..., causal=True)` 启用；默认非因果路径保留。支持 `hyper/concat_DHT/avl` 的 GRU、LSTM、Transformer，av_encoder 与 residue 开关、多层图、非空对话和右 padding。

- GRU/LSTM 使用单向编码；Transformer 同时屏蔽未来和 padding keys。
- 文本实际执行逐话语、仅跨特征维的 LayerNorm。
- 同模态前缀超边分别定义 source membership 与 target delivery，只向当前节点投递；跨模态超边仅覆盖同一时间。GNN 仅使用过去到当前的时序边和同时间跨模态边，归一化不依赖未来出度。
- `Model_Intra/Inter` 是 **GNN-edge ablation**，分别删除同/异 speaker 的 GNN 时间边；编码器和超图仍可保留对应历史，不能称为完全删除某类上下文。
- `prefix_mean_v1` 为 `m_t=(x_1+...+x_t)/t`。完整均值序列可由 `x_t=t*m_t-(t-1)*m_(t-1)` 还原，因此它是前缀平滑视图，不等同于模态完全缺失。没有偷偷改成零值替换。
- 在同权重同 eval 模式下，首句的三种均值替换和两种 GNN 边删除均为 no-op。训练时的 dropout/辅助权重滞后可使置信度比较出现差异，须单独解释。

## 辅助训练策略

`--auxiliary-policy legacy_epoch_snapshot` 保留原实现：每次训练函数开始复制主模型到辅助模型；主模型 train、辅助 eval，后续 batch 的辅助权重滞后。

`--auxiliary-policy same_step_same_mode` 是另一项显式实验配置：每 batch 同步辅助；分类 pass 为正常 train/dropout 且保留梯度，置信度 full reference 与五视图均用当前权重、eval 模式、同一 ATen 后端。full reference 为 CC 保留梯度，移除视图和 HSCL 标签只读，HSCL 特征仍取分类训练 pass。GPU 中 cuDNN eval RNN 不允许 backward，所以该参考路径局部禁用 cuDNN；默认分类训练及 legacy 策略不受此局部设置影响。两种策略不可混为同一实验。

## 不可变 difficulty 与恢复

因果课程必须显式提供 `--difficulty-path`。旧 `*_causal_diff.pkl` 包装即使包含 `causal=True` 也不是来源证明，缺完整 provenance 一律拒绝。

生成器读取可信 checkpoint 后，先检查 causal/LN、validation 选模协议、实际训练/验证 ID 和特征哈希、训练与划分种子、架构、replacement policy 及训练进度。输出为新的 `Dataset_causal_diff_<SHA256>.pkl` 和 `.manifest.json`，或 `--difficulty-output` 指定的新路径；已存在文件安全拒绝。训练固定保存 artifact SHA，读取/恢复时重新校验实际 difficulty 和源 checkpoint 文件。`official_test`、不同划分、缺来源、文件被替换等默认拒绝，不提供未记录的绕过开关。教师 `epoch/best_epoch` 逐项记录；正式计划指定 `best.pt`，不能把任意 latest snapshot 称为最佳教师。

因果 run 目录带配置哈希，包含 baseline/full/消融和图深度等身份。非 resume 已有目录拒绝混写。`--resume` 校验 run manifest、difficulty、配置、历史日志、epoch 预算和 best 状态；只允许延长预算，best/旧 checkpoint 恢复建立有 lineage 的新分支。旧非因果 run 保持原命名但同样拒绝新任务覆盖。缺新 manifest 的历史 run 不会自动补签为本轮已验证恢复对象。

## 正式配对运行与第二轮状态

2026-10-17 第二轮已完成 IEMOCAP 单种子 Baseline 80轮 → 真实 difficulty → 完整 CMERC 80轮，validation 比较冻结后各 best 一次正式测试。[结果与证据](reports/causal_formal_round2/20261017T_round2_1937/README.md)保留全部指标及科学边界；第一轮 [FORMAL_EXPERIMENT_PLAN.md](reports/causal_review_followup/FORMAL_EXPERIMENT_PLAN.md)是当时的历史计划。MELD、多种子和消融仅准备，未启动。

固定划分种子和配对训练种子。使用独立输出目录；先训练关闭 CL/HSCL/CC 的因果 backbone，并显式延后测试：

```bash
bash scripts/run_training.sh IEMOCAP baseline validation \
  --causal --split-seed 1475 --seed 1937 \
  --replacement-policy prefix_mean_v1 \
  --auxiliary-policy legacy_epoch_snapshot \
  --defer-final-test --fail-on-nonfinite --diagnostic-batches 1
```

从实际输出目录选择对应 `best.pt`。使用同一架构、划分与配对种子生成 difficulty（这里的两个路径须替换成真实已验证产物）：

```bash
python caldiff/cal_diff.py --causal --base-model GRU \
  --dropout 0.5 --batch-size 16 --graph_type hyper \
  --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT \
  --modals avl --Dataset IEMOCAP --norm LN --num_L 3 --num_K 4 \
  --seed 1937 --split-seed 1475 --replacement-policy prefix_mean_v1 \
  --ckpt_path /path/to/verified-causal-baseline/best.pt

bash scripts/run_training.sh IEMOCAP cmerc validation \
  --causal --split-seed 1475 --seed 1937 \
  --replacement-policy prefix_mean_v1 \
  --auxiliary-policy legacy_epoch_snapshot \
  --difficulty-path /path/to/immutable-causal-difficulty.pkl \
  --defer-final-test --fail-on-nonfinite --diagnostic-batches 1
```

MELD 对应 `dropout=.4, num_K=3`。不同训练 seed 的教师不被静默混用；原始非因果 difficulty 和随机权重均不是正式因果输入。IEMOCAP/1937 已有本批合规真实教师和不可变 difficulty；其他数据集/seed 必须由对应配对 Baseline 生成，不能静默复用。初始化 state_dict 的真实 SHA256、难度哈希和完整 checkpoint 均须验收。后续直接可执行的教师/难度/配对验收、比较冻结和测试命令见[后续方案](reports/causal_formal_round2/20261017T_round2_1937/FUTURE_EXPERIMENTS.md)。

## 评测与验证

评测器从可信 causal checkpoint 的白名单架构字段构建模型，支持 GRU/LSTM/Transformer、非默认图深度、residue、av_encoder，并严格加载权重；原官方权重与因果结构不兼容，禁止 `strict=False` 掩盖问题。`cuda` 解析为明确逻辑设备索引。

`--split train/validation` 继承 checkpoint 的 split seed；有 provenance 时重算全量特征/ID 哈希后才取 `--max-dialogues` 子集。缺来源的合成或旧版诊断必须给出 `--diagnostic-data-reason` 并标记未验证，不能当作正式结果。已知 `official_test` 来源不能冒充 held-out validation。`--split test` 仅用于冻结方案之后的最终评测。第一轮审查未调用正式测试；第二轮已在 validation 比较冻结后对两个所选 best 各评估一次。`--defer-final-test` 保持训练期间及结束时不自动测试，默认旧行为仍保留。

数值回归、原始日志和可重跑命令见 [COMMANDS.md](reports/causal_review_followup/COMMANDS.md)。本机复用 `multimodalerc310`（PyTorch 2.2.2、PyG 2.4.0）和已有 pytest，无依赖升级；当时核实的 GPU 为本地宿主机 3，进程内 `cuda:0`；后续启动须重新核对健康设备 UUID 与资源，不能沿用旧编号。CPU/GPU 归一化和 mask、未来梯度、负对照、实际训练函数和生产维度真实特征小检查均有各自证据；GPU 恢复的确定性限制单独报告。

第二轮已完成 IEMOCAP 配对单 seed 正式训练/性能比较；单 seed 不支持跨 seed 显著性结论。原 PyTorch 1.13.1 环境和原始流特征提取链仍未验证。图构造仍为平方复杂度，没有流式缓存。旧固定路径批量自动化不是新的因果 config-hash run 发现器，应使用 manifest 中的实际路径。
