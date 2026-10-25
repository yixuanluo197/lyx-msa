# MELD Baseline Checkpoint 先行审计

审计日期：2026-10-25。此文件在本轮新 GPU 推理之前生成；审计本身只读取既有记录、源码、特征和可信本地 checkpoint，并导出数值相同的纯 state_dict。

## 固定候选与结论

按用户预先规定的兼容性优先级，固定使用：

`experiments/official_readme_reproduction/checkpoints/MELD_baseline_seed67137/best.pt`

- SHA-256：`a6ab85cb6c0c12054dade9c6bcdf01cfaa6abc417625059a927de5ba386d3738`。
- 文件大小：142,006,793 字节。
- 来源：本地 `user23` 的 TITAN Xp 上，以 CMERC 官方 README 的 MELD Baseline 命令从随机初始化训练；**不是 M³Net 作者公开的 MELD 权重**。
- 内部 `epoch=5`（从 1 计数），`selection_protocol=official_test_selection`；训练配置 seed=67137、总计40轮。
- 模型身份由权重内部 args、固定命令、状态和观测器源码共同确定；未凭文件名或参数形状单独认定。
- 技术兼容：其 `model` 中的83个张量与作者 MELD CMERC 权重的名称、形状、dtype 完全对应。后续必须对已确认官方结构使用 `strict=True`，禁止静默跳过参数。
- 首选已满足兼容要求，因此不切换其他候选，不以多个测试 F1 进行选择。本次审计不读取测试分数决定候选。

**已确认风险：该历史 checkpoint 使用测试集选模。** 固定观测器在每轮测试后比较官方返回、已四舍五入的 test Weighted F1，仅严格更高时保存，故保存的是40轮中首次达到最高四舍五入测试分数的权重。它不属于独立 Validation 选模；本轮固定加载且不再选模不能消除这项历史偏差。

## 权重读取和安全导出

完整历史 checkpoint 内含 `model`、`optimizer`、`epoch`、`args`、Python/NumPy/Torch/CUDA RNG 和 `selection_protocol`。PyTorch 1.13.1 的 `weights_only=True` 初次 CPU 加载失败（`Unsupported operand 71`）；仅在 SHA 与此前可信本地审计记录完全一致之后，以 `map_location=cpu, weights_only=False` 读取该已知本地文件。未对未知远程 pickle 使用此方式。

只提取 `payload['model']`，另存：

`checkpoints/baseline_candidates/meld_readme_fixed_20261025/model_state.pt`

导出文件以 `weights_only=True` 成功读取；逐键 dtype、形状和完整数值内容指纹一致。源完整 checkpoint 没有改写。导出 SHA、源/导出相同的张量指纹和完整参数清单见 [baseline_checkpoint_audit.json](baseline_checkpoint_audit.json)。权重保留服务器，不作为 Git 报告文件上传。

## 已核实配置

| 项目 | 固定 Baseline |
|---|---|
| Dataset / 类别数 | MELD / 7 |
| 基础模型 | GRU，官方双向完整上下文实现 |
| graph_type / graph_construct | hyper / direct |
| multi_modal / modals | true / avl |
| fusion / norm | concat_DHT / BN |
| dropout / rec_dropout | 0.4 / 0.1 |
| Batch / num_L / num_K | 16 / 3 / 3 |
| calibrate / contrastlearning / courselearning | false / false / false |
| rank_coff / mscl_coff / cscl_coff | 0 / 0 / 0 |
| 训练 seed / Epoch | 67137 / 固定第5轮，共训练40轮 |
| 选模 | official_test_selection |
| causal | args不含该扩展字段；固定源码使用双向GRU及官方完整上下文，未采用本地因果变体 |

原有训练入口为固定官方 `train_cm.py`；Baseline flags 均关闭。官方 `model_cm.Model.__init__` 与推理用 `model.Model.__init__` 的 AST 完全一致；`Model.forward` 在 hyper 分支只多保留特征引用并在返回 tuple 后增加该特征，前五项及 `log_prob` 运算相同。后续评估沿用已验证的 `train.py` / `model.py` 完整输入路径。

原训练和作者验证使用同一官方 commit：`815c7c060914f5b55db816c0ba3662b11d08552f`。原训练冻结运行源码中的 `model.py`、`model_cm.py`、`model_hyper.py`、`HypergraphConv.py`、`dataloader.py`、`train.py`，均与已有官方 CMERC 性能验证 `code_runtime/` 对应文件 SHA 完全一致。

运行源码都只含两项既有 PyG 兼容补丁：删除未使用且新版缺失的 topk import；反向 HypergraphConv propagate 的 size 修正为 `(num_nodes,num_edges)`。没有为了分数改变模型、测试标签或数据。

## 数据、标签与评估协议

| 文件 | SHA-256 |
|---|---|
| MELD_features_raw1.pkl | `d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0` |
| meld_features_roberta.pkl | `58b2bdadc2eaec6abb40ac0cb3ffa3f31e929d636ece8197aecf1ca373a92951` |

本次重新计算本地特征 SHA，并与原训练 `provenance/inputs.json` 对应记录核对一致。模型使用四组1024维 RoBERTa、300维音频、342维视觉。类别顺序为 `neutral, surprise, fear, sadness, joy, disgust, anger`。

测试 DataLoader 使用原始 `testVid` 顺序、Batch16、num_workers2、无 shuffle；长度/mask 排除 padding，按每个对话有效长度顺序展平。目标为2610条有效话语；最终须与已有官方 CMERC CSV 的稳定 ID、原文、真实标签、顺序逐项核对后才能宣称对齐。

原训练环境：`user23` / TITAN Xp，Python3.10.13、torch1.13.1+cu117、CUDA11.7、cuDNN8500、PyG2.4.0。冻结官方入口设置 `cudnn.benchmark=False`、`cudnn.deterministic=True`，模型调用默认 `nn.GRU`，观测器/启动器没有 ATen 后端覆盖。历史 checkpoint 不含逐前向实际内核 dispatch 追踪，因此后端结论属于冻结源码与环境证据，不能伪称具有历史逐内核日志。

**本轮按用户明确要求，在 biggpu 的健康 V100 上统一到已验证官方 CMERC 的 cuDNN 评估路径。** 这是一轮跨训练硬件的固定权重标准化推理，并非声称 Baseline 在 V100 上训练。biggpu 物理 GPU4 为坏卡，必须排除；实际 GPU/UUID 以本轮 manifest 记录为准。不得为此改变 Batch、Padding、测试顺序或模型。

## 其他候选盘点

按预定后备顺序盘点以下文件是否存在并记录 SHA：`runs/formal/baseline/MELD_official_test_seed67137/best.pt`、`runs/formal/baseline/MELD_validation_seed67137/best.pt`、`experiments/xp_mainline_validation/*/runs/baseline/MELD_validation_seed67137/best.pt`。结果见 JSON。它们没有被推理比较或用于挑选更高分；首选兼容后即停止候选切换。

## 可比性判断与范围

1. **相同测试协议下的固定权重描述性行为比较：技术上可执行。** 最终仍须以新推理的 strict load、参数不变、概率校验和2610条样本对齐为验收依据。
2. **严格训练配对及模块因果收益：不成立/未确认。** Baseline 为已知 seed67137、第5轮、历史测试选模的本地复现；CMERC 作者纯权重未提供真实训练 seed、Epoch 和选模记录，也没有与该本地 Baseline 配对的证据。不能将差异归因于 CL、HSCL、CC，不能称为严格模块消融。
3. 本轮不训练、不改权重、不改标签、不执行模态删除、不按测试成绩改配置。此前 IEMOCAP 实验二使用 M³Net 作者公开权重，本轮 MELD 使用本地 README 复现权重，来源须分别表述。

审计机器证据：[baseline_checkpoint_audit.json](baseline_checkpoint_audit.json)。可复现只读审计脚本：[audit/inspect_fixed_checkpoint.py](audit/inspect_fixed_checkpoint.py)。
