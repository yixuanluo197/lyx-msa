# Plan B 新特征下游适配器

`downstream.py` 直接使用 `core_snapshot/` 的 XP04 固定源码 Model、Model_Intra、Model_Inter、官方分类/CC/HSCL损失及容量安全采样器。所有模型 causal=False，四层文本1024维、音频768维、视觉维度从新数据动态读取。源码来自 20261016T050823Z/code；未修改这些模型文件，不导入主仓当前分支。原非因果HyperGCN有硬编码 `.cuda()`，仅支持CUDA；宿主卡由根调度器明确限制。

数据集目录结构：

- `dataset.pt`：dict `splits={train:list,validation:list}`、`fit_dialogue_ids:list[str]`、`inner_dev_dialogue_ids:list[str]`。每条Dialogue为 `dialogue_id:str, utterance_ids:list[str], speaker:Float[N,S], labels:Long[N], visual:Float[N,V], texts, audio_paths`。
- `features/text_features.pt`：`{train:{did:Float[N,4,1024]},validation:{did:Float[N,4,1024]}}`。
- `features/audio_features.pt`：同结构，每Dialogue为 `Float[N,768]`。缺失音频须上游置零并在独立manifest保留原因；这里禁止使用旧特征补值，不增加新mask架构。
- 三个输入文件及features目录已有的 `*manifest*.json` 都纳入输入哈希。Fit与inner-dev必须互斥、并集等于完整effectiveTrain。仅train/validation存在，禁止Test。

CLI在激活原 `multimodalerc310` 环境后运行：

```bash
python downstream.py smoke --dataset IEMOCAP --data-dir DATASET_DIR --device cuda:0
python downstream.py train_m3net --dataset IEMOCAP --data-dir DATASET_DIR --device cuda:0
python downstream.py build_difficulty --dataset IEMOCAP --data-dir DATASET_DIR --device cuda:0
python downstream.py train_cmerc --dataset IEMOCAP --data-dir DATASET_DIR --device cuda:0
python downstream.py evaluate --dataset IEMOCAP --data-dir DATASET_DIR --device cuda:0
```

MELD同样接口。正式启动由根代理统一调度，本适配器开发期间未启动正式训练。smoke在目标目录还没有audio_features.pt时构造小型工程合成张量，报告明确synthetic=True；这不能替代真实新特征small smoke。

训练过程：

1. M3Net完全新初始化，calibrate/contrast/course全部False。只用TrainFit拟合，inner-dev macro-F1选epoch，patience10；IEMOCAP最多80轮、MELD最多40轮，不用官方Validation挑epoch，不选后重fit完整Train。
2. 用新M3Net的best.pt仅生成Fit难度：三个模态均值移除和两个官方上下文移除，真实类别log-prob正增量求和，加max speaker index+1，再除以话语数+speaker项。来源参考为cal_diff_reference.py。保留Fit全部Dialogue，不借用旧difficulty。
3. CMERC重新随机初始化，**不warm-start M3Net**；M3Net与CMERC的连接只有新难度文件。保持XP04 CC/HSCL/CL系数、课程比例和官方课程顺序逻辑。IEMOCAP类权重来自当前TrainFit 4747条的N/n_c（类0..5，缺类直接失败），M3Net和CMERC同权重；独立DOWNSTREAM_CLASS_WEIGHT_AMENDMENT记录数据/统计/哈希，不修改已有PLAN_B_PROTOCOL。MELD仍是原FocalLoss，不用Validation估计权重。
4. 每折/epoch之外不改变模型结构；课程ratio与官方实现一致。容量安全Batch保持固定16，不因长Dialogue删样本。课程从Fit子集排名后恢复原顺序。
5. CC/MSCL/CSCL由原train_or_eval_graph_model计算。正式训练每个SCL_base原始logits在官方零替代之前检查有限性；发现非有限即记录nonfinite_failure.json并失败，禁止静默清零。smoke另外记录三个损失到audio projection的独立梯度。

`runs/{m3net,cmerc}/best.pt,last.pt` 保存模型、Adam、完整Python/NumPy/Torch/CUDA RNG、epoch、best/bad-epoch计数、有效配置及源码/输入哈希。`--resume DATASET_DIR/runs/STAGE/last.pt` 从下一个完整epoch恢复；严格核对代码、输入、配置、difficulty及metrics连续epoch，不把加载权重称为resume。Aten recurrent backend沿用XP04的可保存RNG策略。CUDA scatter仍可能产生浮点归约级差异，不承诺跨进程逐bit相等。

预测API：

```python
from downstream import load_model, infer_dialogue
model = load_model(dataset_dir, checkpoint, device='cuda:0')
prob = infer_dialogue(model, dialogue, textfeatures, audiofeatures)  # numpy [N,C]
```

输入为一个完整Dialogue；只替换目标话语的一种新模态特征后，全Dialogue前向。该API不读取labels，强制eval、不和其他Dialogue合batch，因此BN使用冻结running统计。模型加载拒绝没有new_features_only=True的旧checkpoint。正式Validation评价输出accuracy、macro/weighted F1（0..1）、15等宽置信度bin ECE、按类别求和的Brier、NLL及完整softmax。

正常early-stop或max-epochs结束才写runs/STAGE/completion.json，含终止原因、best/last、输入/源码/协议/amendment哈希；checkpoint存在本身不等于训练完成。若在最后checkpoint后、completion前中断，resume不会重复已完成epoch或越过已满足的patience。新difficulty另外保存difficulty_components.csv/json，逐Fit Dialogue保留长度、speaker项、五路gap求和及最终值，供独立公式复算。
