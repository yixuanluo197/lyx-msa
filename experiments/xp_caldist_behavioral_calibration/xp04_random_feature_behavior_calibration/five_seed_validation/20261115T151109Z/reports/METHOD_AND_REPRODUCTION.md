# 固定五 Seed 方法与复核入口

## 研究问题与方法

本轮检验固定的低容量行为修正能否在原始置信度映射之外改善 Top-label 正确概率。主模型为 XP04 + E1_DA。两个数据集采用相同结构、同一候选正则集合，分别按其训练折拟合参数；不能按最终 OOF 指标替换主模型。

令 C 为冻结 XP04 原始预测类别的最大概率，q0 = sigmoid(a + b C) 为 E1。基础映射在对应训练 Dialogue 上拟合并冻结。对 Gaussian σ=.05 和训练均值替换 Dropout .05 各进行10次联合模态扰动，令 pk 为扰动后**原始预测类别**的概率：

- D = C − mean(pk)
- A = sqrt(mean((pk − C)²))
- B = [Gaussian D, Gaussian A, Dropout D, Dropout A]
- q = sigmoid(logit(q0) + StandardScaler(B) · w)

行为修正没有额外截距，w 为4维；可上调或下调置信度，不改变类别。训练目标为平均二元负对数似然 + λ||w||²/2。基础与行为标准化只使用相应训练 Dialogue。

E1_DA 属于本研究的双向 CaliDist-inspired 扩展，不是 CaliDist 原始 C×sigmoid(R) 评分。D/A 是适配指标，不冒称原论文公式。E1_MUDELTA 用原定义 μ/δ 替换 D/A，L0 系列将基础输入改为训练折标准化 logit(C)，均为预登记消融。

## 防止外层 OOF 结果参与选择

IEMOCAP 每 Seed 外层留一 Dialogue，共12折；MELD 外层5折 GroupKFold。每个外训练集内再用5折 GroupKFold，重拟合基础映射与修正，通过合并内层留出预测的 Brier 从 off/1/.1/.01 选择 λ，差值≤1e-12时优先 off，再优先较强正则。外层留出标签不进入训练与选择。

off 意味行为权重全为零，结果回退至对应基础映射。它是预先允许的选择，应如实报告，不能因其关闭而扩大特征或修改搜索空间。

## 输入和复现

以当前目录为 RUN。完整原始概率见 baseline/original_predictions.csv。Seed1475行为数据复用 ../../pilot/20261113T025053Z，各新 Seed 的完整扰动概率、log-probabilities、随机键和完成证据位于 runs/<Dataset>/seed<Seed>/。每条原话语只生成一条行为训练记录。

冻结核心实现位于 provenance/analysis_code/calibration_core.py，源码与 unified_module_pilot/20261115T145425Z 完全一致。软件环境使用 /data2/yb/conda_envs/cmerc-xp-repro/bin/python。分析入口：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data2/yb/conda_envs/cmerc-xp-repro/bin/python scripts/analyze_behavior.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data2/yb/conda_envs/cmerc-xp-repro/bin/python scripts/verify_behavior.py
```

分析脚本拒绝覆盖已经完成的正式分析；需要复跑时复制整套输入到新的实验目录。独立验证器可在已有结果上运行。GPU原始推理脚本 scripts/run_joint.py 同样拒绝覆盖已存在输出；复核已有数据不需要重新进行 GPU 推理。

源码中保存了原机器绝对路径用于溯源；在其他机器上复现须重新映射项目根目录与数据位置，不能直接把命令视为可移植安装程序。GitHub不包含 checkpoint 和原始特征，均通过哈希核对；派生预测、训练统计量与折参数随结果保存。

## 统计解释边界

主指标为固定15区间 ECE15、Binary Brier、Binary NLL，均使用预测是否正确的二元口径。五 Seed 报告均值及样本标准差；不能把它们当作五份独立数据。2000次配对 bootstrap 共享同一次 Dialogue 抽样，在各 Seed 内计算指标差，再平均五 Seed 的差；区间未重拟合校准器，也没有调整多重比较。消融区间是探索性描述，不能据此宣称筛选后的显著优势。

Validation 曾用于 XP04 checkpoint 选择和 Pilot 方法设计；本轮新增四 Seed 检验训练随机性稳定性，不能提供完全独立的数据泛化证据。未使用 Test 标签、未修改模型权重、未调整温度、未进行原始数据干扰。
