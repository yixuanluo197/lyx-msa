# XP04 原μ/δ评分主线小幅调整

本轮复用原特征随机干扰，不重新运行CMERC，不新增随机扰动。主模型只输入原始C与两种扰动各自的原定义μ、δ，联合学习基础置信度系数和行为分数；所有消融固定登记。它是CaliDist-inspired学习式校准扩展。

## 当前完成批次

- 五Seed联合模态：`joint/MAINLINE_RESULTS.md`、`joint/MU_DELTA_ABLATION.md`、`joint/FINAL_MAINLINE_DECISION.md`。
- 入口协议：`MAINLINE_PROTOCOL.md`、`preregistration.json`。
- 输入与来源：`inputs/`；完整内外折参数、OOF分数、指标、2000次Dialogue配对bootstrap见`joint/`。
- 数值独立审计：`joint/independent_verification.json`。交付只读检查：`python verify_delivery.py`。

## 主结果：五Seed均值，越低越好

|数据集|方法|ECE15|Binary Brier|Binary NLL|
|---|---|---:|---:|---:|
|IEMOCAP|XP04|0.136499|0.234443|0.809438|
|IEMOCAP|Confidence-only|0.072514|0.220685|0.634827|
|IEMOCAP|C+原μ/δ|0.069459|0.222093|0.638626|
|MELD|XP04|0.209270|0.237788|0.668629|
|MELD|Confidence-only|0.041635|0.193374|0.569588|
|MELD|C+原μ/δ|0.043266|0.193714|0.570243|

加入模块后的系统优于XP04，但不能将超过Confidence-only的收益归于行为信息：IEMOCAP只有ECE均值略低，Brier/NLL五Seed均变差；MELD主模型三指标均略差于Confidence-only。完整样本标准差、方向和区间见CSV及主报告。

## 证据边界

随机特征扰动确实引起预测与置信度变化：联合扰动五Seed每次翻转率均值，IEMOCAP Gaussian 0.47%、Dropout 1.66%；MELD Gaussian 0.23%、Dropout 1.18%。这支持干扰响应可测，不证明它提供额外可靠性信息。当前.05强度/10重复的原μδ评分没有建立额外校准价值，不能写成校准机制已经成功。

可以据此提出后续单独登记的、小规模面部遮挡/音色处理机制Pilot，检验更具语义或物理含义的干扰是否产生更有用的μ/δ；这是新的探索假设，不能视为当前结果已证明应进入完整原始数据实验。本轮不修改原始数据。原始数据许可及特征重新提取可行性要在后续工程Pilot中核对。

所有方法只调整原预测正确概率，预测类别和Accuracy/F1不变；所有拟合仅当前训练Dialogue，真实Validation固定；没有Test标签、温度缩放、CMERC训练或权重改变。重复使用Validation仅构成探索性证据，bootstrap不含重新拟合且未作多重比较调整。

## 重现

原数值环境与来源路径见`inputs/environment.json`和`inputs/source_hashes.json`。`scripts/test_mainline.py`提供工程测试；`scripts/run_joint.py --run`和`scripts/run_modality.py --run`拒绝覆盖完整批次。数值脚本依赖原数据路径，换机器需配置对应来源路径，并在独立副本中重现。交付检查只使用Python标准库，任意目录可运行，不修改结果。
