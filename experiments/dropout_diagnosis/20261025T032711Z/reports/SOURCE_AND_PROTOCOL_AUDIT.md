# 第10组源码与协议只读审计

独立审计：Codex native subagent iem_batch_audit；主执行者另核XP04原始manifest与40轮train/validation字段。本阶段不读取final_test或测试预测。

## Dropout实际作用

源为09冻结code_xp04字节副本，model_cm/model_hyper/HypergraphConv/high_fre_conv与XP04字节相同。train_cm仅继承既有skip-final-test三处工程补丁；其反向补丁SHA应等于XP04源，无网络/Loss数学修改。实际--dropout通过train_cm.py:1050、1082、1114、1146传给Model×2/Model_Intra/Model_Inter。

| 路径 | 源码 | 本次作用 |
|---|---|---|
| 文本两层双向GRU | code_xp04/model_cm.py:658、824 | 层间dropout0.4→0.5 |
| hyper输出分类头 | code_xp04/model_cm.py:713、893–898 | dropout_.p0.4→0.5 |
| 音/视GRU | code_xp04/model_cm.py:646、651 | av_using_lstm=False，不构造 |
| HyperGCN属性 | code_xp04/model_hyper.py:102、125 | 仅保存dropout；HypergraphConv use_attention=False/dropout0，无新增图传播Dropout |
| rec_dropout | code_xp04/train_cm.py:854；model_cm.py:562、908、1263 | args始终0.1，构造调用未传dropout_rec，当前形参未使用 |
| 固定门控Dropout | code_xp04/model_cm.py:479–481、712 | 固定0.5，实例化注释，不在当前有效路径 |
| TextCNN/Transformer/位置编码 | model_GCN.py:196、201；model_cm.py:606、663–674；model_hyper.py:65–67、110 | 当前GRU/use_bert_seq=False/use_position=False，均非活动路径 |

train_cm.py:607–613：主模型随train/eval切换，三个辅助模型复制入口权重并eval。因此训练Dropout改变主模型随机前向；三个辅助实例保留同配置但eval时不采样自己的Dropout掩码，验证亦关闭Dropout。本次一个CLI超参数同时影响两个活跃位置，不能将差异独立归因于GRU或分类头。

## CPU初始化审计

独立只读审计隐藏全部CUDA，仅构造四实例，无前向/反向：每实例83个state_dict键、14,109,769参数，0.4/0.5两配置分别同Seed初始化后的张量、形状及Torch CPU RNG均逐字节相同；只观察到gru_l.dropout/dropout_.p的概率差异。此结论只覆盖初始化，不承诺后续轨迹相同，也不替代GPU预检。正式观测器需另外记录四实例实际Dropout属性，初始权重和首步RNG。

## 真实历史对照

MELD完整非因果CMERC seed2024/split1475、dropout0.4、rec_dropout0.1、batch16、lr0.0001、40轮。验证最佳66.01%/Epoch9，末10轮61.223%；e9 train分类Loss3.297963472513052、val分类Loss4.1074，末10轮均值2.70762985926408/5.08928。源文件及核验JSON见provenance/historical_control_verification.json；历史不重训。

实际比较要求全部解析args逐键相等，唯一科学差异dropout；工程允许skip_final_test/output_dir/provenance。两loader仍batch16，完整train1038/val114划分、课程同hash、原sampler AST、每Epoch样本池/组序及更新数一致。训练Loop、class/CC/MSCL/CSCL公式、类权重及Batch聚合不改。

原Loss按Batch均值，同一batch下可比，但在线训练态和验证eval态不是同一固定权重的泛化gap。验证最高采用原百分比两位小数严格提高、平局留最早，测试loader任何迭代即拒绝。新训练禁止final_test及预测NPZ产物。

## 剩余限制

单Seed、历史时点、CUDA scatter非确定性和共享GPU限制因果/速度与跨Seed结论。fixed_author课程含验证键且生成历史未知；同hash配对不等于预处理validation-free。综合特征pickle及测试loader元数据仍由原代码载入，零测试前向不等于没有读取含测试数据的文件。本组不做测试、不根据分数追加配置或预算。
