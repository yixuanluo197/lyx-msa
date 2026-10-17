# 第二轮 IEMOCAP 单种子配对实验：预注册协议

注册时间：2026-10-17 UTC。第一轮基准 `bbc486d6c21cf93a377ce0ee8171d3da181449fc`。新增运行支持以本目录冻结源码清单/commit 为准，算法文件不得改变。服务器绑定本地 本地同一服务器，沿用第一轮因果审计和最近 CMERC 运行归属；配对实验不跨服务器。

## 冻结配置

IEMOCAP；`causal=True`；验证集 W-F1 选模；训练 seed=1937，划分 seed=1475；对话级 10% holdout，108 train / 12 validation。全部特征字节 SHA256、有序 ID 与 ID 哈希见 `DATA_ENVIRONMENT.json`，不对测试集做调试或选模。特征提取过程未审计，不宣称原始输入端到端因果。

GRU；hyper / direct / concat_DHT / avl；LN；num_L=3，num_K=4；四路文本各1024维，audio1582，visual342；D_g=512，graph_hidden=512；speaker embedding启用，modal embedding关闭；residue关闭，av_encoder关闭。batch16，80 epochs，Adam lr=.0001，weight_decay=.00003；dropout=.5；其余参数以官方包装器和保存的完整有效配置为准。

Baseline：CL/HSCL/CC 全关。CMERC：三者全开，rank_coff=.005，mscl_coff=.05，cscl_coff=.05；epoch_ratio=.15，scheduler_steps=1。每轮课程比例 `min(1, round(.15*(epoch_index+1),2))`，保留当前课程方法。

replacement固定 `prefix_mean_v1`：可逆前缀平滑视图，不是完整模态移除。Intra/Inter只是 GNN speaker 时间边消融；编码器和超图仍可传递历史。辅助策略 `legacy_epoch_snapshot`，主模型正常train/dropout，辅助每次训练函数开始同步并eval，保留后续batch权重滞后。保留每次训练/评估函数调用重置seed的既有行为；cuDNN benchmark=False、deterministic=True，未强制所有CUDA算子逐位确定。

实际初始模型 state_dict（含buffer）使用相同算法 SHA256，比对 Baseline 与完整模型相等；不以“同seed”代替实际核查。每轮首训练batch记录未加权及乘系数后的梯度范数，其余batch不额外做分项反向；诊断不修改梯度。所有训练batch在更新前检查loss和活跃梯度有限。

## 顺序与关卡

1. 新增 `--defer-final-test`，两组训练均不运行最终测试；保留原默认行为兼容历史。有限真实 train/validation GPU preflight及新运行控制回归通过后，冻结源码并从头训练 Baseline 80轮，保存best/latest、Adam/RNG、逐轮损失和指标。
2. 逐轮记录连续1..80、选择指标有限、best为验证W-F1首个最大值（现入口保留两位小数选模，平手取早），best_epoch/metric与真实文件SHA完全核对。latest必须epoch79，teacher必须epoch==best_epoch。不得用非因果/随机checkpoint。
3. 只从该best在全部且仅108训练IDs生成真实difficulty。校验feature/ID/架构/训练与划分seed/teacher选模/训练协议/policy/源权重hash；生成独占hash-addressed文件和sidecar，再由严格reader验收，训练绑定artifactSHA。任意失败停止完整模型。
4. 完整CMERC从配对初始化训练80轮，预算、backbone、划分一致；完整状态及历史best可追溯。运行中断仅按既有完整resume协议继续，不调整超参数。
5. 仅评估两个best的validation；先保存并SHA冻结主要比较及选模对象。之后各best进行一次预定正式test评估。测试不能触发重选epoch/seed/参数或重新训练。

## 分析预注册

统一15个等宽置信度bins；Accuracy、Weighted/Macro F1、各类F1、ECE/MCE/RMSCE报告百分数，差值为百分点。CEL按当前评测器对true-class log-prob正差求和，报告total和text/audio/visual/intra_context/inter_context五项；不同policy不直接比较。训练耗时为逐轮记录之和，显存为各轮CUDA peak allocated最大值；另保留进程墙钟时间，不能混同。

预先定义相对位置 `(zero_based_utterance_index+.5)/完整对话长度`：前段[0,1/3)，中段[1/3,2/3)，后段[2/3,1]；长度1落中段。报告三个固定分段的分类和校准，不事后改变边界。两组使用相同对话/话语顺序，给出CMERC-Baseline绝对差，不将单seed差值写成统计显著提升。

本轮只启动此单seed配对闭环。MELD、训练seeds[1937,2025,3407]、w/o CL/HSCL/CC、独立same_step_same_mode策略对照只生成待运行方案。原始特征、权重、预测ID和大型日志只留服务器；公开交付仅源码、配置、去标识化汇总和必要小日志。
