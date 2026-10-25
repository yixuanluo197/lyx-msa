# 第10组Dropout对照：训练前登记

登记时间：2026-10-25T03:27:11.287346+00:00。唯一正式run：`MELD_cmerc_dropout0p5_lr1e-4_seed2024`。只允许1次从头完整40轮训练；不因验证结果重训、换Seed或改参。最多1次独立首Batch Adam探针，仅验前后向/四项Loss/梯度/容量，不作为正式性能结果；正式进程重新用Seed初始化，不读取探针或历史权重。

## 研究问题和唯一科学变量

MELD完整非因果CMERC的--dropout由XP04同Seed的0.4提高至0.5，能否缓解后期过拟合且提高最佳验证W-F1？其余解析后的科学参数必须逐键等于XP04，不仅对照命令行。允许工程差异仅skip_final_test=True、output_dir与provenance。

| 参数 | 固定登记 |
|---|---|
| 数据与Seed | MELD；seed2024；split_seed1475；冻结train1038/validation114 |
| 优化 | Adam；lr0.0001；L2/weight_decay3e-5；40Epoch |
| Dropout | 新0.5；历史0.4；rec_dropout0.1保持原值 |
| Batch | train16 / validation16；capacity_safe；edge1000/node5200/retries10000 |
| 随机与后端 | continuous；ATen GRU；FP32；worker0 |
| 模型 | GRU；hyper/direct；concat_DHT；avl；BN；num_L3/num_K3；非因果 |
| 校准 | CC=True/rank_coff0.002；HSCL=True/mscl0.15/cscl0.15 |
| 课程 | CL=True；epoch_ratio0.4；scheduler_steps1；同fixed_author文件 |
| 选模 | 完整验证池Weighted F1，原程序百分比两位小数严格提高；平局取首次 |
| Test | 零test loader遍历、零test前向、新test F1禁止 |

XP04历史对照真实核实为最佳66.01%/Epoch9、末10轮61.223%；不重训。训练/验证均保持Batch16，冻结划分SHA `5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8`。四个模型实例、课程排序、训练采样与每轮优化步数须CPU重放及运行日志核对。Dropout仅改变掩码概率，其他随机策略相同不保证模型数值轨迹相同。

## 源码作用位置

实际--dropout传入model_cm.Model及三个辅助实例，作用于文本两层双向gru_l层间dropout和分类头dropout_。av_using_lstm=False，音/视recurrent路径不激活。rec_dropout0.1在args中保持，但train_cm.py构造器未传dropout_rec，当前Model路径也不使用该形参；不能把它称为本次GRU实际dropout0.1。超图模块保存dropout属性不等于每层图传播执行Dropout。正式前只读枚举四实例有效模块，源码不修改。

## 评价与后续预算

固定辅助末段为Epoch31–40的验证W-F1算术均值。最佳值提高且末段不降：仅建议第二Seed确认；最佳升/末段降：混合证据，不升级最终配置；最佳未升：本Seed不支持主指标改善，不扩大实验。不得用末段替代最佳主指标，不计算单Seed显著性。第二Seed、五Seed、Batch搜索、其他LR或测试均未授权。

记录全部40轮Train/Validation W-F1/Accuracy、classification/CC/MSCL/CSCL/total Loss、CL比例/有效训练对话数、真实optimizer步数、梯度有限性、best/latest完整Checkpoint及状态。损失沿用原Batch均值，不改数学公式或权重；训练在线态与验证eval态间距不等于严格同模型泛化gap。

## 数据与资源边界

复用XP04同hash作者课程，包含validation键且生成历史未知；不声称预处理链validation-free。原综合pickle和test loader元数据会载入；零测试前向不等于未打开含test的文件。不会据测试调参。

服务器固定本地user23，健康TITAN Xp GPU3与XP04此Seed同物理卡；宿主GPU UUID及CUDA_VISIBLE_DEVICES见registry。可有其他任务共驻，禁止终止/信号他人进程。首Batch探针和正式启动前各需3次新采样（间隔10s），GPU身份匹配、free>=3000MiB、util<=95%、本任务估计峰值2200MiB；该阈值仅用于准入，不是持续运行保证。外部PID变化记录而不作硬拒绝；查询失败不能准入。冻结后不静默换卡。独立目录/排他锁/存在状态即拒绝重复提交。资源不足如实排队，不称完成。

## 完成验收与交付

必须同时确认正式子进程返回0、40轮train/validation、0test、所有记录Loss/梯度有限、同Batch16历史采样重放、best首次最大与latest40、模型/Adam/RNG/实际参数一致、冻结源码输入和04–09历史哈希不变。此后才生成CMERC_DROPOUT_DIAGNOSIS.md、逐轮CSV和曲线、10组档案并同步GitHub。数据/权重/原始日志保留服务器，当前没有发布新的待运行性能报告。
