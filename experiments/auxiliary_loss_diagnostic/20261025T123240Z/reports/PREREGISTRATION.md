# XP13：辅助损失梯度冲突与HSCL伪标签诊断登记

本计划在查看任何本组实际梯度结果前登记。既有Python/PyTorch冻结源码输入和Python/matplotlib绘图工作流沿用，不开展训练/系数搜索；不预设辅助项有害。

## 固定对象与预算

优先四个真实Checkpoint：XP04 MELD完整CMERC seed67137 best e8、latest e40；XP12同Seed w/o HSCL best e5、latest e40。CPU先核存在、SHA、args/provenance、epoch、模型及Adam/RNG记录；缺失则诚实标记，不重训。只用best/latest实际快照，不虚构早期/连续轨迹。四者Dropout.4/lr1e-4/B16、split1475等基础配置相同；本组不更改源网络和Loss数学。

固定32个Batch16、512不重复训练对话，以XP04 train1038为唯一采样池。采样Seed67137；按(length,str id)排序并用floor(q*N/4)分为四长度rank四分位，每层先以私有Seed随机打破并列贪心覆盖可得七情绪类别，再均匀无放回补足128对话、私有shuffle后分8个B16。只使用训练标签/长度，完全不依据预测、梯度、验证或测试表现挑选。容量检查sum(length)+3B<=1000、3sum(length)<=5200；不合格直接报告，不按结果换样本。采样覆盖、类别偏差与规则/输入ID私有清单在GPU前冻结。同一32Batch用于全部Checkpoint。四个目标Epoch的CL比例均已为1，诊断不重演课程或训练。

共128个预登记固定Checkpoint-Batch案例；每例原始主前向1次+teacher移除前向5次=768个顶层模型前向。每例最多4次autograd.grad，共最多512调用，若原实现常量/非有限Loss不能求梯度须记实际跳过/失败，不能伪造正常结果。不额外做其他Seed/Dropout/学习率/Batch搜索。新训练次数0、optimizer构造/step0、Tensor.backward0、测试前向0。

## 原实现复用和状态

使用本独立副本code_xp04，字节继承XP12；模型构造调用及尺寸block从冻结train_cm.py AST提取，不手写替代模型。strict load四权重。调用原train_or_eval_graph_model(train=False)，全程torch.enable_grad，仅复用其eval路径；在原combine_training_losses返回处捕获实际原始class/CC/MSCL/CSCL与伪标签张量，完成autograd诊断后抛内部停止信号，保证到不了原backward/optimizer.step。附源码AST/hash及无更新守卫。只创建显式训练keys的Dataset，不构造验证/测试loader；综合pickle仍含其他split元数据但不访问其对话做前向。

主模型及五路teacher均eval，Dropout关闭、BN使用保存状态。每个Checkpoint-Batch前后核模型参数/所有buffer及RNG相等、.grad缓冲无变，磁盘Checkpoint不写入。teacher按原函数在每次调用入口复制当前Checkpoint学生权重；这不同于历史训练中Epoch入口teacher快照与随后学生更新的关系，不能称为真实历史训练梯度。原历史teacher/中间权重未保存，不重构它们。

## 梯度支持与统计

四原始Loss独立对requires_grad主模型参数求梯度，保留None/常量/零/非有限类别。预登记参考系数CC.002、MSCL.15、CSCL.15。XP04参考即实际；XP12实际HSCL权重0，其raw及参考加权梯度属于离线反事实强度比较，另列actual，不能称为XP12历史优化项。

层组按实际name：text_gru(gru_l.*)、audio_projection(linear_a.*)、visual_projection(linear_v.*)、graph_shared(graph_model.*)、classification_head(smax_fc.*,hidfc.*)、other，另shared_backbone为非head全部参数、all为全部。是否有梯度由测量决定，不能按名称预设。

每对只在双方均非None、finite且该参数Tensor各自L2>0的共同Tensor支持比较，保留Tensor全部坐标、不按坐标挑选。缺失/零/非有限计数及无效分母分列，不能用0余弦代替未定义。报告L2、内积、余弦、cos<0频率、参考/实际加权范数及相对classification范数比；加权辅助和由独立梯度线性和得到。冲突频率分母为实际有效Batch数，同时报告总数和不可比数。按Checkpoint/层组统计分布，32 Batch不是独立Seed，不做伪重复显著性检验。

## 二值伪标签与对比对

伪标签是t/a/v/intra/inter五组关系二值，绝不计算与七情绪类别的一致率。直接记录原实现输出并验证分支：主错/移除对→显式0，主对/移除错→显式1，均对时比较真实类log概率（>=为0，下降为1），均错→未命中规则保留默认0。统计0/1、真实情绪条件分布、主模型正确性、移除前后真实类概率/log概率变化和正确性转移、默认0覆盖与语义混杂；不改规则。

记录CKSCL真实同/异类非自配对、零正或零负anchor、n/nc类别权重、实际base logits有限性及按关系/情绪聚合的loss贡献。特别审计eq+1e-7造成的异类微小正mask与同类微小负mask，以及wrapper NaN/Inf→0原始回退，全部保留原数学，不能把有限返回值误说为内部未异常。

## 资源、隐私和解释

本地user23 GPU0/UUID GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248，与主要XP04来源同卡。启动前三次/10秒新采样，free>=4000MiB/util<=95%；不终止其他任务、不跨服务器。诊断可逐case持久记录，失败保留日志/计数，不根据结果追加有利案例。

所有原始对话ID、逐话语真实标签/预测/概率、原始梯度及敏感样本内容仅private目录/服务器，不公开文本。公开B01等批次号、聚合CSV、覆盖counts、输入哈希、源码/配置/图及验收；不发布原始weights/data/logs。旧报告只读，分析使用净化train/validation证据，不据历史或新测试选择方向。

至少六种真实数据图：cos分布、加权梯度强度、层冲突率、五组二值比例、情绪条件关系heatmap、同Batch不同Checkpoint差异，PNG/SVG和源CSV/SHA。最终事实/机制假设/不能证明的猜测分列，根据预登记固定范围内证据决定研究优先级，不强行选模块。负cos只表示局部方向反向，大范数不等于有害。单Seed/离散eval快照不足证明论文F1根因或XP12因果收益。

独立验收4CP/输入/04～12历史保全、实际前向/grad次数、新训练0/step0/test0、参数/buffer/RNG不变、表图报告和私有明细汇总一致。最终中文reports/CMERC_AUXILIARY_LOSS_DIAGNOSTIC.md与experiment_records/13_auxiliary_loss_diagnostic；通过后发布GitHub，不自动启动下一次训练。
