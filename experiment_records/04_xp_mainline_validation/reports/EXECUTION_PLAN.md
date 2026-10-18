# 04 XP主研究基线执行计划

目标：建立“非因果、验证集选模、固定外部课程、五seed配对”的XP主研究基线。科学协议整体来自experiment_records/03_v100_multiseed/code，模型/损失/训练/评估文件不拼接旧XP实现。旧XP仅作为UUID设备计数与显存缓存工程经验来源。

正式矩阵：IEMOCAP/MELD × baseline/cmerc × [1475,67137,2024,3407,42]，共20项从头训练，80/40epoch，FP32、batch16、split1475、continuous/ATen、capacity_safe(1000边/5200节点/10000重排)。baseline关闭CL/HSCL/CC，CMERC三项全开，超参数继承V2脚本。validation选模，选定best后test，保留独立五路移除预测。

执行顺序：核验来源/环境/输入→CPU tests+真实1200排列→GPU算子与参数更新/分项梯度→官方checkpoint回归→同GPU两epoch连续/中断恢复→两数据集全课程显存与固定batch16五路评估→调度dry-run/锁/防覆盖/失败重试验证→冻结代码/配置/资源阈值→四锚点验收→其余16项→五seed统计/配对/独立重算→验收/主线登记/公开轻量档案。

四锚点：IEMOCAP baseline/cmerc seed1475；MELD baseline/cmerc seed67137，均计入20项。门槛仅技术完整性和协议，不含test分数目标。任何科学变更须新版本；失败保留并按原因恢复/重试原注册任务。

资源：user23 TITAN Xp，候选0/1/3，当前1/3空闲，0/2已有任务；每次启动重新检查型号、UUID、PID、整卡空闲和利用率，一卡一任务，GPU2不候选；不借V100。GPU4/ECC规则不从biggpu移植，XP不支持指标标N/A。草案11000MiB空闲阈值须在完整预检后结合峰值/实际占用冻结；当前空卡约12180MiB，不构造永久等待门槛。

环境：优先复用原XP独立环境，不修改已有环境/共享库；若确需安装或补丁才另建副本。继承UUID计数补丁先核验hash与实际CUDA映射，allocator GC0.8 + memory_fraction1.0仅用于torch1.13.1匹配版本，模型/精度/批量不变，记录工程差异。无matplotlib时沿用V2现成SVG后备，不新增依赖。

预设回归容差：官方checkpoint W-F1绝对误差≤0.05百分点、CEL绝对误差≤0.1；模型恢复rtol1e-3/atol1e-4，整数/采样元数据/选模epoch和Python/NumPy/Torch/CUDA RNG要求完全一致，优化器浮点状态rtol1e-3/atol1e-4。失败不放宽。

课程：fixed_author，与V2同SHA，不按seed重建。difficulty包含当前validation键；本轮实际训练先排除validation，但原评分生成未证实排除本轮holdout，不声称全预处理validation-free。明确数据使用违反协议时停止受影响任务。

数据大文件只读引用，每次启动/最终验证SHA，避免复制；模型权重/预测/日志保留本批目录并生成备份清单。历史01/02/03只读保留且记录树哈希，公开04仅代码/配置/审查材料，不含数据、checkpoint、逐样本预测或原始日志，不自动git提交/推送。

耗时与GPUhours等待本批实际XP完整课程预检后估算，不使用旧12-run工时保证。正式训练和验收在尚未发生前保持planned/pending。
