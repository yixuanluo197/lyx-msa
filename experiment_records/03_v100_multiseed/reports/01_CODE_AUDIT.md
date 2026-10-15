# 多随机种子增量代码审计

原完整审计保留在 `reference_single_seed/01_CODE_AUDIT.md`；本文件只记录独立多seed工程的增量。原工程 `../CMERC` 的代码和结果不修改。

| 检查项 / 位置 | 影响 | V2处理及证据 |
|---|---|---|
| `train_cm.py:468` IEMOCAP划分曾绑定训练seed | 高：跨seed混入验证集合差异 | 独立split_seed=1475，所有seed/模型108 train、12 valid、31 test；完整IDs/hash保存manifest |
| `train_cm.py:545`、`:615` 每次train/eval重置seed | 高：重复dropout随机起点，不是连续训练RNG | 新continuous策略仅初始化播种，评估保存恢复RNG；legacy仍保留，旧分数不重命名为新协议 |
| `train_cm.py:307` DataLoader随机过程 | 中：恢复/worker行为需一致 | privategenerator=train_seed+epoch，worker由torch.initial_seed派生；每轮可重建，测试覆盖worker与恢复 |
| cuDNN GRU内部dropoutbuffer不进入普通checkpoint | 高：真实中断后随机状态偏移 | 仅continuous模式的RNN前向改ATen后端；结构/概率/权重不变，真实GPU恢复通过；详见协议审计 |
| `model_hyper.py` 固定1000边/5200节点参数，V1 batch发生1049边 | 高：某些自然seed/epoch CUDA assert | 保留模型参数和batch16；`train_cm.py:316`拒绝不安全整轮排列，不删样本、不用标签/测试分数；真实1200排列扫描通过 |
| `train_cm.py:146`、`:159` manifest/checkpoint与resume | 高：不能把仅权重加载称作续训 | 完整model/optimizer/RNG/epoch/config、split/data/source哈希、采样参数及实际容量；配置/provenance不一致拒绝恢复 |
| `scripts/run_training.sh` 与multi-seed调度 | 高：因果混用或覆盖旧产物 | 显式禁止causal；独立seed路径；已有run/log/result排他创建；旧单seed入口仍可用legacy |
| `scripts/multiseed_scheduler.py` GPU与任务门控 | 高：故障卡、并占、未验证即批量运行 | whitelist1/2/3/5，明确禁GPU4；每次检查进程/ECC/余量/锁；四项真实短测通过后才启动12项 |
| `scripts/evaluate_checkpoint.py`、`scripts/summarize_results.py` | 高：测试选模、跨协议混算、挑seed | 仅validation-selected best完整训练纳入；seed/config/source/split一致；三seed门控无分数阈值，五seed预注册；V1/V2不混算 |
| 汇总原先依赖缺失matplotlib | 中：训练完成后统计退出 | 标准库SVG真实曲线降级，不安装额外依赖、不伪造点；CPU回归覆盖缺失matplotlib |
| `scripts/summarize_results.py:339` V1失败epoch文字 | 低：零基5误写成第5轮，应为第6轮 | 不编辑在运行的冻结源码；独立 `10_MULTI_SEED_REPRODUCTION_ERRATA.md` 勘误。正式选模/统计的一基epoch无此错误 |

`model.py`、`model_cm.py`、`model_hyper.py`、`clloss.py` 与原工程文件SHA256逐一相同。原非因果BN双向模型/损失、batch16、lr1e-4、80/40epoch、课程参数不变。RNG/ATen/条件批次策略属于明确工程协议变化，不能将与历史单seed差异全部解释为纯训练seed波动。

验证：本地及远端torch1.13.1 CPU各196 passed、1 GPU-only skipped；实际完整CMERC两epoch中断/恢复已通过，4项真实调度短测均训练/独立评估完成且通过paired/source/data/split/finite门控。正式五seed统计尚未完成。
