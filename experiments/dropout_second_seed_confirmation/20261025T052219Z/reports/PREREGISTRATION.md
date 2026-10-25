# 第11组Dropout0.5第二Seed：训练前登记

登记时间：2026-10-25T05:22:19.304367+00:00。唯一正式run `MELD_cmerc_dropout0p5_lr1e-4_seed67137`。研究问题：XP10 seed2024的Dropout0.5改善能否在用户预先固定的seed67137复现？只运行一次从头完整40轮；最多一次独立首Batch环境/容量Adam探针，不作为正式性能结果；正式进程重新以67137初始化，不加载探针或历史权重。

## 固定科学配置

相对于XP04 MELD完整CMERC seed67137，仅--dropout0.4→0.5。完整解析args逐键相等，允许工程差异仅skip_final_test、output_dir与provenance。与XP10的Seed差异是此次第二Seed设计，不把两Seed原始分数直接当单变量效果。

| 参数 | 登记值 |
|---|---|
| 数据 | MELD；同train1038/validation114；split_seed1475 |
| 随机 | seed67137；continuous RNG；ATen GRU；worker0 |
| 优化 | Adam；lr0.0001；L2/weight_decay3e-5；40 Epoch |
| Batch | train16 / validation16；capacity_safe；edge1000/node5200/retries10000 |
| Dropout | 新0.5，对照0.4；rec_dropout0.1保持原CLI |
| 模型 | 完整非因果；GRU；BN；hyper/direct；concat_DHT；avl；L3/K3；FP32 |
| 校准 | CC/rank_coff.002；HSCL/MSCL.15/CSCL.15，全部开启 |
| 课程 | CL开启；epoch_ratio.4；scheduler_steps1；同hash fixed_author |
| 选模 | 全验证池Weighted F1，原程序百分比两位小数严格提高，平局保留首个最大轮 |
| 测试 | 零测试loader迭代、零新增测试前向、新测试F1禁止 |

划分SHA `5317189c62939f28801eadf558763bd901cae04544ae829ebd45d1dd057055b8`。文本GRU层间与hidden-feature Dropout都由同一--dropout控制；rec_dropout.1在当前活跃构造路径未实际传入GRU，严禁修正此源行为。三个辅助模型沿用epoch入口权重及eval语义，不改网络、Loss公式/系数、课程序列、GRU后端或随机策略。

## 对照和预登记判断

仅复用已有40轮train/val历史：XP04 seed67137对照最佳65.24%/e8、末10轮62.909%，不重训；XP10 seed2024最佳66.49%/e15、末段62.775%，对应XP04为66.01%/e9、61.223%。第10组val分类Loss曾恶化，此事实不能因F1改善而省略。

末10轮固定31–40；波动报告时间窗样本标准差(ddof=1)，不称跨Seed标准差。比较最佳值、末段、最佳至末段衰退、Train/Validation分类Loss和完整辅助Loss；历史验证辅助Loss未记录则缺失，不重算或填零。

- 两Seed最佳验证W-F1均提高且末10轮均值均不下降：初步跨Seed一致性支持，可建议另行登记正式配置确认，不能宣称统计显著/直接确定五Seed方案。
- 第二Seed最佳未提高：未一致确认，不扩大为五Seed正式实验。
- 第二Seed最佳提高但末段下降：混合证据。

不根据中途或最终验证结果重训、改Seed/参数/指标门槛；测试分数不参与配置/选模。当前预算不含第二个新增Seed、Baseline、其他参数、五Seed或测试评估。两Seed不代表测试泛化或统计显著。

## 资源、隔离和透明限制

服务器固定本地user23，选健康TITAN Xp GPU0/UUID GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248，与XP04 seed67137同物理卡。XP10 seed2024及其控制在GPU3。任何GPU调用前固定CVD/UUID并冻结；每GPU阶段启动前3次新采样间隔10秒、free>=3000MiB/util<=95%，本任务显存预算2200MiB。阈值仅用于启动准入，不能证明持续隔离；外部PID变化仅记录，禁止终止其他任务。输出路径独立、文件锁阻止重复提交。

04–10档案及raw批次做完整哈希前后核验，代码/配置/输入/命令独立冻结，不覆盖旧文件。数据和原始日志、权重仅服务器。综合特征pickle含测试部分，原程序仍加载test loader元数据；零新增前向不等于未打开含test的数据文件。固定作者课程含validation键且生成史未知，同hash对照不等于完整预处理validation-free。

历史GPU来源审计中，独立审计代理误打印混合run_supplement，偶然接触历史测试汇总字段；本轮科学配置由用户在审计前预先固定，未依其变更任何配置/选模/判据，未进行新测试前向。事件记录不含测试值，见provenance/test_metric_access_incident.json。不得将本组描述成全程未接触历史测试指标。后续只读取明确的train/validation及设备投影。

## 完成门槛与交付

必须检查子进程正常返回、完整40train/40validation/2548预期Adam步、0test，实际模块/参数/数据/划分及每步Loss/梯度有限，同Seed历史采样重放、best首次最大/latest40、模型/Adam/全部RNG及冻结与历史哈希。正式和预检从头初始化且首步匹配。CPU预检/排队不是完成。验收后生成CMERC_DROPOUT_SECOND_SEED_CONFIRMATION.md、两Seed表/曲线/CSV及11档案，再同步GitHub；未授权追加训练。

冻结分两层：训练前主清单固定科学源码/全部参数/数据/预登记判据/守卫；仅分析与绘图的三个脚本及CPU测试证据在编写完成后单独保存analysis_frozen.json，允许与训练并行完成，不读取本次中途指标或改判据。CPU finalizer在分析/绘图前逐项校验此清单。
