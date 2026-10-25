# 辅助Loss只读诊断图契约

- 后端：Python/Matplotlib，已有绘图环境，不安装依赖。图为定量诊断，不预设CC/MSCL/CSCL中哪个机制有害或应优先修改。
- 待实测判断的核心问题：在预先固定的四个Checkpoint、同一组32个训练Batch的eval+grad快照上，各辅助Loss与分类Loss的梯度方向/尺度和关系伪标签分布呈现什么差异；结果能否提供下一步单机制研究线索。
- 证据层级：XP04 best/latest同Batch配对为主要诊断；XP12 best/latest为描述性补充。四个离散Checkpoint不是完整历史梯度轨迹，32个Batch不是独立Seed。负cos不是持续性能损害、大norm不是有害因果证据。
- 梯度有效分母：只使用共同参数Tensor非None、有限、双方共同子空间norm非零的valid pair；缺失、零、非有限分别统计。无HSCL分类头梯度记N/A，不画0cos。
- 计划9图：辅助Loss×Checkpoint cosine分布；reference加权辅助norm/分类norm；参数组冲突频率及有效n；五种关系伪标签0/1比例；情绪×关系伪标签比例heatmap；同Batch XP04 best→latest配对cos变化；默认0/both-wrong及条件分支覆盖；正负对数量/有效anchor覆盖；actual与reference贡献/尺度摘要。各图独立承载证据，不以合成结果替代真实数据。
- 语义：伪标签0/1是模型关系标签，不是情绪类别；情绪类别轴单独标注。epsilon、n/nc权重、fallback与有效分母来自实际实现和计数，不根据图猜测。
- 导出：单面板为主，约180–220mm宽，PNG至少180dpi及SVG可编辑文本，色盲友好配色，轴外图例留空，标注有效n及分母定义。发布CSV仅per-Batch或聚合，禁止原始对话ID/逐话语值/private路径内容。
- 统计：描述性分布、均值/中位数/分位数、冲突频率与同Batch配对差；不做p值或显著性检验，不将Batch当Seed重复。
- 门禁：state完成、128case齐全且独立verification通过后才可生成图与正式结论；脚本测试使用synthetic fixture，不运行绘图main。manifest绑定所有实际输入与输出SHA256和底层CSV。
- 复核风险：reference权重0.002/0.15/0.15与actual分开；XP12 HSCL actual=0，即使离线可求raw梯度也不声称训练优化了HSCL；诊断不optimizer.step、不训练、不测试；采样代表性、同权重eval模式与历史训练模式差异必须保留。

描述性优先级筛选在实测前固定：仅当同一辅助项在XP04 best/latest两个快照上均同时具有最高shared_backbone负cos频率和最高reference加权norm/分类norm中位数，且三项均有32个有效Batch，才建议优先核验该项尺度/方向假设；否则报告不足以唯一优先某项。此为检查顺序，不构成降系数可获益、机制有害或统计显著的结论。

core原pair状态ok/partial_support/partial_nonfinite在共同支持范数有限、非零且cos有限检查后由分析副本显式认证为valid；source_status和全部排除计数保留，原CSV不改，缺失/零/非有限不补造0cos。
