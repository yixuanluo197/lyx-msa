# XP13：辅助损失梯度冲突与伪标签质量诊断

状态：128个固定权重诊断单元全部完成，独立验收通过；不是新训练实验。

- [完整中文报告](reports/CMERC_AUXILIARY_LOSS_DIAGNOSTIC.md)
- [预登记](reports/PREREGISTRATION.md)、[实际配置](configs/registry.json)、[冻结执行清单](provenance/frozen_execution.json)
- [源代码审计](reports/SOURCE_AUDIT.md)、[Checkpoint CPU验收](preflight/checkpoint_inventory.json)
- [核心结果](results/metrics_summary.json)、[解释与下一步](results/interpretation_summary.json)
- [独立验收](provenance/final_verification.json)、[独立聚合复核](provenance/independent_final_review.json)
- [PNG/SVG及哈希](results/figures/manifest.json)

XP04 e8/e40及XP12 e5/e40，共享32个Batch16（512训练对话、4936话语）。eval+grad，每Batch同步teacher；不是历史在线训练梯度。768次模型前向、512次autograd.grad；训练、optimizer.step、Tensor.backward、新测试前向均0。

CC在两个XP04快照的共享主干负余弦均28/32，后期MSCL加权尺度更大；预登记唯一优先项筛选为insufficient。下一步优先核查原分类目标与CC的置信度条件方向关系，不据本批直接选新系数、断言模块有害或开启五Seed。

公共归档不含权重、原始数据、实际对话ID、逐话语标签/预测/置信度、原始日志及原梯度向量。冻结历史参考原件内含ID，仅保留服务器；公开使用xp04_validation_reference_public.json，原SHA仍在冻结清单。数据软链不入Git。作者公开课程文件保留原SHA，生成历史未知限制不变。

## 服务器复核命令

按configs/commands.json、registry.json准备原环境、权重、外部数据与private固定采样清单。diagnose.py带唯一运行标记，不能直接重复启动本批；重做诊断需新目录且不覆盖当前产物。

本次依次执行prepare_diagnostic.py、冻结、diagnose.py --run、verify_diagnostic.py、analyze_diagnostic.py、plot_diagnostic.py。后续纯编辑/发布步骤为render_figure7_layout.py（修复图例裁切，统计不变）、redact_reference.py、curate_report.py、verify_publication.py。前两项图表/报告原始输出已另存并绑定前后SHA，冻结科学代码未改。

只克隆公开仓库可审查代码和聚合证据；缺服务器权重/数据/私有输入时不能完整重跑。CPU统计测试使用合成数据，不代表额外模型前向。
