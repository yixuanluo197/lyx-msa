# 本阶段运行与恢复记录

- 运行归属为本地user23，宿主GPU3（GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456），进程内cuda:0。仅使用XP04原环境，无依赖升级。
- 首次来源预检在模型前向前停止：历史multiseed_scheduler.py与origin初始记录不同。它不参与本批执行；保留差异证据，严格核验实际导入的12份科学源码。未修改该历史调度脚本。第二次完成全部十个GPU对齐。
- 原IEMOCAP Audio进程因原图结构反复构建耗时而受控SIGTERM，主控执行会话返回shell exit143。未产出完整扰动NPZ。保留原日志，不按结果触发重跑。
- 以等价图索引缓存入口完整重跑IEMOCAP Audio，并用于随后Visual和Text Representation。仅缓存结构，特征每次重新计算，全部76次完整输出对照通过。CPU profile及等价证据见provenance目录。
- IEMOCAP Audio使用wrapper初版：显式log-probability容差和argmax检查，并记录最大概率误差；所有最大概率误差小于atol=1e-6。其后wrapper增加显式完整概率allclose；数值运算不变。原版保存为provenance/run_cached_pilot_audio.py。
- 三种类型和两数据集全部按Audio→Visual→Text Representation顺序执行，随机5次、衰减1次。没有联合模态、其他seed扰动、CMERC训练、温度拟合或测试集调参。
- 6个有效Pilot累计进程内目标前向计时1941.32秒，同卡并发有重叠，不是GPU kernel时间或总墙钟。对齐阶段的peak_allocated字段为该进程累计峰值，不能解释成各checkpoint独立显存；Pilot在每run计时前重置峰值。
- 27项单元测试通过。最终verification另从原数据与保存概率核对身份、train-only统计、行为指标、分类指标及固定OOF bootstrap区间。
- 数据质量附加检查发现IEMOCAP 1条、MELD 95条Validation视觉表征全零；仅记录描述性缺失信息，不删除样本或修改扰动配置。
