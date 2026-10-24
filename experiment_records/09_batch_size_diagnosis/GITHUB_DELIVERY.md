# 第09组Batch Size诊断：阶段报告发布

本次按用户要求发布[中文报告](../../reports/CMERC_BATCH_SIZE_DIAGNOSIS.md)及其审查材料。**这是一份资源阻塞状态报告，GPU探针和正式训练均为0次，不是Batch8性能报告。**

控制已从XP04同Seed归档核实：MELD seed2024、LR0.0001、Batch16，最佳验证66.01%/第9轮，末10轮61.223%。Batch8数值待唯一一次40轮正式训练完成后产生。CPU容量通过不代替GPU前后向或完整训练验收，预计5096更新并非实际训练计数。

## 本次公开材料

- [报告](../../reports/CMERC_BATCH_SIZE_DIAGNOSIS.md)、[独立批次](../../experiments/batch_size_diagnosis/20261024T131956Z/README.md)、[训练前登记](../../experiments/batch_size_diagnosis/20261024T131956Z/reports/PREREGISTRATION.md)
- [协议审计](../../experiments/batch_size_diagnosis/20261024T131956Z/reports/PROTOCOL_AUDIT.md)、[配置和完整argv](../../experiments/batch_size_diagnosis/20261024T131956Z/configs/registry.json)
- [CPU保护测试](../../experiments/batch_size_diagnosis/20261024T131956Z/preflight/guard_tests.json)、[40轮容量重放](../../experiments/batch_size_diagnosis/20261024T131956Z/preflight/capacity_cpu.json)
- 冻结训练源码、工程补丁、观测/启动及CPU核验脚本、源与数据哈希、环境和历史文件保全清单
- [上传时监督器快照](../../experiments/batch_size_diagnosis/20261024T131956Z/provenance/publication_status.json)；实际文件及SHA见[publication_manifest.json](publication_manifest.json)

## 状态及复核边界

快照时间：`2026-10-24T14:13:23.658649+00:00`。服务器后台监督器仍在GPU资源准入队列，本次上传不改变队列、配置或任何04～08历史文件，不重复启动任务，不执行模型前向/反向或测试。

GitHub上的快照和报告不会随服务器队列实时更新；正式训练及独立验收后的结果需要另行发布。模型权重、输入数据及其链接、原始日志、逐Batch跟踪、实时state与锁保留在服务器。冻结清单中的服务器绝对路径及未上传文件SHA仅供溯源，不保证仅克隆公开仓库即可重算全部检查。

唯一科学CLI因素为Batch16→8，原参数同时改变train与validation分组。BN/padding、未pack双向GRU、HSCL配对、CC尺度、更新次数及实际RNG消耗的耦合限制完整保留；不得把本次设计说成仅改变训练分组。未生成新测试F1，不据测试或当前未知结果作参数选择。

发布前CPU复查13项全部通过，CUDA未初始化；39项冻结输入哈希保持不变。原冻结源码及补丁的CRLF/空白按08原字节保留，未做格式化。
