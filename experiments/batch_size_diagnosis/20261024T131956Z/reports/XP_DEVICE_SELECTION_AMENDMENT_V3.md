# XP设备选择修订v3（任何GPU调用前）

时间：2026-10-24T17:36:33.419107+00:00

用户明确询问为何不使用GPU0和1。本机0、1、3均为健康TITAN Xp，将其纳入同一服务器的候选设备。此前固定GPU3只是保持同物理卡的工程选择，不能把它说成模型或XP04科学协议的要求。v2仍为0probe/0formal时，已只停止本次三个CPU监督器并保全原配置/脚本/冻结清单/完整日志和状态到provenance/resource_gate_v2；其他项目未动。

独立CPU selector同时检查GPU0 UUID GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248、GPU1 UUID GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a、GPU3 UUID GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456。任一卡连续3次/间隔10秒通过free>=3000MiB、util<=95、卡号/UUID/型号校验即可候选；同时满足优先空闲显存更多、利用率更低、卡号更小。GPU2当前被其他用户近满载占用，未纳入候选。不迁移服务器，不终止他人任务。

选择发生在任何GPU模型调用前。selector只修改实际设备index/UUID、CUDA_VISIBLE_DEVICES和现场共驻PID披露；科学runs/全部argv、其余environment、源/数据/课程/划分/RNG/结构/Loss不变。选中后保存gpu_selection.json，并把实际registry和选择证据加入最终冻结清单；之后启动原单run监督器。监督器还会分别在探针和正式训练前重新3次检查资源；探针与正式任务使用同一张选中卡，不动态跨卡恢复或复制训练。

XP04 seed2024控制实际在GPU3。本次如果选择0或1，必须报告同服务器/同型号但不同物理卡，不再声称同卡。CUDA scatter非确定性、硬件时点、共享资源限制严格因果与耗时解释。科学比较仍只有原CLI Batch16→8（同时影响train/validation分组），不将硬件执行差异隐藏为完全确定性的单变量数值复现。

资源v2的2200MiB预算、min800MiB余量、有限Loss/梯度/真实Adam更新、零测试、唯一一次40轮规则继续。所有04～08原文件不变。
