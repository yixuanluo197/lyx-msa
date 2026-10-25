# 第09组资源准入修订 v2（GPU启动前）

修订时间：2026-10-24T16:46:22.148737+00:00

用户要求继续完成XP卡实验。原GPU3并未故障，环境兼容；原准入门禁固定要求free>=5000MiB/util<=65且仅接受已经结束的PID1843864。该门禁过于保守，误把PID更替本身当作持续资源冲突，造成长期排队。

本修订仅涉及执行资源管理。在原监督器仍为preflight_admission且探针0/正式0/不存在run输出时，核验/proc argv和uid后，仅停止本任务三个无GPU的CPU监督器；不触碰其他项目。原配置、监督脚本、冻结清单、全部队列日志和状态原字节保存在provenance/resource_gate_v1/及其archive_manifest.json。没有重启已训练模型，也没有按成绩重跑。

## 新资源门禁

仍使用本地user23同一张TITAN Xp GPU3 UUID GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456，进程可见cuda:0只代表此卡。启动需3个新采样、间隔10秒，free>=3000MiB/util<=95%、无本任务GPU子进程。PID2527571/2560936登记为本次检查时已有任务，新增PID继续分类和计数，但不以PID更替本身拒绝；资源阈值和身份核查依然必须通过。查询失败闭门，不终止任何外部进程。预检和正式启动之间确认GPU释放并重新采样。

08同环境B16完整运行实测max_allocated=939.482MiB、max_reserved=1630MiB；这与nvidia-smi整进程显存（还含CUDA context等）不同，不混称同一指标。B8的CPU最大超图167边/429节点，小于B16的300/756；这只是容量依据，不单独保证完整训练峰值。采用2200MiB本任务估计预算和至少800MiB余量，先执行独立首Batch前后向+真实Adam更新探针，记录allocated/reserved和整卡资源；若预检不满足则暂停，不擅自减Batch或改精度。共享会影响耗时和各任务吞吐，不宣称独占速度或零算力竞争。

## 科学协议保持原登记

registry.runs与environment逐字结构相等：MELD、完整非因果CMERC、seed2024/split1475、LR0.0001、Batch8、40轮、Adam/L2=3e-5、dropout0.4、continuous/ATen/capacity_safe、BN/hyper/direct/concat_DHT/L3/K3，全部CC/MSCL/CSCL及CL系数原值。训练源码12文件、模型、Loss公式、数据/划分/课程、采样、选模与零测试门禁不变。新冻结清单明确继承旧科学证据，资源工程Diff单独保存。旧登记文字中的PID1843864/5000/65是v1历史条件，本修订替代资源部分，不改变科学部分。

本阶段仍仅1个正式40轮run，探针不向正式初始化提供权重/Adam/RNG；验证首最高选模、平局最早；不执行测试前向、不追加Seed、LR或Batch搜索。所有04～08保持原样。
