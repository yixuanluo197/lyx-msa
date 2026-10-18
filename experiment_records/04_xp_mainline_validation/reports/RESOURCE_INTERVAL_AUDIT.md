# 历史资源区间审计

结论：在本批 queue 快照中已完成的 13 项任务里，没有发现与指定其他项目额外、明确的 GPU 模型训练/推理计算重叠证据。不能据此声称全程经过 NVML 监控的 GPU 独占。

材料生成时间：2026-10-16T08:23:30.467403+00:00。本批 queue 更新时间：2026-10-16T07:46:00.738825+00:00。比较范围仅为 `/data2/yb/paper/experiments/baseline_5models_local_20261015`；不代表审计所有其他项目。

依据本批 29 个已开始的 training/evaluation phase，逐一比较实际 phase 起止、PID、GPU UUID；没有把任务整体开始至结束之间的排队时间当成占卡。对方扫描包括归档失败尝试的 child.json，共 293 份。对方 child/supervisor 生命周期及 registry RUNNING 不等于连续 GPU 运算。

所有下表时间均为 2026-10-16 UTC；对方记录精度为整秒。本批保留原始微秒时间。

| 本批任务 / phase / PID | 物理 GPU | 对方 PID / SDR seed68任务 | 进程区间交集 | 判定 |
|---|---|---|---|---|
| IEMOCAP baseline2024 / training / 3701445 | 0 | 3701459 / mr0.2 旧尝试 | 06:39:45–06:39:47 | 对方 admission guard 失败，未进入模型构建；无模型计算重叠证据 |
| IEMOCAP CM67137 / evaluation / 3746804 | 1 | 3747323 / mr0.2 | 07:26:30–07:26:35.356019 | 本批预测、指标早于对方启动写完；仅确认进程尾部交叠 |
| IEMOCAP CM2024 / training / 3711685 | 0 | 3763549 / mr0.4 | 07:42:55–07:43:03.498145 | final_test 落盘后、训练进程退出前；独立 evaluation 未启动 |
| MELD CM3407 / training / 3764688 | 1 | 3764660 / mr0.5 | 07:43:58.487692–07:46:00.415635 | 已知同卡真实训练竞争，失效 attempt，退出码 -15 |

GPU 0 UUID：`GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248`。GPU 1 UUID：`GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a`。

## 两个已完成任务的新增候选

baseline2024 training 实际区间为 06:39:44.607909–06:49:54.156972；对方旧 child 为 06:39:45–06:39:47。归档 failure.json 的错误为 `Requires exactly one local whitelisted GPU UUID`，executed snapshot 源码显示该错误发生在模型构建前。不能将两秒进程共存算作模型训练并发。

IEMOCAP CM67137 evaluation 实际区间为 07:26:04.856047–07:26:35.356019。对方 child 区间为 07:26:30–07:38:07，manifest 时间为 07:26:32。本批 predictions.npz、predictions.csv、metrics.json 的 mtime 分别为 07:26:21.150560、07:26:21.153560、07:26:21.155560，均早于对方启动。冻结评测代码在保存前已将全部推理结果复制至 CPU，保存之后仅进行 CPU 格式化、输出及退出。因此进程尾部共存不构成额外推理计算并发证据。CUDA context/显存究竟何时释放未被记录。

## 两个已知资源事件的准确表述

IEMOCAP CM2024 training 实际区间为 06:50:14.155701–07:43:03.498145；对方 mr0.4 child 为 07:42:55–07:56:13。本批 final_test.json 已于 07:42:49.994644 写完。准确描述是：**final_test 落盘后、训练进程退出前，对方启动**。不能笼统说当时训练进程已经退出。其独立 evaluation 没有 started_at；不能把后续等待算作 GPU 推理。queue 将该 run 标为 failed，现有 final_test 也不等于独立 evaluation 完成。

MELD CM3407 training 实际区间为 07:43:58.487692–07:46:00.415635；对方 mr0.5 child 为 07:43:57–07:57:27，manifest 为 07:43:59。双方训练日志/历史支持真实 GPU 模型训练共存。本批进程以 -15 退出，该次 attempt 已失效，不能作为已完成正式结果。没有逐 kernel 时间线，无法细分 GPU 调度时隙。

## 证据、边界与复核

机器可读记录为 [resource_intervals_audit.json](../provenance/resource_intervals_audit.json)，保存全部本批 phase 快照、4 个候选的完整区间，以及各证据的绝对原始路径、SHA-256、字节数和纳秒 mtime。JSON 内不复制对方数据或权重。原始证据若后续改变，应以记录的哈希和时间核对，不能把当前文件自动当作历史版本。

没有连续 NVML 进程/显存采样，不能证明全程独占，也不能证明 GPU context 精确释放时间。上面的判定区分进程区间、结果已计算完成和 GPU 运算共存；没有把监督/排队状态直接等同 GPU 计算。此次仅整理材料，没有启动或查询 GPU，没有修改 queue、训练源码、协议或其他项目。
