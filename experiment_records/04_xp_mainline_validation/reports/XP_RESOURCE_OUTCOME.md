# XP 主线资源终态汇总

本报告只汇总已验收批次的资源证据，不再次进行科学验收，不改变训练、配置、队列或登记。

科学源码摘要沿用验收：`fafddfee606270f5626a01e3611ecebcdc8ce2af72536ada3508fe31316481a9`。

20个有效run共1200条epoch记录、20次有效训练、20次独立评估、40份进程显存侧车。另保留1次因GPU竞争失效的MELD3407 partial，因此正式训练实际共21次启动。

正式批次墙钟：2026-10-16T05:29:19.061699+00:00 → 2026-10-16T10:30:45.569280+00:00，18086.51 秒（5.0240 小时）。起点为旧pipeline开始，终点为新formal完成；含资源等待、drain和恢复，排除预检与最终验收/导出。

下表elapsed_seconds来自训练JSONL的epoch计时，包含计时范围内的数据加载、训练和validation；checkpoint保存及最终独立评估不在该epoch计时内。

| Dataset / Variant | epoch总数 | 总秒数 | 每epoch均值（秒） | 最短 / 最长（秒） |
| --- | ---: | ---: | ---: | ---: |
| IEMOCAP / baseline | 400 | 2782.93 | 6.9573 | 6.40 / 8.37 |
| IEMOCAP / cmerc | 400 | 15551.61 | 38.8790 | 9.42 / 46.11 |
| MELD / baseline | 200 | 1258.51 | 6.2926 | 5.80 / 9.02 |
| MELD / cmerc | 200 | 7060.15 | 35.3008 | 17.91 / 41.43 |

阶段起止wall time含模型初始化、IO及调度器poll尾延迟。下表相加的是进程墙钟；并发进程会重叠，不能称为GPU kernel hours，也不能替代整批墙钟。

| Dataset / Variant / Phase | 进程墙钟总秒数 | 每run均值（秒） | allocated峰值 MiB | reserved峰值 MiB |
| --- | ---: | ---: | ---: | ---: |
| IEMOCAP / baseline / training | 3084.71 | 616.94 | 8148.09 | 10440.00 |
| IEMOCAP / baseline / evaluation | 91.70 | 18.34 | 2135.18 | 2452.00 |
| IEMOCAP / cmerc / training | 15903.79 | 3180.76 | 9105.19 | 11210.00 |
| IEMOCAP / cmerc / evaluation | 122.68 | 24.54 | 2135.18 | 2452.00 |
| MELD / baseline / training | 1594.98 | 319.00 | 854.18 | 1862.00 |
| MELD / baseline / evaluation | 93.12 | 18.62 | 467.43 | 612.00 |
| MELD / cmerc / training | 7444.34 | 1488.87 | 952.03 | 1738.00 |
| MELD / cmerc / evaluation | 76.32 | 15.26 | 467.43 | 612.00 |

原pipeline退出130是drain后有意停止旧控制器。归档MELD3407旧attempt退出-15，因同卡竞争不纳入有效epoch/阶段合计，保留原权重与日志，原seed/配置从头重试。IEMOCAP2024原训练成功，第一次独立评估在启动前被资源门阻止，恢复时只评估同一best一次。

整卡离散观测仅覆盖XP3：2026-10-16T08:46:46.675631+00:00 → 2026-10-16T10:31:56.567573+00:00；请求间隔5秒，实际样本间隔见JSON，共1239条、有效GPU记录1239条。
采样错误0条，CSV解析错误0条，unexpected PID样本0条；异常明细保留在JSON。
采样整卡used峰值 10959 MiB，free最小 1225 MiB，free最大 12179 MiB。
此observer在正式批次后段才启动，且是离散采样；即使unexpected PID为0，也不证明未覆盖时段或两次采样之间全程CUDA context独占。整卡used/free与进程allocated/reserved含义不同，不能互相替代。

逐run阶段起止、40份显存峰值、时间/单位定义及全部来源SHA256见 `../provenance/resource_outcome.json`。
