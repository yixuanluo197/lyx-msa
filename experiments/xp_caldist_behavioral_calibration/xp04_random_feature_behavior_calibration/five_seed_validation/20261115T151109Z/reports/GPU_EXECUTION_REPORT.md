# 五Seed GPU推理与完整性证据

Seed1475的joint结果复用已核验Pilot；其余4个Seed在同一台本地服务器补齐。没有CMERC训练或权重更新。

| Dataset | Seed | GPU | UUID | Seconds | Peak allocated MiB | Probability parity max error |
|---|---:|---:|---|---:|---:|---:|
| IEMOCAP | 67137 | 0 | GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248 | 950.33 | 1118.3 | 2.68e-07 |
| MELD | 67137 | 1 | GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a | 371.27 | 203.2 | 2.38e-07 |
| IEMOCAP | 2024 | 0 | GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248 | 951.66 | 1118.3 | 2.38e-07 |
| MELD | 2024 | 1 | GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a | 368.15 | 203.2 | 1.79e-07 |
| IEMOCAP | 3407 | 2 | GPU-d17e4326-9d3b-2025-fbf8-c9b0d16623a0 | 684.76 | 1118.3 | 3.58e-07 |
| MELD | 3407 | 2 | GPU-d17e4326-9d3b-2025-fbf8-c9b0d16623a0 | 633.91 | 203.2 | 1.79e-07 |
| IEMOCAP | 42 | 3 | GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456 | 691.83 | 1118.3 | 3.28e-07 |
| MELD | 42 | 3 | GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456 | 610.61 | 203.2 | 2.38e-07 |

总计130880次新增目标前向；包含复用Seed1475的32720条joint记录后，完整五Seed行为诊断共有163600条扰动记录。

每batch保持原mixed-dialogue batch16，核对uncached/cached基线及扰动前向；每Dialogue核对目标话语之外的全部特征和mask不变，每batch结束恢复原张量，最后完整state_dict精确相同。

正式数据加载器会反序列化含多划分条目的原特征容器，但只使用真正Validation keys索引话语；Test标签不用于推理评价、校准拟合或配置选择。

两任务/卡的容量试运行无OOM，显存留有余量。运行期同卡并发增大总吞吐，但每任务速度受共享资源影响，不能作为模型性能速度比较。详见job_registry和resource_monitor。
