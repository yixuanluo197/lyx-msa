# GPU Resource Report

启动前四张 NVIDIA TITAN Xp 均空闲，无计算进程。正式绑定：IEMOCAP 使用宿主机 GPU 0 / `GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248`；MELD 使用 GPU 1 / `GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a`。GPU 2/3 未使用。两个任务为独立进程，`CUDA_VISIBLE_DEVICES` 隔离，进程内均使用 `cuda:0`；OMP/MKL 各 4 线程，DataLoader worker=0，无 DDP。

| 数据集 | 训练秒数 | GPU-hours | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| IEMOCAP | 1600.31 | 0.4445 | 7.348 | 11.447 |
| MELD | 426.18 | 0.1184 | 0.958 | 1.215 |

合计训练 GPU-hours 为 0.5629。两项并行墙钟由 IEMOCAP 主导，约 26.67 分钟。IEMOCAP 在 12 GiB TITAN Xp 上 reserved 峰值较高，正式训练应按每卡一个 IEMOCAP 任务调度；MELD 显存更低，但仍需根据总吞吐和 CPU 负载逐步验证并发。训练结束后 GPU 0/1 均恢复至 3 MiB、0% 利用率。
