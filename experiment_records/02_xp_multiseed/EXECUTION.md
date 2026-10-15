批次：20261014T1627Z；服务器：本地 user23；GPU 0/1/3 UUID 见 initial_gpu_snapshot.csv。GPU2 不参与。
匹配环境安装命令：CONDA_SH=/home/yangbin/miniconda3/etc/profile.d/conda.sh CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-xp-repro bash scripts/bootstrap_environment.sh
代码来自 91a1bffa763d57015896dee1014f0d11c5d09029；启动前工作区 tracked diff 为空。source.diff 与 snapshot.json 保存初始证据。
preflight.py 的工程7/3 epoch run 覆盖课程100%，仅测资源；不纳入正式12任务。official_test 的validation loader为空，故资源覆盖训练+完整测试；独立评估另走五路信息移除。
profile_run.py 仅观察 torch.cuda.max_memory_allocated 调用时的 allocated/reserved 和墙钟时间；不改模型/损失/随机状态。正式各run使用独立 memory_profile.jsonl。

安装首次失败：/tmp 所在根分区可用约1.1GB，1.8GB torch wheel触发 Errno28。保留原日志，以 TMPDIR=<batch>/tmp 在已有仅Python环境执行 install_remaining.sh，不重复 conda create。模型依赖版本未变。

预检首次GPU失败：torch1.13.1 NVML 的CUDA_VISIBLE_DEVICES解析将UUID变成 -1，runtime设备数1但Python设备数0。复现json与错误日志保留于preflight/failed_uuid_attempt及logs/failed_uuid_*。环境补丁仅让UUID的NVML路径返回 -1，使用既有CUDA runtime设备数回退；torch版本保持1.13.1+cu117，模型与RNG算法未改。补丁命令scripts/fix_torch_uuid_count.py、原源码、diff和hash已保存，单tensor CUDA checkpoint加载验证通过。正式任务尚未开始，scheduler增加实测MELD并发能力后重新冻结helper/hash。

默认分配器完整课程7epoch成功但前缀epoch5 reserved11736MiB，仅余约200MiB context后空间，未通过11200MiB reserved门槛。按PyTorch1.13官方文档验证max_split_size_mb:128，保留默认测量产物，批次所有正式进程统一该分配器配置；不改batch、FP32、损失或样本顺序。正式运行尚未开始，观察器hash已在冻结清单更新。
官方依据：https://raw.githubusercontent.com/pytorch/pytorch/v1.13.1/docs/source/notes/cuda.rst

max_split128 reserved11836MiB未通过；单设GC在torch1.13默认未生效，源码门槛为set_fraction && threshold>0。调用set_per_process_memory_fraction(1.0)保持原显存容量并启用GC，默认split+GC0.8真实0.75课程探测将reserved降为9650MiB。最终候选为GC0.8+fraction1.0，重新做全部7/3课程与checkpoint验证，不将上述工程运行视为正式seed。依据：https://github.com/pytorch/pytorch/blob/v1.13.1/c10/cuda/CUDACachingAllocator.cpp#L581
