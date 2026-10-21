# 环境与资源预检

审计日期：2026-10-21 UTC。模型运行归属为本地 `user23`，沿用 XP04；未迁移到 biggpu。独立实验根目录为 `/data2/yb/paper/lyx-msa/CMERC/experiments/official_readme_reproduction/`。

## 1. 环境核验

| 项目 | README 要求 | 本次实际环境 |
|---|---|---|
| Python | 3.10.13 | 3.10.13 |
| PyTorch | 1.13.1 | 1.13.1+cu117 |
| CUDA runtime | 11.7 | 11.7 |
| torch_geometric | 2.4.0 | 2.4.0 |
| torch-scatter | 2.1.0 | 2.1.0+pt113cu117 |
| torch-sparse | 0.5.15 | 0.6.15+pt113cu117，存在版本差异 |
| cuDNN | 未给出 | 8500 |
| NumPy / scikit-learn | 未给出 | 1.26.4 / 1.3.2 |
| pandas / SciPy | 未给出 | 1.5.3 / 1.11.4 |

解释器：`/data2/yb/conda_envs/cmerc-xp-repro/bin/python`。未安装新依赖，未修改共享环境。复用的 Torch 环境已含 XP04 的 UUID/NVML 枚举补丁；实际文件路径和哈希保存在 `provenance/environment.json`。该补丁仅处理新驱动的 UUID 枚举，不能据此声称环境与作者机器逐位等价。

首次导入提示 `libcusparse.so.11` 不在动态库搜索路径，已沿用项目环境脚本的 cuSPARSE 路径：`LD_LIBRARY_PATH=/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib`。记录脚本已实际成功导入 torch_sparse 和 torch_scatter，而非仅查询安装版本。线程设置均为1，正式运行不设置 `PYTHONHASHSEED` 或 allocator override，不强制 ATen RNN。

驱动为 575.51.03；`nvidia-smi` 显示的 CUDA 12.9 是驱动支持能力，不能替代 Torch 实际 CUDA 11.7 的版本记录。

## 2. GPU 与其他任务

本机4张 NVIDIA TITAN Xp，每张12 GiB。预审计时 GPU0/1 有本用户其他项目任务，GPU2 由其他用户训练占用约11.6 GiB，GPU3空闲。没有停止、修改或迁移任何其他项目进程。完整 GPU 占用、进程所有者与命令快照见 `provenance/environment.json`。

本批仅使用宿主 GPU3，UUID `GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456`；通过 UUID 设置 `CUDA_VISIBLE_DEVICES`，进程内映射为 `cuda:0`。本机没有宿主GPU4；biggpu坏卡规则不被错误套用到本机。

GPU 资源动态检查及本批 PID 保存于 `logs/resources.jsonl`、`state/status.json`；后者是运行实时状态的依据。监督器只派发这4个已登记任务，最多两项并发，并在启动前复查显存、GPU UUID及未知进程；遇到外部占用只暂停新派发，不终止外部任务。已有同名正式输出则拒绝自动重复训练。

同卡并发先从两组 Baseline 开始：IEMOCAP 首 Batch 实测 allocated 5314.78 MiB / reserved 6356 MiB，MELD为449.74 / 580 MiB；实际整轮运行中缓存还会增大，因此单 Batch 读数不能当作正式峰值。当前并发的整卡占用和吞吐由持续资源记录及日志核验。后续派发预留 IEMOCAP Baseline 8000 MiB、CMERC 9500 MiB，MELD 2500 MiB 的启动门槛；门槛是资源估计，不是更改 batch size 或科学参数。

`/data2` 审计时约余93 GiB，可容纳本批独立权重和日志；未复制或覆盖历史大数据和 XP04 checkpoint。

## 3. 数据与课程输入

4个实际数据文件共约491 MiB，独立副本通过符号链接指向既有数据；`provenance/inputs.json` 保存真实路径、大小、SHA256、完整有序对话ID及覆盖检查。

| 数据集 | 官方训练对话/句数 | 测试对话/句数 | 作者难度条目 |
|---|---:|---:|---:|
| IEMOCAP | 120 / 5810 | 31 / 1623 | 120 |
| MELD | 1152 / 11098 | 280 / 2610 | 1152 |

逐对话核查四路文本、音频、视觉、speaker 与标签长度，异常数为0。训练/测试对话无交集。MELD raw.trainVid 为1152，等于 RoBERTa metadata train1038 + valid114；本次保留这一官方行为。

作者难度文件来自固定官方 commit，训练键缺失0、额外0、测试ID交集0，数值全部有限；来源 SHA256 见报告01和 `provenance/inputs.json`。本次选择 README Quick Start 的现成难度来源，4组正式模型仍从随机初始化训练。未使用 XP04 或作者 checkpoint 初始化正式模型，未重新计算难度，更未使用测试标签生成训练难度。难度文件生成历史未知，不能证明不存在间接测试选模依赖。

## 4. 原始错误与兼容补丁

| 尝试 | 结果与证据 |
|---|---|
| 原始 git 字节副本 | `topk` 内部导入失败；`logs/preflight_original_IEMOCAP_baseline.log` |
| 仅去除未使用 topk 导入 | 首 Batch 报 `Encountered tensor with size 2607 ... expected size 917`；`logs/preflight_importfix_IEMOCAP_baseline.log` |
| 两项兼容修复 + 外部观测 | 四组均完成一个真实 Batch 的前向、反向及 Adam step；见各 `logs/preflight_compat_*/status.json` 和 `single_batch.json` |

`official_code/` 始终保持官方原始字节；只在 `runtime_code/` 删除未使用导入，并将 HypergraphConv 第二次 propagate 的 size 改为与原 incidence 两个索引轴一致的 `(num_nodes, num_edges)`。`flow`、边、权重、聚合和 Loss 不变。完整 diff 在 `provenance/compatibility.patch`，分阶段 diff 也保留。

PyG 2.4.0 `_collect` 根据 flow 决定 i/j，但 size 仍对应 edge_index 的两个轴；与实测异常一起支持接口修复：[PyG 2.4.0 官方 MessagePassing 源码](https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/nn/conv/message_passing.html)。修复后 CPU 非方形 incidence 测试返回每节点一个向量，并获得有限输入梯度，证据 `provenance/hypergraph_compatibility_test.json`。原实现无法执行，故不主张与其存在可测的逐位数值等价。

外部观测器 `scripts/observe_run.py` 通过 trace 读取真实局部变量、复制 detached CPU 输出，并在评估完成后写 checkpoint；没有替换原训练函数、额外调用随机前向、修改训练集、loss或优化器。保存 checkpoint 属于新增工程功能；整体标记为 `compatibility_patched_with_external_observer`，不称“完全未经修改运行”。cuDNN RNN内部dropout缓存不是一般 RNG state_dict 的一部分，因此本批即使保存 RNG 也不承诺任意进程重启可逐位恢复；不会静默切换 ATen。

## 5. 预检范围与界限

四组首 Batch 的分类、CC、MSCL、CSCL均有限；Baseline 正则关闭，分项为0。所有非空梯度有限，Adam step 已完成。首 Batch 完整 CMERC 分项保存在 `single_batch.json`，不作为正式性能结果。

IEMOCAP课程各阶段固定顺序 Batch 的最大超边数依次为835/874/857/846/930/932/917，测试最大909，均小于1000；MELD各课程池即使任意排列，最坏16对话的超边上界为415/418/418。节点索引也未超过5200。证据 `provenance/curriculum_capacity.json`。不能以关联数 `6N` 大于5200就认定EW参数越界，实际索引是节点索引。

这些检查证明已具备正式运行条件，不证明能达到论文F1，也不替代完整80/40轮的损失、数值和资源验收。正式状态、失败记录与结果由报告04–06和实时状态单独给出。
