# CMERC 复现环境

检查日期：2026-10-13（UTC）。正式模型归属 `biggpu`，baseline、CMERC、随机种子、消融和评测保持在同一服务器。

## 主机资源

### 本机

- Ubuntu Linux 6.8.0，x86_64。
- NVIDIA 驱动 575.51.03，驱动报告 CUDA 12.9。
- 4 × NVIDIA TITAN Xp 12 GiB；检查时 GPU 0/1/3 空闲，GPU 2 满载。
- `/data2` 可用约 1.2 TiB。
- Base Python 3.13.11；不在 base 环境安装本项目依赖。

### `biggpu`

- Ubuntu 24.04.1，NVIDIA 驱动 575.51.03。
- 8 × Tesla V100-SXM2 32 GiB。
- **宿主机 GPU 4（UUID `GPU-46fb379f-cc90-dc82-9b5e-5d011f552264`）为坏卡，所有命令必须排除。**
- 环境验证和首批运行使用宿主机 GPU 5；进程内显示为逻辑 `cuda:0`。
- `/data2` 检查时仅剩约 66 GiB，数据使用已有可信副本，运行产物必须按 run ID 隔离并控制保留量。

## 专用环境

路径：`/data2/yb/conda_envs/cmerc-repro`

| 组件 | 实测版本 |
|---|---|
| Python | CPython 3.10.13 |
| PyTorch | 1.13.1+cu117 |
| PyTorch CUDA runtime | 11.7 |
| torch-geometric | 2.4.0 |
| torch-scatter | 2.1.0+pt113cu117 |
| torch-sparse | 0.6.15+pt113cu117 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| pandas | 1.5.3 |
| scikit-learn | 1.3.2 |

官方 README 写 `torch-sparse 0.5.15`，但 PyPI 没有该版本；可用版本为 `0.5.1` 和 `0.6.15` 等。PyG 官方的 PyTorch 1.13/CUDA 11.7 wheel 索引提供的是 `torch_sparse-0.6.15+pt113cu117`。因此使用 0.6.15，并把 README 条目判定为漏写数字 `6`。该替换不改变 CMERC 数学定义，但属于复现环境偏差，已显式记录。

## CUDA 动态库处理

PyTorch 1.13.1 wheel 自带多数 CUDA 11 库，但远端主机只有系统 CUDA 12.9，首次导入 `torch-sparse` 报错：

```text
OSError: libcusparse.so.11: cannot open shared object file
```

安装完整 Conda `cudatoolkit=11.7` 时，远端混合 channel 的求解器错误选择了 GraalPy 3.10.8，破坏了 PyTorch CPython ABI。该临时环境已删除并以严格 `conda-forge`、精确 CPython build 重建。最终仅通过 PyPI 安装 `nvidia-cusparse-cu11==11.7.5.86`，由 `scripts/with_cmerc_env.sh` 将其库目录加入 `LD_LIBRARY_PATH`。没有修改系统 CUDA 或驱动。

## 实测验证

在 `CUDA_VISIBLE_DEVICES=5` 下运行 PyG `GraphConv(8,3)` 的真实 GPU forward/backward：

```text
CPython 3.10.13
1.13.1+cu117 11.7 2.4.0 2.1.0+pt113cu117 0.6.15+pt113cu117
Tesla V100-SXM2-32GB (4, 3) True grad_norm=0.4604148268699646
```

这证明 PyTorch、CUDA、PyG、scatter/sparse 扩展能够联合加载，输出有限且梯度可回传。模型级 smoke test 另见 `04_SMOKE_TEST.md`。

## 重建命令

```bash
CMERC_ENV_PREFIX=/data2/yb/conda_envs/cmerc-repro \
  bash scripts/bootstrap_environment.sh

CUDA_VISIBLE_DEVICES=5 \
  bash scripts/with_cmerc_env.sh python -c 'import torch; print(torch.__version__)'
```

`scripts/bootstrap_environment.sh` 会创建全新环境；不要对已有重要环境重复运行。环境启动包装器不会改动系统 Python。
