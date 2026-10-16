# 可直接重跑的有限验证命令

所有命令在 `/data2/yb/paper/lyx-msa/CMERC` 执行。无安装依赖、80/40 epoch、完整教师训练或 seed 矩阵。脚本只运行合成检查与真实训练/验证特征的少量 batch，不使用正式测试集。

完整重放包装器会创建新的 UTC 时间目录，保留本轮原始日志：

```bash
bash reports/causal_review_followup/reproduce_checks.sh
```

本轮 GPU 为本地 `user23` 宿主机3，UUID `GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456`，进程内为 `cuda:0`。在重跑前核实空闲和健康，其他机器不能直接继承卡号；biggpu 禁用宿主 GPU4。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv
RUN_CUDA_CHECKS=1 CHECK_GPU=3 bash reports/causal_review_followup/reproduce_checks.sh
```

本轮实际使用的独立命令如下；不要把已有报告目录中的原始失败日志删掉以获得“全绿”。

```bash
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c \
  'import sys;sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages");import pytest;raise SystemExit(pytest.main(["-q","tests","-ra"]))'

CUDA_VISIBLE_DEVICES=3 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c \
  'import sys;sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages");import pytest;raise SystemExit(pytest.main(["-q","tests","-ra"]))'

CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  scripts/verify_causal_regression.py --output-dir /tmp/cmerc-cpu-causality-replay

CUDA_VISIBLE_DEVICES=3 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  scripts/verify_causal_regression.py --device cuda:0 --output-dir /tmp/cmerc-gpu-causality-replay

CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  scripts/verify_training_integration.py --device cpu --output-dir /tmp/cmerc-training-replay

CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  scripts/verify_training_integration.py --device cuda:0 --output-dir /tmp/cmerc-training-replay

CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  scripts/verify_real_feature_followup.py --output /tmp/cmerc-real-cpu-replay.json

CUDA_VISIBLE_DEVICES=3 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  scripts/verify_real_feature_followup.py --device cuda:0 --output /tmp/cmerc-real-gpu-replay.json
```

上面的 `/tmp` 输出路径用于新重放，与本轮证据独立。GPU 默认前向/cuDNN 与局部 ATen 梯度路径在 JSON 中分别记录。恢复脚本显式启用 deterministic warn_only 并保留不支持算子的警告；不改变生产默认，也不持续放宽容差。

日志索引：`baseline_tests.log` 为修改前153通过；`final_cpu_tests.log` 为374通过/4 CUDA跳过；`final_gpu_tests.log` 为378通过/0跳过。原始 CUDA device、provenance、run collision、架构与后续恢复边界的 red 证据都保存在 `logs/`。最初 GPU 因果梯度失败输出在 `gpu/`，修复后通过输出在 `gpu_fixed/`；两者都留存。

真实 CL 未执行：没有合规真实教师，difficulty 文件只在测试夹具中生成。可用的真实特征仅核查和选取训练/验证对话，生产维度训练各2 batch、验证各2 batch，随机初始化诊断权重不保存为正式教师。
