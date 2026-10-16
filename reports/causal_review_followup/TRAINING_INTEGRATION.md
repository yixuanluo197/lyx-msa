# 真实训练函数集成与恢复验证

本节是合成特征上的真实 `train_cm.train_or_eval_graph_model` 验证，不是数据集性能实验。生产 `Model` / `Model_Intra` / `Model_Inter`、原分类/CC/MSCL/CSCL 损失及 HSCL 标签分支均实际执行，无 mock。

## CPU 运行

`TRAINING_INTEGRATION_cpu.json` 保存逐 batch 原始数值、标签、各参数更新范数、四项损失未加权/乘系数后的梯度范数。固定 seed=25、split_seed=12、float32，四路 1024 维文本、audio=6、visual=5、D_g=8、graph_h=8、图深度 num_L=num_K=2，长度 3 和 2，batch_size=2。每配置至少两个 optimizer step；auxiliary 参数没有梯度且不在 optimizer 中。所有真正参与该次损失的梯度有限，主模型发生更新；不要求未执行的旧 BN/hidfc 参数获得梯度。

16 个配置全部通过：两个 auxiliary policy × dropout=0/.15 × baseline/CC-only/HSCL-only/CC+HSCL。以下 CC/MSCL/CSCL 为函数计算的原始分量；相应开关关闭时即使内部产生诊断分量，也不会加入总损失。

| policy | dropout | enabled | classification | CC | MSCL | CSCL | batch 2 main/aux max weight diff |
|---|---:|---|---:|---:|---:|---:|---:|
|legacy_epoch_snapshot|0.0|baseline|1.402588|0.000000|0.000000|0.000000|0.00100005|
|legacy_epoch_snapshot|0.0|CC|1.402590|0.053915|41.460627|24.875426|0.00100005|
|legacy_epoch_snapshot|0.0|HSCL|1.402830|0.049097|41.452234|24.869830|0.00100005|
|legacy_epoch_snapshot|0.0|full|1.402741|0.051970|41.454378|24.871261|0.00100005|
|legacy_epoch_snapshot|0.15|baseline|1.425099|0.000000|0.000000|0.000000|0.00100005|
|legacy_epoch_snapshot|0.15|CC|1.425046|0.875984|17.170148|8.681391|0.00100005|
|legacy_epoch_snapshot|0.15|HSCL|1.425144|0.873052|17.189469|8.694271|0.00100005|
|legacy_epoch_snapshot|0.15|full|1.425101|0.876955|17.189519|8.694305|0.00100005|
|same_step_same_mode|0.0|baseline|1.402588|0.000000|0.000000|0.000000|0|
|same_step_same_mode|0.0|CC|1.402590|0.035354|33.170404|16.585202|0|
|same_step_same_mode|0.0|HSCL|1.402830|0.035012|33.164808|16.582404|0|
|same_step_same_mode|0.0|full|1.402741|0.035035|33.166239|16.583119|0|
|same_step_same_mode|0.15|baseline|1.425099|0.000000|0.000000|0.000000|0|
|same_step_same_mode|0.15|CC|1.424956|0.035177|33.165781|16.582891|0|
|same_step_same_mode|0.15|HSCL|1.425144|0.035117|33.162411|16.581205|0|
|same_step_same_mode|0.15|full|1.425079|0.035118|33.162344|16.581172|0|

## 连续训练与恢复

实际执行连续两个 epoch（分别 2 和 4 个 batch）以及 epoch 0 保存 best/latest 后重新构造模型、Adam 并恢复的对照。使用正常 dropout=.15、CC+HSCL，课程比例从 .5 到 1；真实 `select_curriculum_keys` 决定合成对话的 batch，合成 manifest 精确覆盖四个训练 ID，另有独立验证 ID。源 teacher 是明确标注的合成元数据夹具，不是训练得到的教师，不能用于正式实验。

CPU 模型权重、Adam 状态、epoch 损失日志、逐 batch 诊断、HSCL 离散标签和 next-step loss 完全相同（max_abs_diff=0）。Python、NumPy、CPU torch RNG 均逐项检查恢复；CUDA RNG 单独在可用 GPU 上检查，CPU 运行明确标记该项跳过。课程 next_epoch_ratio=1。当前训练入口每次函数调用依原逻辑重置随机种子；测试重现的就是这一现有协议。DataLoader 的 epoch 顺序由固定训练 seed 确定，split_seed 独立记录。

best/latest 都保存 best_epoch/metric；latest 另固定 best 文件 SHA。从 best/较早 checkpoint 恢复会创建新分支，复制该 checkpoint 之前的 metrics 前缀；导入 best 保持 weights/optimizer/RNG，重新绑定分支 manifest，并保存源 checkpoint 路径、SHA 和源 manifest SHA 的 ancestry。后续没有改进时仍可追踪原 best。已测试连续两次 best 恢复。

managed run 恢复会检查 manifest 哈希、完整科学配置和数据/难度身份、历史 best 的存在及内容、完整且连续的 metrics 历史。非 resume 使用独占创建，已有 run 安全拒绝。总 epoch 只允许保持或延长且必须有下一 epoch；不允许缩短预算。低层 `load_training_checkpoint` 也在修改模型前检查请求的 difficulty 文件 SHA、run manifest 文件/hash/config、分支 parent checkpoint 及 epoch budget；历史 metrics/best 的综合检查由 `prepare_run` 执行。

## GPU 与后端

原 GPU integration 恢复检查曾因逐位日志差异失败，原始日志 `logs/training_integration_gpu.log` 保留。重跑脚本预先设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，仅在 CUDA audit 中启用 deterministic algorithms。严格模式在 PyTorch 2.2.2 的 `cumsum_cuda_kernel` 遇到无确定性实现错误（原日志 `training_integration_gpu_deterministic.log`），因此后续采用显式 `warn_only=True`，不支持的算子警告保留，兼容算子仍使用确定性规则。这是审计设置，不更改生产训练默认。GPU 独立产物为 `TRAINING_INTEGRATION_cuda.json`。CPU 要求 exact；GPU 在执行前固定 atol=2e-5 / rtol=1e-5，另要求离散标签/模式/结构相同。脚本先保存各差异和通过/失败状态，再返回退出码，不通过不会丢弃数值。

same_step_same_mode 的 GPU eval RNN 梯度最初复现 `cudnn RNN backward can only be called in training mode`，红日志为 `training_gpu_same_step_red.log`。最小修复只让该 policy 的 confidence full reference 和五个 views 使用同一 ATen eval 后端；classification 正常 cuDNN train 后端不变。新增 GPU backward 回归实际通过。没有把 eval dropout 重新打开来绕过此错误。

GPU warn_only 最终运行的 16 个配置全部通过；连续/恢复模型权重、Adam、epoch logs、batch diagnostics、next-step loss 和 next-step diagnostics 的 max_abs_diff=max_rel_diff=0，均为 exact，而不只是容差内。Python、NumPy、CPU torch、CUDA torch RNG 四项恢复检查均为 True。结果仅代表该设备/后端/审计设置；`cumsum` 的非确定性实现警告没有被隐藏，未声称任意 GPU 配置都逐位确定。汇总见 `TRAINING_INTEGRATION.json`，逐 batch 见 CPU/CUDA 独立 JSON。

## 执行命令

在 CMERC 根目录：

```bash
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python scripts/verify_training_integration.py --device cpu
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python scripts/verify_training_integration.py --device cuda:0
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c 'import sys;sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages");import pytest;raise SystemExit(pytest.main(["-q","tests/test_followup_training.py"]))'
```

GPU 编号为本地本次审计示例；在其他服务器启动前须重新核实健康设备和空闲资源。

## 数据边界和阻塞

本节没有读取正式测试集，没有执行完整 CL 预训练、80/40 epoch 实验或种子矩阵。随机初始化只用于有明确标签的合成流程夹具，未冒充正式因果教师。真实 CL 仍被缺少经过验证的因果 baseline teacher/difficulty 阻塞。真实训练/验证特征小规模验证见本目录另附的 root 数据审计产物；这里的合成通过不能替代真实特征通过。原始特征提取链仍未审计。

## 最终真实特征与全套检查（leader补充）

同一固定 split_seed=1475、训练 seed=1937，在 IEMOCAP/MELD 各选2个训练和2个验证对话。数据字节哈希、完整 train/valid IDs、具体被选 ID、speaker、顺序和 mask 均在 `REAL_FEATURE_AUDIT.json` 与最终 CPU/GPU JSON。未选择测试对话或计算测试指标。

| device | dataset | train dialogue lengths | valid dialogue lengths | D_g / audio / visual / graph | train batches / valid batches | prefix log-prob max_abs | prefix hidden max_abs |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| cpu | IEMOCAP | [8, 18] | [18, 26] | 512/1582/342/512 | 2 / 2 | 2.38418579e-07 | 5.06639481e-07 |
| cpu | MELD | [3, 3] | [3, 3] | 1024/300/342/512 | 2 / 2 | 1.1920929e-07 | 4.76837158e-07 |
| gpu | IEMOCAP | [8, 18] | [18, 26] | 512/1582/342/512 | 2 / 2 | 2.38418579e-07 | 4.76837158e-07 |
| gpu | MELD | [3, 3] | [3, 3] | 1024/300/342/512 | 2 / 2 | 1.1920929e-07 | 1.78813934e-07 |

真实训练使用生产默认维度与正常 dropout（IEMOCAP=.5、MELD=.4），IEMOCAP 为生产 class-weighted NLL，MELD 为原 FocalLoss。实际函数计算分类、CC、MSCL、CSCL，有限梯度与真实更新均通过；辅助不被优化。随机初始化只用于这些少量诊断 batch，没有保存或使用为正式教师。四项 loss 的值和逐项梯度均在最终 JSON，非正式性能结果，不跨设备比较分数优劣。

本轮最终 full tests：CPU **374 passed / 4 CUDA skips**；GPU **378 passed / 0 skips**。原始日志/JUnit见 `logs/final_{cpu,gpu}_tests.*`。原环境未升级，无完整主实验。
