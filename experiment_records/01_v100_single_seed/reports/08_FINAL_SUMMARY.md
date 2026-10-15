# CMERC 复现最终总结

状态：**正式从头训练进行中，尚不能宣称整个任务完成。**

## 已完成

- 固定并审计官方 CMERC 源码版本和 M³Net 参考版本；论文 PDF 已归档。
- 建立独立 CPython 3.10.13 / PyTorch 1.13.1+cu117 / PyG 2.4.0 环境，并通过 V100 GPU 图网络 forward/backward。
- 从官方来源核验 4 个真实特征文件、2 个 difficulty 文件和 2 个 checkpoint；完成哈希、维度、类别、speaker、对话和 split 审计。
- IEMOCAP 与 MELD 真实数据 Smoke Test 完成 forward、loss、backward、Adam 更新、有限值检查和完整 checkpoint 恢复。
- 官方 checkpoint 的 W-F1 与 CEL 均复现到论文报告精度；预测、置信度、混淆矩阵和五路信息移除输出已保存。
- 修复 PyG 2.4 API 兼容、difficulty 批聚合、显式 validation 拆分、分项损失日志、checkpoint/恢复和两种选模协议；未改变模型数学定义。
- 正式 baseline/CMERC × official/validation 训练已启动，使用官方 epoch、batch size、seed 和损失系数。

## 当前未完成

- 8 个正式从头训练 run 尚未全部结束；真实最终指标不能提前填写。
- 正式 checkpoint 的独立分类/CEL/校准评估等待训练结束。
- 五随机种子与可选消融尚未执行，不能标记为已完成。

## 当前可复现入口

环境、数据、Smoke Test、官方 checkpoint 评估、单 run 训练、队列训练和恢复命令见仓库根目录 `REPRODUCTION_GUIDE.md`。正式训练一条命令示例：

```bash
CUDA_VISIBLE_DEVICES=5 CMERC_PYTHON=/data2/yb/conda_envs/cmerc-repro/bin/python \
CMERC_OUTPUT_DIR="$PWD/runs/formal" \
  bash scripts/with_cmerc_env.sh bash scripts/run_training.sh \
  IEMOCAP cmerc validation --num-workers 0
```

本报告会在所有进程退出、预期 epoch 数、`best.pt`、独立 `metrics.json` 与汇总结果均核验后更新为最终状态。
