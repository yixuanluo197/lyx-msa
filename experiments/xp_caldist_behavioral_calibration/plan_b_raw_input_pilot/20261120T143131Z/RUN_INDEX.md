# 本地交付索引

本轮方案B：单Seed IEMOCAP1475/MELD67137。没有Commit/Push；大型权重、完整特征、媒体和完整文本均仅在local_only/。

## 报告与统计

- BASELINE_REPORT.md：完整Validation的6项指标；results/baseline_metrics.csv及baseline_summary.json。
- RAW_DATA_PERTURBATION_REPORT.md：输入QC、阴性对照、正确/错误μ/δ/JS、置信度与Dialogue条件分析。
- FINAL_PILOT_DECISION.md：实际信号、负向结果、支持量限制及下一步建议；本轮停止。
- verification.json、REVIEW.md：独立实际核验及历史工程勘误；工程核验通过不等于可靠性增量、听审或多Seed通过。
- reports/：数据准备、文本和音频特征提取报告。
- results/pilot/{dataset}/：固定sample_manifest、完整原概率、全部1200个预设版本记录、变换manifest、clean controls、推理状态。
- results/analysis/：全部完整家族、标签保持文本主分析、独立音频QC探索、失败/不完整家族、置信度变化/翻转、正确/错误比较、Dialogue汇总、共享2000次cluster bootstrap与精确匹配单元。事后特征距离单独标记描述性用途。

## 模型、特征和媒体（仅本地）

- local_only/text_models/{dataset}/best_classifier/：各自最佳监督RoBERTa权重、tokenizer/config；latest_training.pt保留完整末次训练状态。
- local_only/models/facebook_wav2vec2_base/：指定公开revision的原始文件；冻结使用，权重键名兼容映射及逐tensor核验见provenance/。
- local_only/features/{dataset}/{text_features.pt,audio_features.pt}：新四层1024维文本与768维音频；Train/Validation及独立缺失标记。
- local_only/data/{dataset}/dataset.pt：仅新训练允许的元数据、原文、媒体指针、标签/Speaker及官方视觉；无旧T/A输入。
- local_only/data/{dataset}/runs/{m3net,cmerc}/{best.pt,last.pt,completion.json}：新下游模型及完整训练末次状态。
- local_only/data/{dataset}/new_difficulty.pt与对应分量记录：新M³Net生成的Fit Dialogue难度。
- local_only/pilot/{dataset}/：保留原上下文的文本候选、署名AI初审与独立二审、冻结text_reviewed.jsonl、音频变换/听审表、预计算CPU特征、目标特征及完整Dialogue概率。原始音频留原位置，变换音频另存，不覆盖。

## 入口与运行环境

编码器固定TEXT_repro（Torch2.2.0+cu118，Transformers4.42.3）；CMERC固定multimodalerc310（Torch2.2.2+cu121，Transformers4.45.1及既有PyG）。不能在CMERC环境重新实例化HF编码器。精确执行命令、PID、GPU UUID/服务器、代码及输入SHA见provenance/*_stage_*.json、baseline锁及模型完成锁。

scripts/text_encoder.py、audio_encoder.py和downstream.py为实际训练/特征入口；三个run_*_stage.py监督训练和Pilot持久运行。原始干扰分两段：TEXT环境extract_pilot_features.py输出V2 CPU FP32缓存，CMERC环境raw_pilot.py infer只消费已绑定缓存。pilot_analysis.py只读取保存概率；describe_feature_response.py只读取保存特征；两个report脚本只生成报告。

独立核验可在本机运行：

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python \
  scripts/verify_plan_b.py --root /data2/yb/paper/lyx-msa/experiments/xp_caldist_behavioral_calibration/plan_b_raw_input_pilot/20261120T143131Z \
  --stage final
```

上述相对scripts路径以本run目录为工作目录；实际核验器也可从根目录使用其绝对路径。训练和Pilot入口拒绝覆盖已有完成锁/输出；复现实验应使用独立run目录并保持现有数据/权重版本，不在当前已完成目录重跑。当前没有后续排队任务。
