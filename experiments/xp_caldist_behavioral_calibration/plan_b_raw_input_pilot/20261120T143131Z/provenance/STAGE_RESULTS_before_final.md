# Plan B 阶段记录

实验目录：20261120T143131Z；服务器 user23。当前任务不自动提交或推送 Git。

## 数据准备：完成

IEMOCAP Train/Validation=5283/527，MELD=9989/1109。新数据包仅包含原始文本、标签、ID、Speaker、媒体路径与官方视觉特征；无旧文本/音频特征。固定 Train 内部按 Dialogue 划分 Fit/Dev：4747/536、8989/1000。

MELD Train125/utt3 原文件损坏、Validation110/utt7 原文件缺失，恢复检查未找到可用原件。采用零768向量加独立缺失mask，保留所有话语与上下文。Train556/utt6 原CSV文本与旧pkl文本及Speaker记录不一致；使用原CSV文本，按用户要求保留原Speaker ID，并单独披露，不删除该话语。

## 文本监督微调：运行中

真实 Fit 的两步工程检查均通过，主干与分类头梯度有限且参数确实更新。两个正式任务分别从公开 RoBERTa Large 初始化，在 GPU0/2 并行运行，microbatch4、累积4、FP32。各自以 Train 内部 Dev macro-F1 选模型；官方Validation不参与选择。训练和完整断点状态保存在local_only/text_models。持久tmux会话与SHA锁定命令见provenance/text_stage_*.json。

## 音频特征：准备中

公开 facebook/wav2vec2-base 指定revision权重分段下载并验证完整SHA；解码缓存仅覆盖Train与Validation。使用16kHz单声道、整句话归一化、固定30s分块与真实帧加权池化，冻结参数。下载未完成前不启动模型推理。

## 新M3Net、难度与CMERC：等待新特征

冻结核心源码和新输入薄适配已完成，两数据集合成前后向、辅助损失梯度及checkpoint恢复检查通过。新特征齐备后须先通过真实数据small smoke，再正式训练。尚无baseline性能结论。

## 原始干扰Pilot：尚未执行

仅完成合成DSP、μ/δ/JS与target-only不变性检查。正式冻结baseline后才抽取同一批120话语及生成处理版本；情绪保持未确认的版本单列探索分析，不能自动认定标签保持。

## 2026-11-20T15:46Z 更新

两套文本和音频特征均已完成，features独立验证9010项通过；两数据集新M3Net、新difficulty和新CMERC全部训练完成，models验证10156项通过。IEMOCAP文本选第8轮，MELD选第3轮。冻结CMERC的完整Validation指标见BASELINE_REPORT.md。IEMOCAP Acc58.06%/MacroF155.22%，MELD Acc62.85%/MacroF151.20%；没有据此重调训练。固定Train-only真实raw路径smoke IEMOCAP已PASS，MELD检查中；尚未抽取120 Pilot目标。

下游首次环境缺PyG的导入失败已归档并切回现有完整CMERC环境。所有原/扰动编码器仍严格在同一TEXT_repro环境产生CPUFP32特征，再由既有CMERC环境消费；没有跨HF版本重提、未新增依赖。共卡时实际显存不足后，将MELD非训练difficulty移至同机健康GPU1重算，保留已完成M3Net，未更改batch或新增Seed。
