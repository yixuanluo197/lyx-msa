# Independent Calibration/Evaluation Split Feasibility

## 审计结果

- IEMOCAP 的 Dialogue ID 包含 Session 信息，可按 Session 分组；MELD 的冻结产物含 Dialogue ID，但当前特征/配置审计未建立可靠 Episode 映射。
- XP04 checkpoint 已见过原始 Training Split。把其中样本划作 calibration 或 evaluation 不能称为模型未见数据。
- 当前 Validation 已用于 checkpoint 选择，并已用于多轮校准方法探索，因此适合作探索性 OOF 分析，不足以承担最终确认。
- 历史 Test 未参与训练，但已被多轮实验分析，不再是对当前新假设完全新鲜的确认集。
- 现有数据划分中没有识别出第四个、模型未见且未用于方法开发的样本池。

## 可行路径

真正独立的训练、校准、评价三分组需要：（1）按 Session/Dialogue 重新设计分组并重新训练 CMERC；或（2）引入外部模型未见数据。IEMOCAP 可优先按 Session 设计；MELD 至少按 Dialogue，若能从原始元数据可靠恢复 Episode，应按 Episode 避免跨对话内容泄漏。本阶段未重新划分、未重训练。
