# 正式 Causal Difficulty 来源验收

状态：**真实教师生成完成并通过严格reader校验**。不使用合成教师、随机权重或原始非因果difficulty。

- 教师：本批Causal Baseline validation-selected best.pt，第57轮；checkpoint的epoch==best_epoch==56（零起始）。
- 源checkpoint SHA256：`1d9bbae19f0e4f8bd976bc3e4285f93c14f114cc54e6d28afcc76be8f90dd3bc`。
- difficulty SHA256：`73791cfb5755710e74424329bf8f3e4dc51d6ba464d19d6c8b71566cf9e73463`。
- 不可变文件名：`IEMOCAP_causal_diff_73791cfb5755710e74424329bf8f3e4dc51d6ba464d19d6c8b71566cf9e73463.pkl`，完整SHA寻址，pickle与.manifest.json独占创建并互相核对。
- 精确覆盖全部且仅108个允许的训练对话；有序训练ID SHA256=`6bf578ff93a22b88d822615c3cbe1a6f68f17fb1e59a0e117a953e758f1ac305`；值有限，范围[0.046714842, 1.292021513]。
- train seed1937、split seed1475；causal LN/GRU、图深度、全部架构白名单、特征SHA、训练/验证ID哈希与阶段A一致；教师训练/选模协议=validation。
- replacement=`prefix_mean_v1`，是前缀平滑视图，不是完整模态移除。

严格校验教师实际文件SHA与保存的训练/选模provenance；同时单独检查输出ID。没有把“只计算train难度”当成教师没见过validation的证据。阶段B通过的artifactSHA在完整训练前后均重新校验，完整模型best/latest必须固定同一SHA。任意来源或完整性失败会阻断下游。

原始不可变artifact/sidecar、源checkpoint和全部生成日志留服务器。源checkpoint绝对路径保留用于真实SHA复核，不移动或删除；公开报告只发布内容哈希及脱敏来源。特征提取上游仍未审计。

证据：DIFFICULTY_VERIFICATION.json、difficulty/*、difficulty_generation.log、PIPELINE_STATE.json以及阶段A验收报告。
