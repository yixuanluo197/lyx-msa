# XP13 GitHub交付说明

本归档是在四快照×32固定训练Batch的只读诊断真实完成、正式验收及独立聚合复核后生成。

新增科学运行：训练0、optimizer.step0、测试前向0；768次模型前向、512次独立autograd.grad仅用于诊断。当前结果不支持唯一选择新的CC/MSCL/CSCL系数；建议先验证原分类Loss与CC的置信度条件方向关系。

提交范围仅第13组独立目录、中文报告和两处导航新增入口。模型权重、原始数据、逐话语标签/预测/置信度、实际对话ID、原始日志、缓存、state不上传。作者课程pkl也保留服务器，公开原SHA和来源限制。

publication_manifest.json列出交付文件SHA256，不包含自身以避免循环哈希。服务器state/publication_verification.json保存发布门禁，state/github_remote_verification.json保存推送后远端commit/blob核验。冻结的原历史参考含划分ID，因此不发布原件；公开脱敏副本带原SHA与删除字段计数，冻结原件保持不变。

不覆盖04～12，不将其他会话的工作并入本次提交。图7仅修复图例裁切，原版和前后哈希保留；报告的生成版本与人工解释增补均可追溯。
