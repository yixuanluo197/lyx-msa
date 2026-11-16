# XP04 强度响应实验实施计划

用户已批准：Seed1475、两个数据集完整真正Validation，新增Gaussian 0.10和Dropout10%，四模态条件、10次重复；复用0.05历史参照。

1. 核验旧输入、Training统计、checkpoint和推理源码SHA256；禁止读Test输出或标签。
2. 冻结推理助手，新增可指定dataset/modality的运行器；保持原batch16、FP32、eval、ATen，单target替换并恢复，检查mask/图/权重。
3. 用合成样本核对零标准差、目标索引、缺失视觉、噪声尺度和精确10%替换；每run核对完整Validation原始概率，抽查缓存/非缓存前向一致。
4. 本地健康GPU按模态拆分，每run独立目录；原记录显示同卡双任务会饱和，当前不并发堆入同卡。先观察进度/显存和吞吐。保存命令、PID、设备UUID、环境和结束证据。
5. 从10repeat整类别概率复算原始mu/delta/JS；每话语每条件一条记录。视觉缺失保留，响应统计另外列有效视觉，避免零填充稀释解释。
6. 复用mainline_core，冻结E0/E1/M_joint/M_gaussian/M_dropout、折和嵌套正则化选择；只训练可靠性映射，不训练CMERC。复算E1/0.05旧结果一致性。所有强度结果均报告。
7. 使用历史2000次Dialogue共享抽样复算响应与OOF差值；强度随机流独立，未声称逐repeat噪声配对。不以增加翻转代替校准收益。
8. 独立脚本复核覆盖率、行为公式、指标、折分组、参数重放、源文件哈希及无Test访问；完成后生成中文报告、完整CSV、verification和provenance。
9. 仅将本次实验目录及需要的代码/数据镜像至CMERC仓库，按Lore提交并push origin/main，核对远端commit。保留无关未跟踪文件。

固定0.05为历史主条件；0.10为新增敏感性条件。无自动0.20或30repeat扩展。新结果仅单Seed探索性证据。
