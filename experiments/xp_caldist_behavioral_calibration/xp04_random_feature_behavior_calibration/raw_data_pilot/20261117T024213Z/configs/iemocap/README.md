# IEMOCAP 候选提取配置

这些是本轮重建研究的固定候选，**不是已确认的 XP04 原始生产配置**。来源是上一轮缓存的 audEERING OpenSMILE 官方 `v3.0.1` 配置；递归配置树及逐文件 SHA-256 记录于 `provenance/iemocap_extraction_configs.json`。执行工具为已安装 Python opensmile 2.6.0、libSMILEapi 自报引擎 3.0.1；未安装新依赖。

| 候选 | 原官方文件 | 本轮修改 |
|---|---|---|
| `is10` | `is09-13/IS10_paraling.conf` | 文件 source/sink include 改为 Python external source/sink |
| `is10_compat` | `is09-13/IS10_paraling_compat.conf` | 去除内嵌 `cWaveSource` instance/section，使用 external source；标准输出 include 改为 external sink |
| `emobase2010` | `emobase/emobase2010.conf` | 去除 `cWaveSource`、`cArffSink` instance/section；使用 external source；按原 ARFF reader 顺序 `func;funcNz;functOnsets` 拼接后使用 external sink |

所有修改仅接入输入、输出，保留官方 DSP、LLD 与 functionals 参数。Python wrapper 文件以 `.python_external.conf` 结尾；旁边保留未经修改的官方原文件。外接 source/sink include 从已安装官方 Python 包复制，并与其余文件一起留存哈希。初次 compat 包装遗留了内嵌 source，静态初始化失败，已在首轮音频 smoke 前修正；初始记录没有用来产生候选特征。

输入为数据集已发布的话语 WAV，不重新切段。使用 `scipy.io.wavfile.read`，强制检查每条为 16 kHz、单声道，否则保留失败记录和 NaN。PCM 有符号整数按位宽除以 `2**(bits-1)` 后转 float32；这是数值解码，不是外部增益或 z-score。无重采样、下混、强度门限、静音裁剪、时长修改、每段响度调整或特征标准化。外部标准化如有需要，仅由后续 Train 拟合阶段负责。

提取器按 manifest 行序保存 `(N,1582)` float32 数组、1582 个有序字段名、逐行音频/解码/输出哈希及状态、独立有效行 mask。失败行不填零。40 行为一块，块级持久保存允许恢复；最多 6 个进程，OMP/BLAS 各 1 线程。Train smoke 为前 3 行，之后使用完整有效 Train 5283 行。Validation 提取需额外提供模型锁定文件；不得用于训练、预处理参数或候选选择。

执行脚本为 `scripts/extract_iemocap_candidates.py`。运行命令、manifest SHA、PID、环境及结果 SHA 分别位于 `provenance/iemocap_extraction_*`；原媒体和候选完整特征只保留本地。

公开副本说明：以上配置树、wrapper文件及逐行产物描述的是本地实验归档；GitHub仅发布配置来源/修改说明和哈希，完整上游配置镜像保留本地，见 ../../PUBLICATION_NOTE.md。
