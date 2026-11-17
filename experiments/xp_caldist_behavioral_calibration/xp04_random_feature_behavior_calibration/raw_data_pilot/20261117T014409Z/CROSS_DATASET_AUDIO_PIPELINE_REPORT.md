# 双数据集音频生产链恢复对照

本轮取得了更具体的原始文献、版本差异与来源冲突证据，**两个数据集仍未满足原始音频Clean Feature Parity门控**。没有运行候选配置提取或CMERC推理。

## 五类证据分别列示

| 证据类别 | MELD | IEMOCAP |
|---|---|---|
| 官方原始资料 | 早期arXiv直接写IS13-ComParE；早期提强度阈值/Z-standardization，后续提SVM L2；作者Issue另称PCA | MMGCN原论文/代码写IS10；M3NET补充写IS13，但作者说明自己只用预提取特征 |
| 可执行代码与原权重 | 原1611输入、BiGRU150×2原HDF5与对话构造/导出已具备；无实际原选择器/声学前端 | 官方IS10/compat/emobase2010参考配置可用；无连接到XP04原文件的producer命令/构建/后处理 |
| 只能推断的可能流程 | 配置家族IS13有直接支持；选择方法、准确变体与归一化执行细节仍不能靠名称或维度决定 | IS10是更有来源支持的家族；具体非compat/compat/旧版及scaler不能由1582确定 |
| 实际数值验证 | 上轮全部13708条300维同值及24条1611→300通过；本轮仅静态检查已选数组，不重跑原网络 | 历史官方文件同值；本轮没有WAV→1582数值验证 |
| 仍无法恢复 | 原WAV预处理→相同6373→相同1611，包括selector/转换、列顺序、scaler | 原引擎/配置/include/字段顺序、生产音频与所有后处理记录 |

MELD早期GRU描述与本地原HDF5结构相符，是额外文献佐证，不等于识别原SVM/PCA或完整训练预处理。其它项目如MUStARD、CMN、ICON中可执行或可描述的流程只能作为参考；没有文件级生产关系就不能移植为XP04原链。

## 为什么同维度或同名配置不足

OpenSMILE官方配置静态比较表明：非compat IS10、compat与emobase2010的1582输出并非可互换；IS13与ComParE_2016同为6373但存在数值兼容差异。相同配置字节在不同编译设置/引擎上仍不保证逐维相同。本轮已找到当前环境参考工具，未将其版本回填到历史生产记录。

MELD选择后的1611/1422数组是输出数据，不包含原6373字段名、selector对象或scaler。跨任务相同列只能证明一部分输出坐标共享，不能确定它们在原6373中的位置或说明实际选择算法。现存GRU输入层也不会包含缺失的OpenSMILE与selector。

## 新鲜本地核验及其边界

固定IEMOCAP6条/6Dialogue，MELD24条/3完整Dialogue均属于各自effective config的真正Validation。现有音轨metadata：IEMOCAP16kHz、mono、PCM16；MELD48kHz、六声道5.1、AAC。媒体与特征/配置/checkpoint已有字节哈希留档。它们不等于历史下混、重采样、切段与标准化的生产证明。

MELD300维具有双向上下文依赖，完整同Dialogue必须重编码后再按协议取目标行；IEMOCAP若确认为逐话语functionals则可逐WAV处理，但这一点也需生产证据。两套协议不能只因保存为二维数组就视作相同。

本轮所有新raw Clean Parity和预测概率误差均为null，未将历史部分parity提升为全链通过。原始数据干扰、训练、完整数据集扩展、Test标签使用及Git提交/推送均未执行。

细节见[MELD来源](MELD_AUDIO_SELECTOR_RECOVERY.md)、[IEMOCAP来源](IEMOCAP_IS10_RECOVERY.md)、[配置版本](OPENSMILE_CONFIG_VERSION_COMPARISON.md)、[MELD Clean状态](MELD_CLEAN_PARITY_REPORT.md)、[IEMOCAP Clean状态](IEMOCAP_CLEAN_PARITY_REPORT.md)。
