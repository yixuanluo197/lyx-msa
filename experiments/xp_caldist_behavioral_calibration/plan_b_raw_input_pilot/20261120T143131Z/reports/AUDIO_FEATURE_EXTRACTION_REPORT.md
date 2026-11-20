# 新音频特征提取：已完成

|数据集|划分|Dialogue|话语|缺失|维度|
|---|---|---:|---:|---:|---:|
|IEMOCAP|train|108|5283|0|768|
|IEMOCAP|validation|12|527|0|768|
|MELD|train|1038|9989|1|768|
|MELD|validation|114|1109|1|768|

使用公开 facebook/wav2vec2-base，revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8。官方权重SHA-256：3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21。未使用base-960h替代。

所有参数冻结、FP32 eval。原始话语完整音轨解码为单声道16kHz；官方processor整句零均值/单位方差后，最多30秒非重叠窗口，尾段不足400样本合并前段；最后层真实帧加权均值池化，输出768维。所有Validation音轨均不需分块；MELD仅一条Train超过30秒。

MELD一条Train音轨损坏、一条Validation音轨不存在，采用零768+独立audio_missing_mask，保留其上下文位置；未补入旧XP04音频特征。没有新增缺失，未使用Test。

默认Transformers加载旧官方权重时，Torch2.2 weight-norm参数命名发生不兼容，初次检查主动失败并保留日志。最终仅将weight_g/weight_v一对一改名为parametrizations.original0/original1，211个base状态张量strict加载逐值一致，有效卷积核最大误差0；没有随机参数或训练替代。

CPU及真实原媒体GPU前向检查通过：输出有限FP32、重复一致、参数冻结、静音缺失返回零向量与mask。正式特征文件及逐话语source/cache/embedding SHA只保留本地。输入、配置、代码和完整输出哈希见provenance/audio_stage_*.json及local_only/features/*/audio_features.metadata.json。
