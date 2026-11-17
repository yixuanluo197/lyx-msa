# 发布与运行说明

研究者在实验结束后明确要求“上传到lyx的github”。此次只向既有LYX仓库发布经过筛选的副本，不改写本地实验与历史资产。原报告“未提交/推送”及verification中的git字段描述冻结实验阶段，发布是其后的独立操作。

所有服务器路径和宿主机名已替换为`/LOCAL/`、`LOCAL_RUN`、`LOCAL_HOST`占位符。代码文件中同样处理这些环境常量；原始源码SHA与公开副本SHA在publication_manifest.json中分别记录。配置冻结文件中的阈值、划分数量、候选菜单和CV规则保持不变；原运行的哈希绑定对应本地字节，不能拿公开脱敏副本直接冒充原运行输入。

本轮转换器均判为C；原始Checkpoint概率比较因特征门失败而NOT_RUN。概率误差为null，不能解释成0。

## 可执行的合成测试

在安装NumPy、SciPy及scikit-learn的兼容Python环境中，从本目录运行：

```bash
python -m unittest discover -s scripts -p 'test_*.py' -v
```

测试只用合成输入；不会拟合真实Validation、加载原始Checkpoint或访问Test。公开副本发布前实际运行结果记录于publication_verification.json。

## 原始音频实验的依赖边界

提取需要有权访问的原始Train/Validation媒体、原始XP04配置与资产、已有官方MELD HDF5权重，以及版本/字段/哈希记录所指向的OpenSMILE v3.0.1配置和本地既有环境。没有重新分发这些资产或新增下载任务。

原始配置/include清单与哈希见provenance/iemocap_extraction_configs.json；MELD包装变更见configs/meld/IS13_source_wrapper.diff。源码镜像及本地配置树未公开，需要独立准备其对应版本并设置环境常量、目录、清单与锁定记录。具有本地资产时可参考scripts的manifest、提取、GroupCV及单次Validation接口。公开副本不宣称脱离这些依赖即可重新运行完整实验；不要为运行归档脚本而重新打开本轮Validation反复调参。

训练参数、每折scaler和最终转换器二进制保留本地，公开方法与网格在configs/reconstruction_protocol.json及CONVERTER_METHOD_COMPARISON.md中。scripts/verify_delivery.py和write_global_delivery.py是原本地交付流程的归档实现，依赖被排除的本地资产；它们不是公开目录的完整性检查器。公开完整性由publication_manifest.json和publication_verification.json给出。
