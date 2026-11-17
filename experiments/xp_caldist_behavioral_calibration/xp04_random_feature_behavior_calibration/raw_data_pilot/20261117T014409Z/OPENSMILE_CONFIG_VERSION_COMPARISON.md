# OpenSMILE 配置与版本补充核查

审计日期：2026-11-17 UTC。范围仅限官方资料、配置和源码的静态比较；没有执行音频特征提取、训练、干扰、CMERC 前向或 Test 标签分析。

## 结论与证据等级

已找到可执行的官方 IS10、IS10 compatibility、emobase2010、IS13/ComParE 配置家族以及源码。**尚未找到把这些具体文件、引擎版本和预处理命令关联到 XP04 那份特征的生产记录。** 1582 维不能唯一识别 IS10 的具体变体，6373 维也不能唯一识别 IS13 与数值修订后的 ComParE_2016。下表属于官方候选流程的静态事实，不是 XP04 Clean Parity 结果。

| 官方候选 | 维度依据 | 关键差别 | 对 XP04 的含义 |
|---|---|---|---|
| IS10_paraling.conf | 官方文档说明 1582 个 functionals | 文件警告其数值不兼容 `<2.1` 的原始 IS10 | 不能把新版本默认 IS10 当成原始配置 |
| IS10_paraling_compat.conf | 原 IS10 的兼容家族 | 文件注明 Jitter/Shimmer 修复不保证旧数值兼容 | 即使选 compat 也须确定引擎与预处理 |
| emobase2010.conf | 官方配置头部与文档均写 1582 | 极值位置等归一化不同，单体配置也有不同显式/默认参数 | 同维度不代表等价 |
| IS13_ComParE.conf | 原挑战论文明确 6373 | 较大 LLD/functionals 组合，含 Viterbi F0、HNR 等 | 不能与完整 IS10 1582 直接视作同一特征 |
| ComParE_2016.conf | 官方 Python 文档表格明确 6373 | 官方称其为 ComParE 2013–2015 的数值优化版本 | MELD 早期论文已明确 IS13；仍缺实际版本/配置哈希，不能拿同维度的 2016 替换 |

维度和配置映射来自 [官方 IS10 文档](https://github.com/audeering/opensmile/blob/v3.0.2/doc/sphinx/get-started.rst)、[IS13 原论文第 3 节](https://www.isca-archive.org/interspeech_2013/schuller13_interspeech.pdf)及[官方 Python 特征表](https://audeering.github.io/opensmile-python/)。这些来源没有提供 XP04 生产命令。

## 版本历史与缓存

本轮从官方 `audeering/opensmile` 下载三个 tag 的源码归档，只展开相关配置、共享 include、文档、FFT/WAV/Jitter 源码。没有另建 Git 仓库。完整 URL、文件 SHA-256、归档 SHA-256、大小和 API 历史快照保存在 `provenance/opensmile_source_manifest.json`。

| 官方 tag | 归档大小（字节） | 归档 SHA-256 |
|---|---:|---|
| v3.0.0 | 4,811,100 | `0f7a91276a647c4558211b7a635b8aa41f2418f84b603e76c9b363f848cbbb73` |
| v3.0.1 | 4,776,121 | `b8e5d69d6c3d729eb6e7825e2bd913c819e61315524a739b245069dcaed0d8c3` |
| v3.0.2 | 4,788,935 | `b423c785c948e6e8d5b84c002419875ba456f048f5f86ce29f5e21bba9ef5bdd` |

截至本次查询，官方 tags/releases API 仅列出 3.0.0、3.0.1、3.0.2；IS10 路径的可见提交历史从 `810cb37377deed6e0ea41564d1b51e60317ca25d`（2020-10-21，首次加入 3.0 源码）开始。旧版顶层 `config/IS10_paraling.conf` 路径提交查询返回空列表。这不能证明旧版从未发布，只说明本次官方 Git 历史不能提供 2.1/2.3 的逐文件快照。[官方安装说明](https://audeering.github.io/opensmile/get-started.html)也说明 GitHub 源码从 3.0 开始，旧源码随发布归档提供。

官方 [CHANGELOG](https://github.com/audeering/opensmile/blob/v3.0.2/CHANGELOG.md)记载 2.1 发布于 2014-12-23，2.3 发布于 2016-10-28；2.3 引入数值优化的 ComParE_2016，并宣称标准集对 2.1/2.2 的兼容性经过核验。这是发布方说明，不能代替对 XP04 的验证。3.0.1 去掉 `-ffast-math` 和 `/fp:fast`，有利于跨编译器复现；所以配置字节一致仍不足以证明不同引擎/构建产出的逐维一致。

## 配置字节比较与真实处理参数

下列关键文件在 v3.0.0、v3.0.1、v3.0.2 中逐字节一致；其它已缓存 include 的逐文件哈希见 manifest。

| 文件 | 三个版本共同 SHA-256 |
|---|---|
| IS10_paraling.conf | `797eb7c3b912540cb5f8f8d5963151a706a60c00b50de5706a77f1976b96e0f5` |
| IS10_paraling_compat.conf | `75e28f7ad5de545fa68b9e392aa3def37f10aa8e8c4e736883ab8a14dd107973` |
| IS10_paraling_core.lld.conf.inc | `8179a9725904ee334ecd0f3f5c8d4cadc1655ff06ca10c67e15da585a1d9a4d7` |
| IS10_paraling_core.func.conf.inc | `ac45cdbc3a928c513254d85262da80e767fb32064f7d5c3268a3fad87429aa3f` |
| emobase2010.conf | `e8550e5b49f7ed26457bafddd1752cd978f0d04685d4f6fdb14257ef72fe4fe7` |
| IS13_ComParE.conf | `c3550ca81e7d7c32194e05b61c9a36427b302832e29cc894cc24706c44de51f2` |
| standard_wave_input.conf.inc | `f7d3be1bd36c434e0681219f8c9be584f189028638cfeaecec39de50b1caef75` |

IS10 的官方 [LLD core](https://github.com/audeering/opensmile/blob/v3.0.2/config/is09-13/IS10_paraling_core.lld.conf.inc)显式使用 60ms 高斯窗分支、25ms Hamming 窗谱分支，均以 10ms 步长进行处理。`fr40` 是组件名，其实际 `frameSize=0.060`，不能按名字写成 40ms。谱分支预加重 0.97；MFCC 0–14；8 个 log-Mel 频带；LPC 阶数 8；平滑窗口 3，delta 回归窗口 2。`cSpecResample targetFs=11000` 只位于 LPC 所用的分支，不能据此称原 WAV 全部被重采样到 11kHz。functionals 默认以完整输入段聚合，输出顺序为普通 LLD+delta functionals、非零/音高 LLD+delta functionals、onset/duration。component sections 以原文顺序缓存到 `provenance/opensmile_static_sections.json`。

[compat 配置](https://github.com/audeering/opensmile/blob/v3.0.2/config/is09-13/IS10_paraling_compat.conf)与非 compat 并非只改文件名：compat 的两个 FFT 都显式 `zeroPadSymmetric=0`，Jitter 搜索范围 0.25 且启用旧阈值兼容；非 compat 的搜索范围是 0.20，并未在 FFT 实例覆盖该选项。官方 [FFT 源码](https://github.com/audeering/opensmile/blob/v3.0.2/src/dspcore/transformFft.cpp)的默认值为 1，注释说明 2.3 起改为对称补零，设计上保持 FFT 幅度、改变相位。不能由这个说明推导全部上游/下游数值相等。

[emobase2010 配置](https://github.com/audeering/opensmile/blob/v3.0.2/config/emobase/emobase2010.conf)显式将极值位置按 segment 归一化；IS10 核心为 frame 口径。它还存在显式参数覆盖、边界处理与默认值差别。1582 相同不能证明任何逐维同值。IS13 与 ComParE_2016 均使用 60ms、20ms 两种帧及 10ms 步长，但 IS13 显式 FFT 对称补零为 0、`useBrokenJitterThresh=1`，2016 配置为 1 和 0。即使字段名和总维数相同也不能自动互换。[IS13 core](https://github.com/audeering/opensmile/blob/v3.0.2/config/is09-13/IS13_ComParE_core.lld.conf.inc)、[2016 core](https://github.com/audeering/opensmile/blob/v3.0.2/config/compare16/ComParE_2016_core.lld.conf.inc)。

## WAV、字段与标准化的边界

官方共享 WAV 输入读整个文件，默认 `start=0,end=-1`，并 `monoMixdown=1`；是否按对话时间戳裁切、裁切前后是否重采样、选用哪一路声道，均必须由生产脚本确定。`cWaveSource` 读取 RIFF 的实际采样率；源码中的默认 16000 仅是读取无头 raw PCM 时的选项，不能证明这些特征一律使用 16kHz。查看频谱配置的最高频率也不能反推原始采样率。[WAV 输入 include](https://github.com/audeering/opensmile/blob/v3.0.2/config/shared/standard_wave_input.conf.inc)、[WAV 源码](https://github.com/audeering/opensmile/blob/v3.0.2/src/iocore/waveSource.cpp)。

配置本身有时间位置、回归等内部归一化，不能等同于训练集外部 z-score/标准化。尚无 XP04 生产时的字段 header、外部 scaler、统计量或列选择顺序。这里已核实候选的拼接顺序，但未生成或确认 XP04 1582 列的完整有序字段名，也没有确认 MELD 1611 维选中了 6373 中哪些字段。官方 standard_data_output 支持 ARFF/CSV 的实例名、时间戳等非特征字段，未来复现必须显式剥离并校对 header，而非简单按列数读取。

## 本地可用工具核验

只读检查 `msa_vision` 环境中已安装的 Python `opensmile` 分发版为 **2.6.0**。这是 Python 包版本，本轮没有执行引擎版本命令，不把它称作 OpenSMILE 引擎 2.6。其[官方枚举](https://audeering.github.io/opensmile-python/api/opensmile.FeatureSet.html) `FeatureSet.IS10` 默认指向 **IS10_paraling_compat**；本地没有非 compat 的顶层 `IS10_paraling.conf`。该包的 compat 相对官方 v3.0.2 主体只改为外部 source/sink 接入；IS10 LLD/func 两个核心 include 与 v3.0.2 逐字节一致。29 个本地相关文件中 11 个与对应官方文件字节一致，其余包含包装接口和 include 差别，逐文件结果见 `provenance/opensmile_installed_candidate.json`。没有把当前安装工具认作 XP04 原生产工具。

## 本轮决策

本轮静态比较解决了“候选集是否唯一、版本是否可互换”的问题，不能解决“XP04 当年实际运行了什么”。可信 Clean Parity 的前置证据至少包括：生产者原始引擎/构建、实际主配置及递归 include、原 WAV/分段和声道命令、有序 header、外部标准化；MELD 还需原选择器/字段映射。找到这些记录后才可对固定少量真正 Validation 样本执行 Clean 提取；在此之前不运行候选配置来宣称恢复成功。

复核脚本：`scripts/opensmile_static_audit.py`、`scripts/opensmile_compare_configs.py`。缓存含 3 个官方归档、131 个展开文件记录、官方 API 历史与支持资料 URL/SHA。未修改历史数据或特征；未 Git Commit/Push。

## 最后一次有界检索补充

重新检查官方安装文档和官方 release 页的实际链接，未发现直接指向 2.1/2.3 归档的链接。搜索找到一条历史线索：[aria2 2018 年 Issue #1118](https://github.com/aria2/aria2/issues/1118)记载官方域名 `http://audeering.com/download/1318/` 曾返回 `opensmile-2.3.0.tar.gz`。这是第三方对旧链接的记录，不是官方源码校验声明。本轮跟随其实际链接并升级 HTTPS 后，最终官方 URL `https://www.audeering.com/download/1318/` 返回 HTTP 404、HTML 内容，未将该响应认作归档。没有猜测新的旧包 URL。此有界检索没有恢复官方 2.1/2.3 源码包，已经停止继续扩展搜索。

对本地 `libSMILEapi.so` 只读进行 SHA-256 和 ELF 字符串检查：文件大小 **2,376,360 字节**，SHA-256 为 `7d708bd3e02ec29d1fdc94e8b60595a65052d6f955b96bb7c88b181d65d81231`。相邻 banner 字符串包含 `openSMILE`、版本字面量 **3.0.1**、构建日期字面量 `2023-10-19T14:14:02Z`；它与 Python 分发包版本 2.6.0 是不同版本概念。`readelf -n` 没有打印 Build-ID notes。这里只识别二进制的静态自报标识，没有加载共享库，也没有证明其完整源码/编译选项或与 XP04 的关系。

请求状态、页面响应哈希、实际链接、二进制 banner 上下文保存在 `provenance/opensmile_final_bounded_search.json`，相关响应保存在 `sources/opensmile/`。新增事实不改变 Clean Parity 门控：MELD 论文指向 IS13 比单纯维度推断更强，但实际生产版本和配置仍未知；IEMOCAP 的实际配置也仍未锁定。
