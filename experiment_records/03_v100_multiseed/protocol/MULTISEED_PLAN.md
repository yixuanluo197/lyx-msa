# 原始非因果 CMERC 多 seed 执行计划

本工程独立于 `../CMERC`，从其已发布版本 `91a1bffa763d57015896dee1014f0d11c5d09029` 建立源码快照；保留已验证的非因果分支和官方数学定义。旧训练、checkpoint、预测和报告不写入本工程。

## 预注册矩阵

- validation 选模；IEMOCAP 固定 split-seed=1475，108 train / 12 valid / 31 test 对话；MELD 官方 1038 train / 114 valid / 280 test 对话。
- 两数据集统一训练 seeds=[1475,67137,2024,3407,42]；前三个构成首轮，后两个在前三轮数值、完整性、配对和协议检查通过后执行。预先固定，不按测试成绩删选。
- 每 seed × 两数据集 × baseline/CMERC：首轮 12 runs，最终 20 runs；IEMOCAP 每项80 epoch，MELD 每项40 epoch，共1200正式 epoch。
- 所有模型超参数沿用原复现；显式 causal=false、BN、batch16、lr1e-4；不增加因果模块。
- 严格多 seed 随机过程：只在初始化播种、评估保存/恢复 RNG、DataLoader seed=training_seed+epoch，独立固定 split-seed；默认 legacy RNG 行为仍可用以保持单 seed 入口兼容。随机过程差异必须在报告中披露。
- V2统一使用capacity_safe随机排列。原模型固定1000超边/5200节点；batch16不变，整轮含所有训练对话且仅末批可不足16。仅在任一批超过容量时拒绝整轮候选排列，重抽不消耗模型全局RNG，不使用标签/指标筛选。记录重排次数和最大图容量。此条件随机排列与原单seed/V1协议不同，全部V2种子均统一应用，禁止混算。
- 首轮V1在seed67137的IEMOCAP第6epoch遇到1049超边并失败；V1完成/失败/中断产物完整保留在独立旧路径，不能纳入V2统计。

## 执行顺序与验收

1. 源码与非因果路径审计；训练 seed / split-seed / epoch sampling / checkpoint provenance 与恢复回归测试。
2. 脚本、队列、seed 独立评估和自动统计；拒绝覆盖任意已有 run/log/result。
3. biggpu 健康白名单核验（排除GPU4），短调度测试与真实训练/恢复检查；仅空闲卡1/2/3/5、一卡一任务。
4. 冻结源码，保存文件哈希、环境、数据哈希和完整split IDs；持久队列启动首轮12 runs。
5. 完整训练、独立测试评估、三 seed 稳定性门控；通过后补齐后两 seed。
6. 保存每seed分类/CEL/ECE、成对增益、样本标准差(ddof=1)、收敛曲线、epoch分布和中文报告 `reports/10_MULTI_SEED_REPRODUCTION.md`。

## 初始资源预算

V100 32GiB健康卡4张，一卡一任务；实际IEMOCAP峰值allocated约5–9GiB，但ATen/dropout与CUDA缓存可令宿主nvidia-smi占用接近30GiB，因此仅用没有其他计算进程的整张卡，不能按allocated预算并卡。MELD明显更小但本批仍一卡一实验。根据V1实际baseline40/80epoch约7/13分钟，CMERC耗时更长，首轮保守2–4小时墙钟、全20项约4–6小时，非完成承诺。20组best/latest约3.8GiB，日志/预测/快照及额外smoke预算约2GiB，V1诊断另行保留；V2启动前biggpu /data2约19GiB可用。数据使用独立目录内指向经过哈希核验的只读特征副本的链接，不重复下载或改写原数据。
