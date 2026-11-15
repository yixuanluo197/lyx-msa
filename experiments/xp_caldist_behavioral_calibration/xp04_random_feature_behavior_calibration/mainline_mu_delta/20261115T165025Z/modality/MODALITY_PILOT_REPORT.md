# 原μ/δ单模态技术Pilot

仅Seed1475，text/audio/visual/joint分别学习同一四维μδ评分；每个模态每条原始话语一条记录，不把条件和随机重复当独立样本。

| dataset | modality | method | ece15 | binary_brier | binary_nll |
|---|---|---|---|---|---|
| IEMOCAP | text | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | text | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | text | M_joint | 0.080723 | 0.222036 | 0.638888 |
| IEMOCAP | audio | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | audio | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | audio | M_joint | 0.088840 | 0.223429 | 0.642747 |
| IEMOCAP | visual | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | visual | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | visual | M_joint | 0.070781 | 0.222937 | 0.640842 |
| IEMOCAP | joint | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | joint | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | joint | M_joint | 0.085903 | 0.223983 | 0.645440 |
| MELD | text | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | text | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | text | M_joint | 0.042674 | 0.193197 | 0.569222 |
| MELD | audio | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | audio | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | audio | M_joint | 0.043731 | 0.194260 | 0.573347 |
| MELD | visual | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | visual | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | visual | M_joint | 0.042415 | 0.192715 | 0.568501 |
| MELD | joint | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | joint | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | joint | M_joint | 0.049019 | 0.194164 | 0.571276 |

## 相对E1的配对差

| dataset | modality | metric | mean_difference | ci_low | ci_high |
|---|---|---|---|---|---|
| IEMOCAP | text | ece15 | -0.009051 | -0.021411 | 0.006659 |
| IEMOCAP | text | binary_brier | 0.000827 | 0.000000 | 0.002446 |
| IEMOCAP | text | binary_nll | 0.001824 | 0.000000 | 0.005425 |
| IEMOCAP | audio | ece15 | -0.000934 | -0.006351 | 0.023181 |
| IEMOCAP | audio | binary_brier | 0.002220 | 0.000000 | 0.006943 |
| IEMOCAP | audio | binary_nll | 0.005683 | 0.000000 | 0.017994 |
| IEMOCAP | visual | ece15 | -0.018993 | -0.035327 | 0.007159 |
| IEMOCAP | visual | binary_brier | 0.001727 | 0.000000 | 0.003591 |
| IEMOCAP | visual | binary_nll | 0.003778 | 0.000000 | 0.007967 |
| IEMOCAP | joint | ece15 | -0.003871 | -0.011234 | 0.017091 |
| IEMOCAP | joint | binary_brier | 0.002774 | 0.000000 | 0.008968 |
| IEMOCAP | joint | binary_nll | 0.008376 | 0.000000 | 0.028048 |
| MELD | text | ece15 | -0.000556 | -0.015341 | 0.016003 |
| MELD | text | binary_brier | 0.000433 | -0.000876 | 0.001759 |
| MELD | text | binary_nll | 0.000639 | -0.002156 | 0.003537 |
| MELD | audio | ece15 | 0.000501 | -0.016688 | 0.014334 |
| MELD | audio | binary_brier | 0.001496 | -0.000730 | 0.004201 |
| MELD | audio | binary_nll | 0.004763 | -0.001956 | 0.013868 |
| MELD | visual | ece15 | -0.000815 | -0.016262 | 0.007634 |
| MELD | visual | binary_brier | -0.000049 | -0.001170 | 0.000985 |
| MELD | visual | binary_nll | -0.000082 | -0.002818 | 0.002266 |
| MELD | joint | ece15 | 0.005789 | -0.014175 | 0.014815 |
| MELD | joint | binary_brier | 0.001400 | -0.000184 | 0.003114 |
| MELD | joint | binary_nll | 0.002693 | -0.000665 | 0.006436 |

各模态为预先登记探索，不能选择赢家替换五Seed joint主结果。单Seed不存在跨模型Seed确认，样本标准差标记为NA，不能将其解释为无变异。视觉缺失样本保留，原始跳过视觉扰动产生的零响应不能解释为视觉可靠。joint逐条复现第一批Seed1475结果。

分数R=a+bC+Z(B)w，q=sigmoid(R)，μ/δ输入与联合训练方案和五Seed完全相同；每次拟合/标准化仅使用相应训练Dialogue。区间按Dialogue配对抽样2000次，不含重新拟合。所有bootstrap区间均为探索性区间，未做多重比较调整。当前Validation已反复分析；只能作为探索性机制信息。
