# GPU 原预测对齐

全部10个checkpoint通过；FP32、eval、ATen、batch16、原始Test顺序。atol=1e-6，rtol=1e-5。

| Dataset | Seed | Prob max error | Log-prob max error | Passed |
|---|---:|---:|---:|---|
| IEMOCAP | 1475 | 2.3841858e-07 | 3.8146973e-06 | True |
| IEMOCAP | 67137 | 2.9802322e-07 | 2.8610229e-06 | True |
| IEMOCAP | 2024 | 2.3841858e-07 | 3.8146973e-06 | True |
| IEMOCAP | 3407 | 3.5762787e-07 | 3.8146973e-06 | True |
| IEMOCAP | 42 | 2.3841858e-07 | 1.9073486e-06 | True |
| MELD | 1475 | 2.3841858e-07 | 7.1525574e-07 | True |
| MELD | 67137 | 2.9802322e-07 | 9.5367432e-07 | True |
| MELD | 2024 | 1.7881393e-07 | 7.1525574e-07 | True |
| MELD | 3407 | 1.7881393e-07 | 7.1525574e-07 | True |
| MELD | 42 | 1.7881393e-07 | 4.7683716e-07 | True |

逐项ID、话语索引、标签、prediction、argmax严格相同。完整向量逐元素通过容差。保存新Test概率只用于本次复现对齐，不参与任何Pilot诊断或拟合。
