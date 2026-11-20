#!/usr/bin/env python3
"""Summarize completed fresh Plan B models, without running inference."""
import csv,json,pathlib
D=pathlib.Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def main():
 rows=[];details=[]
 for ds in ['IEMOCAP','MELD']:
  status=load(D/f'provenance/downstream_stage_{ds}.json')
  if status['status']!='COMPLETE':raise RuntimeError(ds+' downstream incomplete')
  data=D/f'local_only/data/{ds}'
  tlock=load(D/f'local_only/text_models/{ds}/best_model_lock.json')
  metrics=load(data/'validation_metrics.json')
  stages={name:load(data/f'runs/{name}/completion.json') for name in ['m3net','cmerc']}
  assert all(v['status']=='COMPLETE' for v in stages.values())
  rows.append(dict(dataset=ds,seed=1475 if ds=='IEMOCAP' else 67137,**{k:metrics[k] for k in ['n','accuracy','macro_f1','weighted_f1','ece15','brier_sum_classes','nll']}))
  details.append(dict(dataset=ds,text=tlock,training_completion=stages,baseline_metrics=metrics))
 out=D/'results/baseline_metrics.csv';out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (D/'results/baseline_summary.json').write_text(json.dumps(details,indent=2)+'\n')
 lines=['# 自主特征单 Seed CMERC Baseline','', '两个数据集均已完成新的文本/音频提取、全新 M³Net、新 Dialogue 难度与全新 CMERC。旧 XP04 文本/音频特征、Checkpoint、M³Net 权重及难度文件未用于新模型输入或初始化。','', '|数据集|Seed|Validation N|Accuracy|Macro-F1|Weighted-F1|ECE(15 bins)|Brier|NLL|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append('|'+ '|'.join(str(r[k]) if k in ['dataset','seed','n'] else f'{r[k]:.6f}' for k in ['dataset','seed','n','accuracy','macro_f1','weighted_f1','ece15','brier_sum_classes','nll'])+'|')
 lines+=['','所有指标来自真正 Validation，未使用 Test。模型和训练轮数由固定 Train 内部 Dialogue dev 选择，之后冻结；没有使用官方 Validation 反复调参或选择最佳 Seed。','', '训练梯度只使用 Train Fit：IEMOCAP 4747、MELD 8989；内部 dev 为536/1000。为保持选择边界，没有再对完整 Train 进行重拟合。完整 Train/Validation 的新特征均已提取。','', 'RoBERTa Large 分数据集监督微调，最后四层 BOS 原尺度表示；音频为冻结官方 Wav2Vec2 Base，单声道16kHz、整句标准化及真实帧均值池化；视觉沿用官方特征。具体模型版本、输入/代码SHA及启动命令见 configs/ 和 provenance/。','', 'IEMOCAP 使用当前 Fit 统计的逆频率加权 NLL，详见 DOWNSTREAM_CLASS_WEIGHT_AMENDMENT.json；MELD 保留原 FocalLoss。CMERC 保留原 CC、MSCL、CSCL 与课程学习机制，实际非有限辅助损失会使任务失败，不能静默替零。','', 'MELD Train125/utt3、Validation110/utt7 音频不可恢复，均保留原话语并使用零768向量和缺失标记，没有补入旧音频特征。Train556/utt6 使用原CSV文本并保留原Speaker ID，历史文本/Speaker冲突已在数据报告披露。','', '这些是新特征与新的 Train 内部模型选择协议下的单 Seed 基线，不能将它与历史 XP04 指标的差别归因于单一因素，也不能据此宣称五 Seed 稳定性。原始干扰 Pilot 使用此处冻结模型，不再训练。','']
 (D/'BASELINE_REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
