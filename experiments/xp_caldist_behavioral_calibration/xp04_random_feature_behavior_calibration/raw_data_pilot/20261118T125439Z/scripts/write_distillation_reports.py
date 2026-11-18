#!/usr/bin/env python3
"""Render completed evidence. Does not evaluate models or change scientific gates."""
from pathlib import Path
import csv, datetime, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 records={}; fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for ax,ds in zip(axes,['IEMOCAP','MELD']):
  run=OUT/'local_only'/ds/'training';summary=read(run/'training_summary.json');config=read(run/'training_config.json');lock=read(run/'best_model_lock.json');parity=read(OUT/'results'/ds/'feature_parity.json')
  history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines() if line.strip()]
  records[ds]=dict(summary=summary,config=config,lock=lock,parity=parity,history=history)
  initial=read(run/'initial_inner_dev.json')
  ax.plot([r['epoch'] for r in history],[r['train_mse'] for r in history],label='Train (dropout)',marker='o',color='#0072B2')
  ax.plot([0]+[r['epoch'] for r in history],[initial['mse']]+[r['inner_dev_mse'] for r in history],label='Train-internal dev',marker='s',color='#D55E00')
  ax.axvline(summary['best_epoch'],color='0.5',linestyle='--',linewidth=1,label='Selected epoch')
  ax.set(title=ds,xlabel='Epoch',ylabel='Four-layer MSE');ax.legend(fontsize=8);ax.grid(alpha=.2)
  curve=OUT/'results'/ds/'loss_curve.csv'
  with curve.open('w',newline='') as f:
   keys=['epoch','train_mse','inner_dev_mse','global_step','seconds','amp_skipped_steps_rank0'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in history)
  passed=parity['feature_gate_passed'];probpath=OUT/'results'/ds/'probability_parity.json'
  if not passed:
   prob=dict(dataset=ds,status='NOT_RUN_FEATURE_GATE_FAILED',metrics=None,CMERC_checkpoint_loaded=False,reason='At least one frozen four-layer feature compatibility gate failed.');probpath.write_text(json.dumps(prob,indent=2)+'\n')
  lines=[f'# {ds} 蒸馏文本特征验证','',f"特征门：{'通过' if passed else '未通过'}。真正 Validation：{parity['rows']} 条、{parity['dialogues']} 个 Dialogue；仅在训练锁定后评估一次。",'', '| 层（隐藏层偏移） | 最大绝对误差 | MAE | RMSE | 平均余弦 | 最小余弦 | 逐维相关 p05 | 通过 |','|---|---:|---:|---:|---:|---:|---:|---|']
  for r in parity['layers']:
   def v(key):return f"{r[key]:.8g}" if r.get(key) is not None else '未定义'
   lines.append(f"| {r['layer']} ({r.get('hidden_offset', -r['layer'])}) | {v('max_abs_error')} | {v('MAE')} | {v('RMSE')} | {v('mean_nonzero_row_cosine')} | {v('min_nonzero_row_cosine')} | {v('correlation_p05')} | {r['passed']} |")
  lines += ['',f"最佳 epoch：{summary['best_epoch']}；完成 {summary['completed_epochs']} 轮；内部验证 MSE={summary['best_inner_dev_mse']:.8g}。",'',f"概率验证状态：{parity['probability_status']}。",'','逐坐标相关性、均值/标准差差异及 Wasserstein 距离见 results/'+ds+'/per_coordinate_metrics.csv。完整学生特征及模型只保留 local_only。','', '此模型是公开预训练模型经 XP04 Train 特征监督得到的学生编码器，不是原始 CMERC 权重；没有训练 CMERC。']
  (OUT/f'{ds}_TEXT_CLEAN_PARITY.md').write_text('\n'.join(lines)+'\n')
 fig.savefig(OUT/'results/loss_curves.pdf');fig.savefig(OUT/'results/loss_curves.png',dpi=160);plt.close(fig)
 lines=['# XP04 双数据集 RoBERTa 特征蒸馏报告','',f"生成时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。",'', '两套独立 RoBERTa Large 学生均从公开 roberta-large 的缓存快照 722cf37b1afa9454edce342e7895e588b6ff1d59 初始化（权重 SHA-256 047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38），仅使用各自有效 Train 文本及四层缓存目标进行全编码器蒸馏。损失为四层 BOS 原尺度 MSE 的平均；没有情绪标签、分类头、回归转换器或目标归一化。','', '每个数据集按固定种子将约10%的 Train Dialogue 留作内部验证，最小 FP32 MSE 选择最佳 epoch，patience=2、最多10轮；不再以全部 Train 重拟合。官方 Validation 未参与早停或任何超参数选择。','', '| 数据集 | 有效 Train | 实际拟合 / 内部验证 | 完成 / 最佳 epoch | 最佳内部 MSE | 训练耗时（小时） | 特征门 |','|---|---:|---:|---:|---:|---:|---|']
 for ds,r in records.items():
  s,c,p=r['summary'],r['config'],r['parity'];lines.append(f"| {ds} | {c['fit_samples']+c['dev_samples']} | {c['fit_samples']} / {c['dev_samples']} | {s['completed_epochs']} / {s['best_epoch']} | {s['best_inner_dev_mse']:.8g} | {s['elapsed_seconds']/3600:.3f} | {'通过' if p['feature_gate_passed'] else '未通过'} |")
 lines += ['', '硬件为本地四张 TITAN Xp 12GB：IEMOCAP 使用宿主 GPU0/1，MELD 使用 GPU2/3。每套模型双卡 DDP；训练 FP16 autocast/动态缩放，模型和 AdamW 状态保持 FP32，内部与最终提取均为 FP32；开启梯度检查点。有效 batch、显存峰值、AMP 跳步及完整命令保存在训练配置、摘要和日志。DistributedSampler 为双卡均分，每个 epoch 的拟合部分补齐重复1条 Train 样本；内部验证使用无重复分片，逐样本加权汇总。','', 'Tokenizer 使用本地公开 RoBERTa 资产，经 COSMIC byte-BPE 与 Fairseq 字典 ID 全词表核对；IEMOCAP 原句直接输入，MELD 按 COSMIC 参考进行 ASCII 忽略清理。BOS=0、PAD=1、EOS=2；最多512 token（含特殊 token），固定右侧截断，四层顺序为 [-1,-2,-3,-4]，每层1024维 BOS。','', '特征兼容性阈值完整沿用历史 TEXT_RECOVERY_PROTOCOL.json：MAE≤1e-5、RMSE≤1e-4、元素 allclose(atol=rtol=1e-4)、平均余弦≥0.99999、最小余弦≥0.9999、有效逐维相关 p05≥0.999，且所有目标非恒定坐标相关性可定义。当前授权仅替代旧协议中禁止新微调的来源前提，未放宽数值门槛。','', 'IEMOCAP 风险：上游文本缓存的历史划分与 XP04 Seed1475 effective split 不同；此前审计发现部分当前 Validation Dialogue 属于缓存自己的历史 Train。上游编码器监督是否实际覆盖这些话语仍不确定，不能将本轮划分隔离解释为已经排除缓存的历史重叠风险。','', '共享 pkl 的反序列化包含多个划分，但本轮只按明确的有效 Train 和最终 Validation ID 索引文本及四层目标；未索引标签槽或 Test 话语。没有处理音频/视觉，没有修改 XP04 pkl、Checkpoint，没有执行干扰，没有 Git commit/push。交互监控进程在会话中断后消失，已通过 tmux 接管原有分离训练工作进程；没有重启或重复训练。接管后无法取得原父进程的子进程退出码，完成依据同时核对完成日志、训练摘要、最佳模型锁、完整断点和最终验证产物，事件及 PID 记录见 results/execution_status.json。','', '产物：local_only/{dataset}/training/best_student 保存各学生编码器、Tokenizer 与配置；latest_training.pt 保存 epoch 边界模型/优化器/Scaler/各 rank RNG，history.jsonl 和 logs 保存全部训练记录；results 保存曲线和逐层/逐坐标指标。数值不理想的条件完整保留。Train-only 相同分词输入的目标一致性诊断见 results/{dataset}/train_duplicate_consistency.json；两套拟合数据的重复输入均未出现超出 allclose 容差的目标冲突。这一诊断不参与选模型或改变训练。']
 (OUT/'DISTILLATION_REPORT.md').write_text('\n'.join(lines)+'\n')
 decisions=['# 最终蒸馏判断','', '| 数据集 | 学生训练 | 四层特征兼容性 | XP04 概率验证 | 可否直接开始文本干扰 |','|---|---|---|---|---|']
 for ds,r in records.items():
  p=r['parity'];prob=read(OUT/'results'/ds/'probability_parity.json') if (OUT/'results'/ds/'probability_parity.json').exists() else {'status':'PENDING'}
  approved=p['feature_gate_passed'] and prob.get('passed') is True
  decisions.append(f"| {ds} | 完成并保存 | {'通过' if p['feature_gate_passed'] else '未通过'} | {prob['status']} | {'可以（本轮未执行）' if approved else '不可以'} |")
 decisions+=['', '若特征门未通过，说明本轮蒸馏模型未能在既定标准下重建 XP04 的文本空间。内部损失改善或较高余弦不能代替逐层兼容性，也不能据此宣称保留原始 XP04 Checkpoint 已成功。未通过的数据集不加载冻结 CMERC 做概率验证、不进入文本干扰。','', '本轮按原定训练上限和独立验证完成后停止；保留最佳模型、完整断点与失败结果。没有用 Validation 继续调参或训练，没有把学生编码器称为作者原始提取器。']
 (OUT/'FINAL_DISTILLATION_DECISION.md').write_text('\n'.join(decisions)+'\n')
 print('reports and loss curves written',flush=True)
if __name__=='__main__':main()
