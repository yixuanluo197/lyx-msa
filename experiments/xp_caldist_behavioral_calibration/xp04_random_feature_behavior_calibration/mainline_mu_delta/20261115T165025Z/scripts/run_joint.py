"""CPU-only joint five-seed original mu/delta calibration experiment."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from mainline_core import FEATURES, METRICS, PENALTIES, calibration_metrics, dump, fit_outer, markdown_table, metrics, sha

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'preregistration.json').read_text())
OLD=Path(CONFIG['source_five_seed'])
SEEDS=CONFIG['seeds']
KEY=['dataset','seed','dialogue_id','utterance_index']
METHODS=['E0','E1']+list(FEATURES)


def read(path):
    return pd.read_csv(path,dtype={'dataset':str,'dialogue_id':str})


def checked_input(path,modality=False):
    gate=json.loads((ROOT/'inputs/verification.json').read_text())
    assert gate['all_passed'] and gate['source_files_unchanged'], 'Source input integrity gate required'
    f=read(path)
    k=KEY+(['modality'] if modality else [])
    assert not f.duplicated(k).any()
    assert set(f.dataset)==set(CONFIG['datasets'])
    assert ((f.prediction==f.label).astype(int)==f.correct).all()
    assert np.isfinite(f[['confidence']+FEATURES['M_joint']].to_numpy(float)).all()
    assert ((f.confidence>=0)&(f.confidence<=1)).all()
    for (ds,seed,mod),g in f.groupby(['dataset','seed','modality']):
        cfg=CONFIG['datasets'][ds]
        assert len(g)==cfg['utterances'] and g.dialogue_id.nunique()==cfg['dialogues']
        assert seed in SEEDS
        assert mod in (['text','audio','visual','joint'] if modality else ['joint'])
    if modality:
        assert len(f)==6544 and set(f.seed)=={1475}
    else:
        assert len(f)==8180 and set(f.seed)==set(SEEDS)
    return f


def outer_definitions(ds,seed):
    all_folds=json.loads((OLD/'baseline/fold_parameters.json').read_text())
    return [r for r in all_folds if r['dataset']==ds and r['seed']==int(seed)]


def run_group(frame,methods,out):
    frame=frame.reset_index(drop=True)
    ds=str(frame.dataset.iloc[0]); seed=int(frame.seed.iloc[0]); mod=str(frame.modality.iloc[0])
    output=[]; params=[]; selections=[]
    assignments=np.zeros(len(frame),int)
    for fold in outer_definitions(ds,seed):
        train=np.flatnonzero(frame.dialogue_id.isin(fold['train_dialogues']))
        held=np.flatnonzero(frame.dialogue_id.isin(fold['held_dialogues']))
        assert len(train)==fold['train_n'] and len(held)==fold['held_n']
        assert len(train)+len(held)==len(frame)
        scores,param=fit_outer(frame,train,held,methods)
        param.update({'dataset':ds,'seed':seed,'modality':mod,'fold':fold['fold']})
        params.append(param); assignments[held]+=1
        e0=frame.iloc[held].copy(); e0['method']='E0'; e0['fold']=fold['fold']; e0['selected_penalty']='none'
        e0['base_score']=np.nan; e0['behavior_score']=0.; e0['reliability_score']=np.nan
        e0['oof_confidence']=e0.confidence; output.append(e0)
        for method,values in scores.items():
            rows=frame.iloc[held].copy(); rows['method']=method; rows['fold']=fold['fold']
            rows['selected_penalty']=param['methods'][method]['selected_penalty'] if method!='E1' else 'off'
            for col,value in values.items(): rows[col]=value
            output.append(rows)
        for method,item in param['methods'].items():
            for penalty,value in item['candidate_brier'].items():
                selections.append({'dataset':ds,'seed':seed,'modality':mod,'fold':fold['fold'],'method':method,
                    'penalty':penalty,'pooled_inner_brier':value,'selected':str(item['selected_penalty'])==penalty})
        print(f'{ds} {seed} {mod} fold {fold["fold"]} completed',flush=True)
    assert (assignments==1).all()
    predicted=pd.concat(output,ignore_index=True).sort_values(KEY+['method']).reset_index(drop=True)
    dump(out/f'fold_parameters/{ds}_seed{seed}_{mod}.json',params)
    rows=[]; shifts=[]; dialogues=[]
    for method,p in predicted.groupby('method',sort=False):
        common={'dataset':ds,'seed':seed,'modality':mod,'method':method}
        rows.append(dict(common,**metrics(p,p.oof_confidence.to_numpy(float))))
        for correct,g in p.groupby('correct'):
            shift=g.oof_confidence.to_numpy(float)-g.confidence.to_numpy(float)
            shifts.append(dict(common,correct=int(correct),n=len(g),original_mean_confidence=float(g.confidence.mean()),
                calibrated_mean_confidence=float(g.oof_confidence.mean()),mean_change=float(shift.mean()),
                raised_fraction=float(np.mean(shift>1e-12)),lowered_fraction=float(np.mean(shift< -1e-12))))
        for dialogue,g in p.groupby('dialogue_id'):
            dialogues.append(dict(common,dialogue_id=str(dialogue),**calibration_metrics(g.oof_confidence,g.correct)))
    return predicted,pd.DataFrame(rows),pd.DataFrame(shifts),pd.DataFrame(dialogues),pd.DataFrame(selections),params


def bootstrap(predictions,comparisons,seeds,modalities):
    result=[]
    for ds in CONFIG['datasets']:
        with np.load(OLD/f'baseline/bootstrap/{ds}_dialogue_draws.npz') as source:
            dialogues=source['dialogue_ids'].astype(str)
            draws=source['draws']
            multiplicities=np.stack([np.bincount(x,minlength=len(dialogues)) for x in draws])
        assert multiplicities.shape==(2000,CONFIG['datasets'][ds]['dialogues'])
        for modality in modalities:
            distributions={}; point={}
            for seed in seeds:
                f=predictions[predictions.dataset.eq(ds)&predictions.seed.eq(seed)&predictions.modality.eq(modality)]
                for method,g in f.groupby('method',sort=False):
                    g=g.sort_values(KEY).reset_index(drop=True)
                    idx=np.searchsorted(dialogues,g.dialogue_id.to_numpy(str))
                    assert np.array_equal(dialogues[idx],g.dialogue_id.to_numpy(str))
                    counts=np.bincount(idx,minlength=len(dialogues)); total=multiplicities@counts
                    q=g.oof_confidence.to_numpy(float); y=g.correct.to_numpy(int)
                    clip=np.clip(q,1e-15,1-1e-15)
                    bins=np.minimum(np.floor(q*15).astype(int),14)
                    residual=np.zeros((len(dialogues),15)); np.add.at(residual,(idx,bins),q-y)
                    distributions[seed,method]={
                        'ece15':np.abs(multiplicities@residual).sum(axis=1)/total,
                        'binary_brier':multiplicities@np.bincount(idx,weights=(q-y)**2,minlength=len(dialogues))/total,
                        'binary_nll':multiplicities@np.bincount(idx,weights=-y*np.log(clip)-(1-y)*np.log1p(-clip),minlength=len(dialogues))/total}
                    point[seed,method]=calibration_metrics(q,y)
            for candidate,reference in comparisons:
                for metric in METRICS:
                    delta=np.array([point[s,candidate][metric]-point[s,reference][metric] for s in seeds])
                    distribution=np.mean([distributions[s,candidate][metric]-distributions[s,reference][metric] for s in seeds],axis=0)
                    result.append({'dataset':ds,'modality':modality,'candidate':candidate,'reference':reference,
                        'metric':metric,'mean_difference':float(delta.mean()),
                        'seed_difference_sample_sd':float(delta.std(ddof=1)) if len(seeds)>1 else np.nan,
                        'improved_seeds':int((delta<0).sum()),'seeds':len(seeds),
                        'ci_low':float(np.quantile(distribution,.025)),'ci_high':float(np.quantile(distribution,.975)),
                        'bootstrap_fraction_improving':float((distribution<0).mean()),
                        'resamples':2000,'unit':'shared_dialogue_across_seeds','refit':False})
    return pd.DataFrame(result)


def aggregate(metrics_frame):
    group=['dataset','modality','method']
    numeric=[c for c in metrics_frame.select_dtypes(include=np.number) if c not in ['seed','utterances','dialogues']]
    rows=[]
    for key,g in metrics_frame.groupby(group,sort=False):
        r=dict(zip(group,key)); r['seeds']=len(g)
        for c in numeric:
            r[c+'_mean']=float(g[c].mean()); r[c+'_sample_sd']=float(g[c].std(ddof=1)) if len(g)>1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def legacy_parity(predicted):
    legacy=read(OLD/'baseline/oof_predictions.csv')
    if 'oof_confidence' not in legacy: print(legacy.columns); raise AssertionError('legacy schema')
    old=legacy[legacy.method.eq('E1')]
    new=predicted[predicted.method.eq('E1')]
    matched=new.merge(old[KEY+['oof_confidence']],on=KEY,validate='one_to_one',suffixes=('_new','_old'))
    assert len(matched)==8180
    rows=[]
    for (ds,seed),g in matched.groupby(['dataset','seed']):
        rows.append({'dataset':ds,'seed':int(seed),'rows':len(g),
            'max_absolute_confidence_difference':float(np.max(np.abs(g.oof_confidence_new-g.oof_confidence_old)))})
    return rows


def reports(out,agg,boots,selections):
    main=agg[agg.method.isin(['E0','E1','M_joint'])][['dataset','method']+[m+'_mean' for m in METRICS]]
    primary=boots[boots.candidate.eq('M_joint')&boots.reference.eq('E1')]
    text='# 原始μ/δ主线五 Seed结果\n\n主方法只使用C与Gaussian μ/δ、Dropout μ/δ四维。μ为带扰动最大置信度权重的预测翻转均值；δ为原最大置信度与扰动最大置信度均值的绝对差。没有D/A、JS、熵、分类间隔或类别条件。\n\n'
    text+='## 五Seed均值\n\n'+markdown_table(main)+'\n\n## 相对E1的行为增量\n\n'+markdown_table(primary[['dataset','metric','mean_difference','improved_seeds','ci_low','ci_high']])+'\n\n'
    text+='差值为主模型减E1，负值更好。标准差见aggregate_metrics.csv；完整消融均报告，不按最优结果替换M_joint。Bootstrap共2000次，以Dialogue为配对单位、五Seed共享抽样，再平均五Seed指标差；区间不含校准器重新拟合不确定性。所有bootstrap区间均为探索性区间，未做多重比较调整。\n\n'
    text+='## 评分与校准\n\nR=a+bC+Z(B)w，R_behavior=Z(B)w，q=sigmoid(R)。基础C与行为权重联合学习。λ由外训练集内GroupKFold5的合并留出Brier选定。仅行为列标准化；off精确退回同训练子集的E1。E1优化精度与旧实现的差异单独报告，不能记为行为收益。\n\n'
    text+='## 限制\n\n同一Validation已用于XP04 checkpoint选择及多轮方法设计，外层Dialogue OOF只隔离校准器拟合和选参，不是独立确认。所有Seed共享数据，不能当五份独立数据。仅调整置信度，Accuracy/Weighted-F1/Macro-F1不变；没有GPU推理、CMERC训练、重新扰动、Test标签或原始数据修改。\n'
    (out/'MAINLINE_RESULTS.md').write_text(text)
    ablation='# 预登记原μ/δ输入消融\n\n'+markdown_table(agg[['dataset','method']+[m+'_mean' for m in METRICS]])+'\n\n所有消融使用同一评分结构和内外Dialogue折。M_mu检验翻转不稳定性，M_delta检验置信度不稳定性；两种干扰的μδ分别输入，不混成独立话语。未把单个最优消融作为主模型。\n'
    (out/'MU_DELTA_ABLATION.md').write_text(ablation)
    decisions=[]
    for ds,g in primary.groupby('dataset'):
        x=g.set_index('metric'); improvement={m:float(x.loc[m,'mean_difference']) for m in METRICS}
        signal=improvement['ece15']<0 and min(improvement['binary_brier'],improvement['binary_nll'])<0 and max(improvement['binary_brier'],improvement['binary_nll'])<=0
        proper=[m for m in ['binary_brier','binary_nll'] if improvement[m]<0]
        signal=bool(signal and any(int(x.loc[m,'improved_seeds'])>=3 for m in proper))
        decisions.append(f'- **{ds}**：'+('达到预登记的初步校准信号条件' if signal else '未达到预登记的初步校准信号条件')+'；区间'+('全部位于改善方向' if (g.ci_high<0).all() else '仍有跨零或反向风险')+'。')
    decision='# 三层证据与后续边界\n\n## 工程证据\n\n本批仅复用已审计的随机特征干扰结果，按每条话语一条μ/δ记录学习评分。内外层模型参数、Dialogue IDs和分数完整保存；最终以独立验证报告为准。\n\n## 行为响应证据\n\n预测翻转及置信度变化属于可测响应，不等同于预测错误。原始响应统计见输入审计及后续单模态Pilot，不能将模型敏感性直接视为可靠性。\n\n## 校准增量证据\n\n'+'\n'.join(decisions)+'\n\n即使存在微弱探索性增量，也不能称为已经证明校准有效；重复Validation无法提供独立确认。可测行为或弱信号可以支持另行登记面部遮挡/音色处理的小规模机制与工程Pilot动机，不直接支持完整原始数据实验。本次不自动执行原始数据修改。\n'
    (out/'FINAL_MAINLINE_DECISION.md').write_text(decision)


def snapshots(out,inputs):
    paths=[ROOT/'preregistration.json',ROOT/'MAINLINE_PROTOCOL.md',*sorted((ROOT/'scripts').glob('*.py')),
        *inputs,ROOT/'inputs/verification.json',OLD/'baseline/fold_parameters.json',OLD/'baseline/oof_predictions.csv']
    paths += [OLD/f'baseline/bootstrap/{ds}_dialogue_draws.npz' for ds in CONFIG['datasets']]
    sources={str(p):sha(p) for p in paths}
    dump(out/'provenance/source_hashes.json',sources)
    dump(out/'provenance/environment.json',{'python':sys.executable,'host':platform.node(),'numpy':np.__version__,
        'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'argv':sys.argv,'device':'CPU'})
    return sources


def verify_snapshot(sources):
    assert all(sha(path)==value for path,value in sources.items()),'Input, protocol or implementation changed during run'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',action='store_true'); args=parser.parse_args()
    assert args.run,'Formal execution requires explicit --run after parent input gate'
    assert CONFIG['methods']==METHODS and CONFIG['features']==FEATURES
    out=ROOT/'joint'; assert not (out/'verification.json').exists(),'Refuse overwrite of prior run'
    out.mkdir(exist_ok=True); started=time.time()
    f=checked_input(ROOT/'inputs/joint_features.csv')
    sources=snapshots(out,[ROOT/'inputs/joint_features.csv'])
    outputs=[[] for _ in range(5)]; all_params=[]
    for ds in CONFIG['datasets']:
        for seed in SEEDS:
            part=f[f.dataset.eq(ds)&f.seed.eq(seed)].reset_index(drop=True)
            result=run_group(part,list(FEATURES),out)
            for target,value in zip(outputs,result[:5]): target.append(value)
            all_params.extend(result[5])
    predicted,per_seed,shift,dialogue,selections=[pd.concat(parts,ignore_index=True) for parts in outputs]
    assert len(predicted)==57260 and len(per_seed)==70 and len(all_params)==85
    for name,df in [('oof_predictions.csv',predicted),('per_seed_metrics.csv',per_seed),('correct_wrong_adjustment.csv',shift),
        ('per_dialogue_results.csv',dialogue),('inner_cv_results.csv',selections)]: df.to_csv(out/name,index=False)
    agg=aggregate(per_seed); agg.to_csv(out/'aggregate_metrics.csv',index=False)
    boots=bootstrap(predicted,CONFIG['joint_comparisons'],SEEDS,['joint']); assert len(boots)==54
    boots.to_csv(out/'bootstrap_results.csv',index=False)
    dump(out/'legacy_e1_precision_comparison.json',{'note':'optimizer precision only; not evidence of behavior gain','comparisons':legacy_parity(predicted)})
    selected=[dict(dataset=p['dataset'],seed=p['seed'],fold=p['fold'],method=m,selected_penalty=v['selected_penalty']) for p in all_params for m,v in p['methods'].items()]
    pd.DataFrame(selected).to_csv(out/'selected_penalties.csv',index=False)
    reports(out,agg,boots,selections)
    verify_snapshot(sources)
    dump(out/'verification.json',{'status':'completed','independent_verification_pending':True,'cpu_only':True,
        'feature_rows':len(f),'oof_rows':len(predicted),'metric_rows':len(per_seed),'outer_folds':len(all_params),
        'bootstrap_rows':len(boots),'input_and_code_hashes_unchanged':True,'elapsed_seconds':time.time()-started,
        'methods':METHODS,'primary_method':'M_joint','training_record_per_utterance':1,'test_labels_used':False,
        'cmerc_training':False,'new_perturbations':False,'raw_data_modified':False})
    print(agg[['dataset','method']+[m+'_mean' for m in METRICS]].to_string(index=False),flush=True)

if __name__=='__main__': main()
