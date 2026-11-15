"""Seed1475 modality Pilot: identical core and nested score, no modality selection."""
import argparse
import time
import numpy as np
import pandas as pd
from mainline_core import dump, markdown_table
from run_joint import ROOT, CONFIG, KEY, METRICS, aggregate, bootstrap, checked_input, read, run_group, snapshots, verify_snapshot


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',action='store_true'); args=parser.parse_args()
    assert args.run,'Formal execution requires explicit --run after joint batch approved'
    joint=ROOT/'joint'
    assert (joint/'verification.json').exists()
    out=ROOT/'modality'; assert not (out/'verification.json').exists(),'Refuse overwrite of completed run'
    out.mkdir(exist_ok=True); started=time.time()
    f=checked_input(ROOT/'inputs/modality_features.csv',modality=True)
    sources=snapshots(out,[ROOT/'inputs/modality_features.csv',joint/'oof_predictions.csv'])
    outputs=[[] for _ in range(5)]; joint_errors=[]; all_params=[]
    reference=read(joint/'oof_predictions.csv')
    for ds in CONFIG['datasets']:
        for modality in ['text','audio','visual','joint']:
            part=f[f.dataset.eq(ds)&f.modality.eq(modality)].reset_index(drop=True)
            result=run_group(part,['M_joint'],out)
            p=result[0]
            if modality=='joint':
                ref=reference[reference.dataset.eq(ds)&reference.seed.eq(1475)&reference.method.isin(['E0','E1','M_joint'])]
                joined=p.merge(ref[KEY+['method','oof_confidence']],on=KEY+['method'],validate='one_to_one',suffixes=('_new','_reference'))
                assert len(joined)==3*len(part)
                error=float(np.max(np.abs(joined.oof_confidence_new-joined.oof_confidence_reference)))
                assert error<1e-12
                joint_errors.append({'dataset':ds,'rows':len(joined),'max_absolute_error':error})
            for target,value in zip(outputs,result[:5]): target.append(value)
            all_params.extend(result[5])
    predicted,per_seed,shift,dialogue,selections=[pd.concat(x,ignore_index=True) for x in outputs]
    assert len(predicted)==19632 and len(per_seed)==24 and len(all_params)==68
    for name,df in [('oof_predictions.csv',predicted),('per_seed_metrics.csv',per_seed),('correct_wrong_adjustment.csv',shift),
        ('per_dialogue_results.csv',dialogue),('inner_cv_results.csv',selections)]: df.to_csv(out/name,index=False)
    agg=aggregate(per_seed); agg.to_csv(out/'aggregate_metrics.csv',index=False)
    boots=bootstrap(predicted,[['M_joint','E1']],[1475],['text','audio','visual','joint'])
    assert len(boots)==24
    boots.to_csv(out/'bootstrap_results.csv',index=False)
    pd.DataFrame([dict(dataset=p['dataset'],seed=p['seed'],modality=p['modality'],fold=p['fold'],
        method=m,selected_penalty=v['selected_penalty']) for p in all_params for m,v in p['methods'].items()]).to_csv(out/'selected_penalties.csv',index=False)
    text='# 原μ/δ单模态技术Pilot\n\n仅Seed1475，text/audio/visual/joint分别学习同一四维μδ评分；每个模态每条原始话语一条记录，不把条件和随机重复当独立样本。\n\n'
    text+=markdown_table(per_seed[['dataset','modality','method']+METRICS])+'\n\n## 相对E1的配对差\n\n'
    text+=markdown_table(boots[['dataset','modality','metric','mean_difference','ci_low','ci_high']])+'\n\n'
    text+='各模态为预先登记探索，不能选择赢家替换五Seed joint主结果。单Seed不存在跨模型Seed确认，样本标准差标记为NA，不能将其解释为无变异。视觉缺失样本保留，原始跳过视觉扰动产生的零响应不能解释为视觉可靠。joint逐条复现第一批Seed1475结果。\n\n'
    text+='分数R=a+bC+Z(B)w，q=sigmoid(R)，μ/δ输入与联合训练方案和五Seed完全相同；每次拟合/标准化仅使用相应训练Dialogue。区间按Dialogue配对抽样2000次，不含重新拟合。所有bootstrap区间均为探索性区间，未做多重比较调整。当前Validation已反复分析；只能作为探索性机制信息。\n'
    (out/'MODALITY_PILOT_REPORT.md').write_text(text)
    verify_snapshot(sources)
    dump(out/'joint_reproduction.json',joint_errors)
    dump(out/'verification.json',{'status':'completed','independent_verification_pending':True,'cpu_only':True,
        'feature_rows':len(f),'oof_rows':len(predicted),'metric_rows':len(per_seed),'outer_folds':len(all_params),
        'bootstrap_rows':len(boots),'input_and_code_hashes_unchanged':True,'joint_reproduction':joint_errors,
        'elapsed_seconds':time.time()-started,'methods':['E0','E1','M_joint'],'test_labels_used':False,
        'cmerc_training':False,'new_perturbations':False,'raw_data_modified':False})
    print(per_seed[['dataset','modality','method']+METRICS].to_string(index=False),flush=True)

if __name__=='__main__': main()
