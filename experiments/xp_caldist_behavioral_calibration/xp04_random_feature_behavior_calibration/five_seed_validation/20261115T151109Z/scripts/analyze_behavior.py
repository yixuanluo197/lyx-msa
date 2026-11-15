"""Frozen five-seed joint-behavior validation; CPU, one training row per utterance."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

for thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_key] = "1"

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
import sklearn
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'behavior'
OLD = ROOT.parents[1] / 'unified_module_pilot/20261115T145425Z'
PILOT = ROOT.parents[1] / 'pilot/20261113T025053Z'
CORE = ROOT / 'provenance/analysis_code/calibration_core.py'
spec = importlib.util.spec_from_file_location('frozen_calibration_core', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
ROWKEY = ['dialogue_id', 'utterance_index']
SEEDS = [1475, 67137, 2024, 3407, 42]
DATASETS = ['IEMOCAP', 'MELD']
METHODS = ['XP04', 'E1', 'L0', 'E1_MUDELTA', 'E1_DA', 'L0_MUDELTA', 'L0_DA']
FEATURES = {'DA': ['gaussian_D', 'gaussian_A', 'dropout_D', 'dropout_A'],
            'MUDELTA': ['gaussian_mu', 'gaussian_delta', 'dropout_mu', 'dropout_delta']}
METRICS = ['ece15', 'binary_brier', 'binary_nll']
COMPARISONS = [(m, 'XP04') for m in METHODS if m != 'XP04'] + [
    (f'{base}_{variant}', base) for base in ['E1', 'L0'] for variant in ['MUDELTA', 'DA']
] + [('E1_DA', 'E1_MUDELTA'), ('L0_DA', 'L0_MUDELTA')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def read_csv(path):
    return pd.read_csv(path, dtype={'dialogue_id': str})


def input_paths():
    sources = [ROOT/'preregistration.json', CORE, ROOT/'baseline/original_predictions.csv',
               ROOT/'baseline/oof_predictions.csv', OLD/'feature_matrix.csv', OLD/'oof_predictions.csv',
               OLD/'metrics.csv', PILOT/'original_predictions.csv']
    sources.extend(ROOT/f'baseline/bootstrap/{ds}_dialogue_draws.npz' for ds in DATASETS)
    for ds in DATASETS:
        for seed in SEEDS[1:]:
            folder = ROOT/f'runs/{ds}/seed{seed}'
            sources.extend(folder/n for n in ['original_predictions.csv', 'perturbed_predictions.csv', 'complete.json'])
    return sources


def preflight():
    missing, incomplete = [], []
    for p in input_paths():
        if not p.is_file():
            missing.append(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
        elif p.name == 'complete.json':
            try:
                assert json.loads(p.read_text())['status'] == 'completed'
            except Exception:
                incomplete.append(str(p.relative_to(ROOT)))
    return {'ready': not missing and not incomplete, 'missing': missing, 'incomplete': incomplete,
            'seed1475': 'reuse frozen previous Pilot feature matrix, no new inference'}


def checked_original(original, dataset, seed, canonical):
    f = original[original.dataset.eq(dataset) & original.seed.eq(seed)].copy().reset_index(drop=True)
    expected_n, expected_g = (527, 12) if dataset == 'IEMOCAP' else (1109, 114)
    assert len(f) == expected_n and f.dialogue_id.nunique() == expected_g
    assert not f.duplicated(KEY).any()
    for row in f.itertuples():
        p, lp = np.asarray(json.loads(row.probabilities), float), np.asarray(json.loads(row.log_probabilities), float)
        assert len(p) == (6 if dataset == 'IEMOCAP' else 7)
        assert np.isfinite(p).all() and np.isfinite(lp).all() and abs(p.sum()-1) < 1e-5
        assert np.allclose(np.exp(lp), p, atol=1e-6, rtol=1e-5)
        assert int(p.argmax()) == row.prediction and abs(p.max()-row.confidence) < 1e-12
    ref = canonical[canonical.dataset.eq(dataset) & canonical.seed.eq(seed)]
    aligned = f.merge(ref, on=KEY, validate='one_to_one', suffixes=('', '_canonical'))
    assert len(aligned) == expected_n
    for col in ['label', 'prediction']:
        assert aligned[col].equals(aligned[col+'_canonical'])
    assert np.max(abs(aligned.confidence-aligned.confidence_canonical)) < 1e-12
    assert aligned.probabilities.equals(aligned.probabilities_canonical)
    return f


def build_feature(dataset, seed, canonical):
    if seed == 1475:
        original = checked_original(read_csv(PILOT/'original_predictions.csv'), dataset, seed, canonical)
        feature = read_csv(OLD/'feature_matrix.csv')
        feature = feature[feature.dataset.eq(dataset)].copy().reset_index(drop=True)
        assert feature[KEY].equals(original[KEY])
        assert np.max(abs(feature.confidence-original.confidence)) < 1e-12
        assert feature.label.equals(original.label) and feature.prediction.equals(original.prediction)
    else:
        folder = ROOT/f'runs/{dataset}/seed{seed}'
        complete = json.loads((folder/'complete.json').read_text())
        assert complete['status'] == 'completed' and complete['dataset'] == dataset and complete['seed'] == seed
        assert complete['weights_unchanged'] and complete['batch_inputs_restored'] and complete['parity']['passed']
        original = checked_original(read_csv(folder/'original_predictions.csv'), dataset, seed, canonical)
        raw = read_csv(folder/'perturbed_predictions.csv')
        assert len(raw) == len(original)*20 == complete['perturbed_rows']
        assert set(raw.modality) == {'joint'} and set(raw.corruption) == {'gaussian', 'dropout'}
        assert raw.strength.eq(.05).all() and raw.perturbation_seed.eq(1475).all()
        assert not raw.duplicated(KEY+['corruption','repeat_id']).any()
        raw = raw.merge(original[KEY+['prediction','confidence']].rename(columns={
            'prediction':'original_prediction','confidence':'original_confidence'}), on=KEY, validate='many_to_one')
        assert len(raw) == len(original)*20
        probabilities = [np.asarray(json.loads(x), float) for x in raw.probabilities]
        logp = [np.asarray(json.loads(x), float) for x in raw.log_probabilities]
        for p, lp, row in zip(probabilities, logp, raw.itertuples()):
            assert np.isfinite(p).all() and np.isfinite(lp).all() and abs(p.sum()-1) < 1e-5
            assert np.allclose(np.exp(lp), p, atol=1e-12, rtol=1e-12)
            assert int(p.argmax()) == row.prediction and abs(p.max()-row.confidence) < 1e-12
        raw['q'] = [p[int(k)] for p, k in zip(probabilities, raw.original_prediction)]
        rows = []
        for key, g in raw.groupby(KEY+['corruption'], sort=False):
            g = g.sort_values('repeat_id')
            assert g.repeat_id.tolist() == list(range(10)) and g.visual_missing_skipped.nunique() == 1
            c, q = float(g.original_confidence.iloc[0]), g.q.to_numpy(float)
            rows.append({**dict(zip(KEY+['corruption'], key)), 'D':float(c-q.mean()),
                'A':float(np.sqrt(np.mean((q-c)**2))),
                'mu':float(np.mean(g.prediction.ne(g.original_prediction)*g.confidence)),
                'delta':float(abs(c-g.confidence.mean())),
                'visual_missing':bool(g.visual_missing_skipped.iloc[0])})
        long = pd.DataFrame(rows)
        feature = original.drop(columns=['probabilities','log_probabilities']).copy()
        for kind in ['gaussian','dropout']:
            part = long[long.corruption.eq(kind)].drop(columns='corruption').rename(columns={
                n:f'{kind}_{n}' for n in ['D','A','mu','delta','visual_missing']})
            feature = feature.merge(part,on=KEY,validate='one_to_one',how='left',sort=False)
        assert feature.gaussian_visual_missing.equals(feature.dropout_visual_missing)
        feature['visual_missing'] = feature.gaussian_visual_missing
        feature['correct'] = feature.prediction.eq(feature.label).astype(int)
    assert np.isfinite(feature[sum(FEATURES.values(),[])].to_numpy(float)).all()
    assert len(feature) == len(original) and not feature.duplicated(KEY).any()
    assert feature.correct.eq(feature.prediction.eq(feature.label).astype(int)).all()
    assert int(feature.visual_missing.sum()) == (1 if dataset == 'IEMOCAP' else 95)
    return feature


def run_dataset(feature, dataset, seed, verbose=True):
    f = feature.reset_index(drop=True)
    c, y, groups = f.confidence.to_numpy(float), f.correct.to_numpy(int), f.dialogue_id.to_numpy(str)
    splits = list((LeaveOneGroupOut() if dataset == 'IEMOCAP' else GroupKFold(n_splits=5)).split(f,groups=groups))
    scores = {m:np.full(len(f),np.nan) for m in METHODS}
    scores['XP04'] = c.copy()
    foldids, parameters, choices = np.full(len(f),-1,int), [], []
    for fold,(tr,ho) in enumerate(splits):
        train_groups, held_groups = sorted(set(groups[tr])), sorted(set(groups[ho]))
        assert set(train_groups).isdisjoint(held_groups)
        foldids[ho] = fold
        shared = {'dataset':dataset,'seed':seed,'fold':fold,'train_dialogues':train_groups,'held_dialogues':held_groups}
        for base in ['E1','L0']:
            bp = core.fit_base(c[tr],y[tr],base)
            scores[base][ho] = expit(core.base_logits(c[ho],bp))
            parameters.append({**shared,'method':base,'base':bp})
            for variant in ['MUDELTA','DA']:
                method = f'{base}_{variant}'
                b = f[FEATURES[variant]].to_numpy(float)
                fit = core.fit_nested_fold(c[tr],b[tr],y[tr],groups[tr],base)
                assert fit['base'] == bp
                scores[method][ho] = core.predict(c[ho],b[ho],fit)
                parameters.append({**shared,'method':method,'features':FEATURES[variant],**fit})
                for candidate in fit['candidates']:
                    choices.append({'dataset':dataset,'seed':seed,'method':method,'outer_fold':fold,
                        'penalty':candidate['penalty'],'selected':candidate['penalty']==fit['selected_penalty'],
                        **{k:candidate[k] for k in METRICS}})
        if verbose:
            print(json.dumps({'dataset':dataset,'seed':seed,'outer_fold_completed':fold+1,'folds':len(splits)}),flush=True)
    assert (foldids>=0).all()
    predicted, metrics, shifts, dialogues = [], [], [], []
    for method,p in scores.items():
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        rows = f[KEY+['label','prediction','correct','visual_missing']].copy()
        rows['fold'], rows['method'] = foldids, method
        rows['original_confidence'], rows['oof_confidence'], rows['shift'] = c,p,p-c
        predicted.append(rows)
        metrics.append({'dataset':dataset,'seed':seed,'method':method,'utterances':len(f),'dialogues':len(set(groups)),
            **core.metrics(p,y),**core.auxiliary(p,y,c,f.label,f.prediction)})
        for correct in [0,1]:
            mask = y==correct
            shifts.append({'dataset':dataset,'seed':seed,'method':method,'correct':correct,'n':int(mask.sum()),
                'mean_original':float(c[mask].mean()),'mean_calibrated':float(p[mask].mean()),
                'mean_shift':float((p[mask]-c[mask]).mean()),'raised_fraction':float(np.mean(p[mask]>c[mask]+1e-12)),
                'lowered_fraction':float(np.mean(p[mask]<c[mask]-1e-12))})
        for dialogue in sorted(set(groups)):
            mask = groups==dialogue
            dialogues.append({'dataset':dataset,'seed':seed,'dialogue_id':dialogue,'method':method,
                              'n':int(mask.sum()),'accuracy':float(y[mask].mean()),**core.metrics(p[mask],y[mask])})
    return pd.concat(predicted,ignore_index=True),pd.DataFrame(metrics),pd.DataFrame(shifts),pd.DataFrame(dialogues),parameters,pd.DataFrame(choices)


def check_parity(predictions, seed, dataset):
    if seed == 1475:
        reference = read_csv(OLD/'oof_predictions.csv')
        methods = METHODS
    else:
        reference = read_csv(ROOT/'baseline/oof_predictions.csv')
        methods = ['XP04','E1','L0']
    current = predictions[predictions.method.isin(methods)]
    ref = reference[reference.dataset.eq(dataset)&reference.seed.eq(seed)&reference.method.isin(methods)]
    joined = current.merge(ref,on=KEY+['method'],suffixes=('','_reference'),validate='one_to_one')
    assert len(joined) == len(current) == len(ref)
    assert joined.fold.equals(joined.fold_reference)
    error = float(np.max(abs(joined.oof_confidence-joined.oof_confidence_reference)))
    assert error < 1e-12,(dataset,seed,error)
    return error


def shared_bootstrap(predictions):
    rows, all_draws = [], []
    for dataset in DATASETS:
        pds = predictions[predictions.dataset.eq(dataset)]
        groups = np.sort(pds.dialogue_id.unique())
        with np.load(ROOT/f'baseline/bootstrap/{dataset}_dialogue_draws.npz') as z:
            assert np.array_equal(z['dialogue_ids'], groups) and z['draws'].shape == (2000,len(groups))
            draw = np.stack([np.bincount(indices,minlength=len(groups)) for indices in z['draws']])
        all_draws.append({'dataset':dataset,'dialogue_ids':groups.tolist(),'multiplicities':draw.tolist()})
        pooled, full = {}, {}
        for seed in SEEDS:
            ps = pds[pds.seed.eq(seed)]
            base = ps[ps.method.eq('XP04')].reset_index(drop=True)
            idx = np.searchsorted(groups,base.dialogue_id.to_numpy(str))
            counts = np.bincount(idx,minlength=len(groups))
            total = draw@counts
            y = base.correct.to_numpy(int)
            for method in METHODS:
                pm = base[KEY].merge(ps[ps.method.eq(method)][KEY+['oof_confidence']],on=KEY,validate='one_to_one')
                p = np.clip(pm.oof_confidence.to_numpy(float),1e-15,1-1e-15)
                logloss = -(y*np.log(p)+(1-y)*np.log1p(-p))
                bids = np.minimum(np.floor(p*15).astype(int),14)
                residual = np.zeros((len(groups),15)); np.add.at(residual,(idx,bids),p-y)
                pooled[seed,method] = {'ece15':np.abs(draw@residual).sum(axis=1)/total,
                    'binary_brier':(draw@np.bincount(idx,weights=(p-y)**2,minlength=len(groups)))/total,
                    'binary_nll':(draw@np.bincount(idx,weights=logloss,minlength=len(groups)))/total}
                full[seed,method] = core.metrics(p,y)
        for candidate,reference in COMPARISONS:
            for metric in METRICS:
                changes = [full[s,candidate][metric]-full[s,reference][metric] for s in SEEDS]
                distribution = np.mean([pooled[s,candidate][metric]-pooled[s,reference][metric] for s in SEEDS],axis=0)
                rows.append({'dataset':dataset,'candidate':candidate,'reference':reference,'metric':metric,
                    'mean_difference':float(np.mean(changes)),'seed_difference_sample_sd':float(np.std(changes,ddof=1)),
                    'improved_seeds':int(np.sum(np.array(changes)<0)),'seeds':5,
                    'ci_low':float(np.quantile(distribution,.025)),'ci_high':float(np.quantile(distribution,.975)),
                    'bootstrap_fraction_improving':float(np.mean(distribution<0)),
                    'resamples':2000,'unit':'shared_dialogue_across_seeds','refit':False})
    return pd.DataFrame(rows),all_draws


def summaries(metrics,params):
    cols = [c for c in metrics.select_dtypes(include=np.number).columns if c not in ['seed','utterances','dialogues']]
    records=[]
    for (dataset,method),g in metrics.groupby(['dataset','method'],sort=False):
        assert set(g.seed)==set(SEEDS)
        record={'dataset':dataset,'method':method,'seeds':len(g),'utterances_per_seed':int(g.utterances.iloc[0])}
        for col in cols:
            record[col+'_mean']=float(g[col].mean())
            record[col+'_sample_sd']=float(g[col].std(ddof=1))
        records.append(record)
    selected = pd.DataFrame([{'dataset':r['dataset'],'seed':r['seed'],'method':r['method'],'fold':r['fold'],
                             'selected_penalty':r['selected_penalty']} for r in params if 'correction' in r])
    counts=[]
    for (ds,seed,method),g in selected.groupby(['dataset','seed','method'],sort=False):
        counts.append({'dataset':ds,'seed':seed,'method':method,'outer_folds':len(g),
                       'off_folds':int(g.selected_penalty.eq('off').sum()),
                       'active_folds':int(g.selected_penalty.ne('off').sum())})
    return pd.DataFrame(records),selected,pd.DataFrame(counts)


def reports(metrics,aggregate,bootstrap,off):
    def table(frame):
        columns=list(frame.columns)
        lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
        for row in frame.itertuples(index=False,name=None):
            lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    main=aggregate[aggregate.method.isin(['XP04','E1','L0','E1_DA'])][['dataset','method']+[m+'_mean' for m in METRICS]]
    text='# XP04 五 Seed 固定行为模块结果\n\n'
    text+='仅使用真正 Validation：IEMOCAP 每 Seed 527 条 / 12 Dialogue；MELD 每 Seed 1109 条 / 114 Dialogue。原模型和预测类别保持不变。\n\n'
    text+='主方法为预先登记的 E1_DA；Gaussian 与 Dropout 的 D/A 分开共 4 维。IEMOCAP 外层 LODO12、MELD GroupKFold5；行为分支正则强度在每个外训练集的内 GroupKFold5 中按 Brier 选择 off / 1 / 0.1 / 0.01。\n\n'
    text+='## 五 Seed 指标均值\n\n'+table(main)+'\n\n'
    text+='完整均值和样本标准差见 aggregate_metrics.csv；逐 Seed 全部结果见 per_seed_metrics.csv。ECE 为固定15区间，Brier/NLL 为 Top-label 二元正确性口径。\n\n'
    text+='## 主方法与 XP04 / E1 的共享 Dialogue bootstrap\n\n'
    relevant=bootstrap[bootstrap.candidate.eq('E1_DA')&bootstrap.reference.isin(['XP04','E1'])]
    text+=table(relevant[['dataset','reference','metric','mean_difference','improved_seeds','ci_low','ci_high']])+'\n\n'
    text+='差值为候选减对照，负值更好。同一次抽到的 Dialogue 重数用于全部五 Seed，再对五个 Seed 的指标差取均值；没有把五 Seed 当作独立数据。2000次区间不包含重新拟合不确定性。\n\n'
    text+='## 行为分支关闭比例\n\n'+table(off[off.method.eq('E1_DA')])+'\n\n'
    text+='Seed1475复用前轮冻结特征并逐条复现7方法 OOF；其余Seed从新完成的 joint2type×10repeat 概率计算。共享同一扰动随机种子1475用于配对。视觉缺失样本保留，其联合扰动仍作用于可用文本/音频。\n\n'
    text+='这是固定配置的后续确认；Seed1475曾参与设计，四个新Seed为新增验证。只有一个固定Validation划分，不能把本结果视作独立测试集验证。外层 Dialogue OOF 仅隔离校准器拟合和调参；XP04 best.pt 本身曾按 Validation 选择 checkpoint，因此不等价于完全未参与模型选择的新 Dialogue 泛化确认。五 Seed 共享 Dialogue，跨 Seed 稳定性也不是新数据证据。\n'
    (OUT/'DATA_RESULTS_REPORT.md').write_text(text)
    decisions=[]
    for ds in DATASETS:
        b=bootstrap[bootstrap.dataset.eq(ds)&bootstrap.candidate.eq('E1_DA')]
        x=b[b.reference.eq('XP04')].set_index('metric')
        e=b[b.reference.eq('E1')].set_index('metric')
        point_pass=all(x.loc[m,'mean_difference']<0 for m in METRICS)
        behavior_gain=all(e.loc[m,'mean_difference']<0 for m in METRICS)
        robust_x=all(x.loc[m,'ci_high']<0 for m in METRICS)
        robust_e=all(e.loc[m,'ci_high']<0 for m in METRICS)
        decisions.append(f'- **{ds}**：主方法相对 XP04 三指标均值'+('均改善' if point_pass else '未全部改善')+
            '；相对 E1 '+('三指标均值均改善' if behavior_gain else '没有三指标一致增量')+
            '。共享 Dialogue bootstrap 对 XP04 '+('三指标均支持降低' if robust_x else '仍存在跨零或反向区间')+
            '，对 E1 '+('三指标均支持降低' if robust_e else '尚无三指标一致的统计支持')+'。')
    decision='# 五 Seed 最终判断\n\n'+'\n'.join(decisions)+'\n\n'
    decision+='研究判断须区分「整体模块优于原 XP04」和「扰动行为在基础置信度映射之外的贡献」。不能因某个消融最好而替换预登记 E1_DA 主结果。若行为分支多数关闭，说明已测局部随机扰动在当前输入和固定协议下缺少可用增量，并非可靠性由敏感性直接决定。\n\n'
    decision+='本阶段停止并等待研究者审核。当前结果不自动授权原始文本、音频、视频干扰。\n'
    (OUT/'FINAL_DECISION.md').write_text(decision)
    ablation='# 固定消融与分支使用\n\n'+table(aggregate[['dataset','method']+[m+'_mean' for m in METRICS]])+'\n\n'
    ablation+='μ/δ 为原 CaliDist 定义，D/A 为本研究适配；没有平均两种干扰、添加JS、类别或重新选择扰动强度。四个行为变体共享完全相同的内外Dialogue划分。\n\n'+table(off)+'\n'
    (OUT/'BEHAVIOR_ABLATION_REPORT.md').write_text(ablation)
    spread = aggregate[['dataset','method']].copy()
    for metric in METRICS:
        spread[metric] = [f'{r[metric+"_mean"]:.6f} ± {r[metric+"_sample_sd"]:.6f}'
                          for r in aggregate.to_dict('records')]
    cross = '# 五 Seed 稳定性分析\n\n'+table(spread)+'\n\n'
    cross += '以上为五 Seed 均值 ± 样本标准差（ddof=1），不把标准差当作独立样本置信区间。\n\n'
    cross += table(bootstrap[bootstrap.candidate.eq('E1_DA')][['dataset','reference','metric','mean_difference','improved_seeds','ci_low','ci_high']])+'\n\n'
    cross += 'Seed1475参与过方法设计；其余四Seed使用冻结配置。五Seed共享Dialogue，bootstrap以Dialogue为重采样单位，五Seed共用抽样后先各自算指标差，再取均值。区间不包括重新训练校准器及XP04 checkpoint选择的不确定性。\n'
    (OUT/'CROSS_SEED_ANALYSIS.md').write_text(cross)
    adjustment_cols = ['dataset','method','mean_confidence_mean','correct_mean_confidence_mean',
                       'wrong_mean_confidence_mean','raised_fraction_mean','lowered_fraction_mean','correctness_auc_mean']
    adjustment = '# 正确与错误预测的置信度调整\n\n'+table(aggregate[aggregate.method.isin(['XP04','E1','L0','E1_DA'])][adjustment_cols])+'\n\n'
    adjustment += '表中均值均先按Seed统计再取五Seed均值。更细的正确/错误样本变化量与上调比例见correct_wrong_adjustment.csv。推理不知道真实正确性，不能要求每个错误样本必然降置信度；是否合理仍以OOF二元正确性校准及配对对照评价。\n'
    (OUT/'CONFIDENCE_ADJUSTMENT_REPORT.md').write_text(adjustment)


def self_test():
    canonical=read_csv(ROOT/'baseline/original_predictions.csv')
    results=[]
    for ds in DATASETS:
        f=build_feature(ds,1475,canonical)
        p,m,s,d,params,c=run_dataset(f,ds,1475,verbose=False)
        err=check_parity(p,1475,ds)
        results.append({'dataset':ds,'utterances':len(f),'oof_rows':len(p),'max_prior_oof_error':err,
                        'outer_parameters':len(params),'nested_selection_decisions':len(c)//4})
    report={'engineering_fixture_only':True,'formal_five_seed_complete':False,'seed1475_frozen_method_reproduced':True,
            'datasets':results,'core_sha256':sha(CORE),'python':sys.executable}
    write_json(OUT/'engineering_self_test.json',report)
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    assert np.__version__=='1.26.4' and pd.__version__=='1.5.3' and scipy.__version__=='1.11.4' and sklearn.__version__=='1.3.2'
    config=json.loads((ROOT/'preregistration.json').read_text())
    assert config['seeds']==SEEDS and config['calibration_methods']==METHODS
    assert config['primary_method']=='E1_DA' and core.LAMBDAS==['off',1.,.1,.01]
    assert sha(CORE)==sha(OLD/'calibration_core.py')
    if args.preflight:
        result=preflight(); print(json.dumps(result,indent=2)); return 0 if result['ready'] else 2
    if args.self_test:
        self_test(); return 0
    ready=preflight()
    assert ready['ready'], 'All ten completed inputs required: '+json.dumps(ready)
    if (OUT/'verification.json').exists():
        assert json.loads((OUT/'verification.json').read_text()).get('status')!='completed','Refusing to overwrite completed analysis'
    started=time.time()
    OUT.mkdir(exist_ok=True)
    sources={str(p):sha(p) for p in input_paths()}
    code={str(Path(__file__)):sha(__file__),str(ROOT/'scripts/verify_behavior.py'):sha(ROOT/'scripts/verify_behavior.py')}
    write_json(OUT/'provenance/source_hashes.json',sources)
    write_json(OUT/'provenance/run_snapshot.json',{'start_unix':started,'python':sys.executable,'host':platform.node(),
        'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
        'argv':sys.argv,'device':'CPU','code_sha256':code})
    write_json(OUT/'verification.json',{'status':'running','all_inputs_complete':True})
    canonical=read_csv(ROOT/'baseline/original_predictions.csv')
    output=[[] for _ in range(6)]; all_parameters=[]; checks=[]
    for ds in DATASETS:
        for seed in SEEDS:
            f=build_feature(ds,seed,canonical)
            result=run_dataset(f,ds,seed)
            p,m,s,d,params,cv=result
            err=check_parity(p,seed,ds)
            write_json(OUT/f'fold_parameters/{ds}_seed{seed}.json',params)
            for target,value in zip(output,[f,p,m,s,d,cv]): target.append(value)
            all_parameters.extend(params)
            checks.append({'dataset':ds,'seed':seed,'utterances':len(f),'dialogues':f.dialogue_id.nunique(),
                'visual_missing_retained':int(f.visual_missing.sum()),'prior_base_parity_max_error':err,
                'completed_source_inputs':True})
    features,predictions,metrics,shift,dialogues,cv=[pd.concat(x,ignore_index=True) for x in output]
    filenames=['feature_matrix.csv','oof_predictions.csv','per_seed_metrics.csv','correct_wrong_adjustment.csv','per_dialogue_results.csv','inner_cv_results.csv']
    for name,df in zip(filenames,[features,predictions,metrics,shift,dialogues,cv]): df.to_csv(OUT/name,index=False)
    aggregate,selected,off=summaries(metrics,all_parameters)
    aggregate.to_csv(OUT/'aggregate_metrics.csv',index=False)
    selected.to_csv(OUT/'selected_penalties.csv',index=False)
    off.to_csv(OUT/'behavior_off_folds.csv',index=False)
    bootstrap,draws=shared_bootstrap(predictions)
    bootstrap.to_csv(OUT/'bootstrap_results.csv',index=False)
    for draw in draws: write_json(OUT/f'provenance/bootstrap_draws_{draw["dataset"]}.json',draw)
    reports(metrics,aggregate,bootstrap,off)
    assert all(sha(p)==h for p,h in sources.items()),'Input source changed during analysis'
    assert all(sha(p)==h for p,h in code.items()),'Implementation changed during analysis'
    write_json(OUT/'verification.json',{'status':'completed','primary_method':'E1_DA','cpu_only':True,
        'ten_inputs_complete':True,'one_training_record_per_utterance':True,'temperature_scaling':False,
        'cmerc_training':False,'test_labels_used':False,'source_hashes_unchanged':True,'running_code_unchanged':True,
        'elapsed_seconds':time.time()-started,'datasets_and_seeds':checks,'metric_rows':len(metrics),
        'prediction_rows':len(predictions),'feature_rows':len(features),'independent_verification_pending':True})
    print(aggregate[['dataset','method']+[m+'_mean' for m in METRICS]].to_string(index=False),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
