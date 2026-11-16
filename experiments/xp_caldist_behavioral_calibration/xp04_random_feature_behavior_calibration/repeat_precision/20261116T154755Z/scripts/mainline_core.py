"""Frozen mainline: raw confidence and original CaliDist mu/delta only."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

PENALTIES = ['off', 1., .1, .01]
FEATURES = {
    'M_joint': ['gaussian_mu','gaussian_delta','dropout_mu','dropout_delta'],
    'M_mu': ['gaussian_mu','dropout_mu'],
    'M_delta': ['gaussian_delta','dropout_delta'],
    'M_gaussian': ['gaussian_mu','gaussian_delta'],
    'M_dropout': ['dropout_mu','dropout_delta'],
}
METRICS = ['ece15','binary_brier','binary_nll']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def fit_model(frame, indices, features, penalty):
    """One fit: all transformations fitted only on supplied training rows."""
    indices=np.asarray(indices,dtype=int)
    train=frame.iloc[indices]
    n=len(train)
    C=train.confidence.to_numpy(float)[:,None]
    if penalty == 'off':
        assert not features
        mean=[]; scale=[]; multiplier=1.; design=C
    else:
        assert float(penalty)>0 and features
        scaler=StandardScaler().fit(train[features].to_numpy(float))
        mean=scaler.mean_.tolist(); scale=scaler.scale_.tolist()
        multiplier=float(np.sqrt(n*float(penalty)))
        design=np.column_stack([C,scaler.transform(train[features].to_numpy(float))/multiplier])
    model=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=2000,tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(design,train.correct.to_numpy(int))
    coeff=model.coef_[0]
    return {'features':list(features),'penalty':penalty,'n_train':n,
        'train_indices':indices.tolist(),'train_dialogues':sorted(train.dialogue_id.unique().tolist()),
        'scaler_mean':mean,'scaler_scale':scale,'design_divisor':multiplier,
        'raw_coefficients':coeff.tolist(),'intercept':float(model.intercept_[0]),
        'confidence_coefficient':float(coeff[0]),
        'effective_behavior_coefficients':(coeff[1:]/multiplier).tolist(),
        'iterations':int(model.n_iter_[0]),'optimizer':{'C':1.,'solver':'lbfgs','max_iter':2000,'tol':1e-8}}


def score(frame, fitted):
    base=fitted['intercept']+fitted['confidence_coefficient']*frame.confidence.to_numpy(float)
    if fitted['features']:
        Z=(frame[fitted['features']].to_numpy(float)-fitted['scaler_mean'])/fitted['scaler_scale']
        behavior=Z@np.asarray(fitted['effective_behavior_coefficients'])
    else:
        behavior=np.zeros(len(frame))
    R=base+behavior
    return {'base_score':base,'behavior_score':behavior,'reliability_score':R,'oof_confidence':expit(R)}


def select_penalty(losses):
    best=PENALTIES[0]
    for penalty in PENALTIES[1:]:
        if losses[str(penalty)] < losses[str(best)]-1e-12:
            best=penalty
    return best


def fit_outer(frame, train, held, methods):
    """Cache E1 per training subset; all off paths reuse identical fitted object."""
    train=np.asarray(train,dtype=int); held=np.asarray(held,dtype=int)
    assert not set(frame.iloc[train].dialogue_id)&set(frame.iloc[held].dialogue_id)
    cache={}
    def base(indices):
        key=tuple(int(i) for i in indices)
        if key not in cache: cache[key]=fit_model(frame,indices,[],'off')
        return cache[key]
    outer_base=base(train)
    result={'train_indices':train.tolist(),'held_indices':held.tolist(),
        'train_dialogues':sorted(frame.iloc[train].dialogue_id.unique().tolist()),
        'held_dialogues':sorted(frame.iloc[held].dialogue_id.unique().tolist()),
        'E1':outer_base,'methods':{}}
    inner=[]
    for k,(a,b) in enumerate(GroupKFold(5).split(train,groups=frame.iloc[train].dialogue_id)):
        it,iv=train[a],train[b]
        assert not set(frame.iloc[it].dialogue_id)&set(frame.iloc[iv].dialogue_id)
        inner.append({'inner_fold':k,'train_indices':it.tolist(),'held_indices':iv.tolist(),
            'train_dialogues':sorted(frame.iloc[it].dialogue_id.unique().tolist()),
            'held_dialogues':sorted(frame.iloc[iv].dialogue_id.unique().tolist()),'E1':base(it)})
    result['inner_folds']=inner
    outputs={'E1':score(frame.iloc[held],outer_base)}
    for method in methods:
        features=FEATURES[method]
        candidate_losses={}; candidate_fits={}
        for penalty in PENALTIES:
            squared=[]; fits=[]
            for fold in inner:
                fitted=fold['E1'] if penalty=='off' else fit_model(frame,fold['train_indices'],features,penalty)
                check=frame.iloc[fold['held_indices']]
                q=score(check,fitted)['oof_confidence']
                squared.extend(((q-check.correct.to_numpy(int))**2).tolist())
                fits.append({'inner_fold':fold['inner_fold'],'fit':fitted,
                    'held_indices':fold['held_indices'],'held_dialogues':fold['held_dialogues'],
                    'held_brier':float(np.mean((q-check.correct.to_numpy(int))**2))})
            candidate_losses[str(penalty)]=float(np.mean(squared))
            candidate_fits[str(penalty)]=fits
        selected=select_penalty(candidate_losses)
        fitted=outer_base if selected=='off' else fit_model(frame,train,features,selected)
        result['methods'][method]={'features':features,'selected_penalty':selected,
            'candidate_brier':candidate_losses,'inner_candidate_fits':candidate_fits,'outer_fit':fitted}
        outputs[method]=score(frame.iloc[held],fitted)
        if selected=='off':
            assert np.array_equal(outputs[method]['oof_confidence'],outputs['E1']['oof_confidence'])
    return outputs,result


def calibration_metrics(p,y,weights=None):
    p=np.asarray(p,float); y=np.asarray(y,int)
    w=np.ones(len(p)) if weights is None else np.asarray(weights,float)
    assert len(p)==len(y)==len(w) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    bins=np.minimum(np.floor(p*15).astype(int),14)
    residual=np.bincount(bins,weights=w*(p-y),minlength=15)
    clip=np.clip(p,1e-15,1-1e-15)
    return {'ece15':float(np.abs(residual).sum()/w.sum()),
        'binary_brier':float(np.average((p-y)**2,weights=w)),
        'binary_nll':float(np.average(-y*np.log(clip)-(1-y)*np.log1p(-clip),weights=w))}


def metrics(frame,p):
    y=frame.correct.to_numpy(int); raw=frame.confidence.to_numpy(float)
    r=calibration_metrics(p,y)
    r.update({'utterances':len(frame),'dialogues':frame.dialogue_id.nunique(),
        'accuracy':float(accuracy_score(frame.label,frame.prediction)),
        'weighted_f1':float(f1_score(frame.label,frame.prediction,average='weighted')),
        'macro_f1':float(f1_score(frame.label,frame.prediction,average='macro')),
        'correctness_auc':float(roc_auc_score(y,p)),
        'mean_confidence':float(np.mean(p)),
        'correct_mean_confidence':float(np.mean(p[y==1])),
        'wrong_mean_confidence':float(np.mean(p[y==0])),
        'raised_fraction':float(np.mean(p>raw+1e-12)),
        'lowered_fraction':float(np.mean(p<raw-1e-12))})
    return r


def markdown_table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)
