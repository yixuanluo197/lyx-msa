"""Engineering tests with synthetic rows, never experiment outcome selection."""
import json
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from mainline_core import FEATURES, PENALTIES, fit_model, fit_outer, score, select_penalty


def fixture():
    rng=np.random.default_rng(81425); n=180
    f=pd.DataFrame({'confidence':rng.uniform(.2,.98,n),'dialogue_id':np.repeat([f'd{i}' for i in range(9)],20)})
    for col in FEATURES['M_joint']: f[col]=rng.normal(size=n)*.01
    f['correct']=rng.binomial(1,expit(-.2+f.confidence+12*f.gaussian_delta))
    f['label']=f.correct; f['prediction']=1
    return f


class MainlineEngineering(unittest.TestCase):
    def test_scaling_regularization_equivalence(self):
        f=fixture(); ids=np.arange(140); features=FEATURES['M_joint']; lam=.1
        fit=fit_model(f,ids,features,lam)
        train=f.iloc[ids]; C=train.confidence.to_numpy(float)
        z=(train[features].to_numpy()-fit['scaler_mean'])/fit['scaler_scale']
        raw=np.array(fit['raw_coefficients']); w=np.array(fit['effective_behavior_coefficients']); n=len(ids)
        design=np.column_stack([C,z/np.sqrt(n*lam)])
        R=fit['intercept']+design@raw
        q=expit(R); y=train.correct.to_numpy()
        objective_design=np.mean(np.logaddexp(0,R)-y*R)+np.sum(raw**2)/(2*n)
        objective_effective=np.mean(np.logaddexp(0,R)-y*R)+raw[0]**2/(2*n)+lam*np.sum(w**2)/2
        self.assertAlmostEqual(objective_design,objective_effective,places=14)
        gradient=np.r_[np.mean(q-y),design.T@(q-y)/n+raw/n]
        self.assertLess(np.max(np.abs(gradient)),2e-6)

    def test_training_only_scaler(self):
        f=fixture(); ids=np.arange(140); other=f.copy(); other.loc[140:,FEATURES['M_joint']]=1e8
        a=fit_model(f,ids,FEATURES['M_joint'],.1); b=fit_model(other,ids,FEATURES['M_joint'],.1)
        self.assertEqual(a,b)
        np.testing.assert_allclose(a['scaler_mean'],f.iloc[ids][FEATURES['M_joint']].mean())

    def test_score_decomposition_and_bidirectionality(self):
        f=fixture(); fitted=fit_model(f,np.arange(140),FEATURES['M_joint'],.01)
        values=score(f.iloc[140:],fitted)
        np.testing.assert_allclose(values['reliability_score'],values['base_score']+values['behavior_score'],atol=1e-15)
        np.testing.assert_allclose(values['oof_confidence'],expit(values['reliability_score']),atol=1e-15)
        # Explicit admissible parameters prove the module can increase or decrease C.
        explicit=dict(fitted,intercept=0.,confidence_coefficient=0.,effective_behavior_coefficients=[0.]*4)
        q=score(f,explicit)['oof_confidence']; self.assertTrue((q>f.confidence).any() and (q<f.confidence).any())

    def test_off_exact_and_tie_order(self):
        f=fixture(); ids=np.arange(140); base=fit_model(f,ids,[],'off'); values=score(f,base)
        self.assertEqual(base['features'],[])
        self.assertTrue(np.array_equal(values['behavior_score'],np.zeros(len(f))))
        self.assertEqual(select_penalty({str(p):.2 for p in PENALTIES}),'off')
        loss={'off':.3,'1.0':.2,'0.1':.2,'0.01':.2}; self.assertEqual(select_penalty(loss),1.)
        loss['1.0']=.2+5e-13; self.assertEqual(select_penalty(loss),1.)
        # Constant behavior makes active candidates mathematically identical to E1.
        for col in FEATURES['M_joint']: f[col]=0.
        outputs,params=fit_outer(f,np.arange(160),np.arange(160,180),['M_joint'])
        self.assertEqual(params['methods']['M_joint']['selected_penalty'],'off')
        self.assertTrue(np.array_equal(outputs['M_joint']['oof_confidence'],outputs['E1']['oof_confidence']))

    def test_dialogue_group_isolation(self):
        f=fixture(); train=np.arange(160); held=np.arange(160,180)
        outputs,params=fit_outer(f,train,held,['M_mu'])
        self.assertFalse(set(params['train_dialogues'])&set(params['held_dialogues']))
        all_held=[]
        for fold in params['inner_folds']:
            self.assertFalse(set(fold['train_dialogues'])&set(fold['held_dialogues']))
            self.assertTrue(set(fold['train_indices'])<=set(train))
            self.assertTrue(set(fold['held_indices'])<=set(train))
            all_held.extend(fold['held_indices'])
        self.assertEqual(sorted(all_held),list(train))

    def test_classifications_and_inputs_unchanged(self):
        f=fixture(); before=f.copy(deep=True)
        params=fit_model(f,np.arange(140),FEATURES['M_joint'],1.)
        result=score(f,params)
        pd.testing.assert_frame_equal(f,before)
        np.testing.assert_array_equal(f.prediction,before.prediction)
        self.assertTrue(np.isfinite(result['oof_confidence']).all())
        self.assertEqual(len(result['oof_confidence']),len(f))
        self.assertEqual(set(params['features']),set(FEATURES['M_joint']))


if __name__=='__main__': unittest.main(verbosity=2)
